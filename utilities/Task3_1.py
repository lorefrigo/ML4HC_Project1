from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from utilities.models import SwiGLUEncoderLayer


PoolingMode = Literal["last", "mean"]
EncoderType = Literal["lstm", "transformer"]


@dataclass
class AugmentConfig:
    jitter_std: float = 0.02
    scaling_std: float = 0.1
    time_dropout_prob: float = 0.1
    feature_dropout_prob: float = 0.1


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inv_mask = (~mask).float().unsqueeze(-1)
    summed = (x * inv_mask).sum(dim=1)
    denom = inv_mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def _gather_last(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = (~mask).sum(dim=1).clamp(min=1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(2))
    return x.gather(1, idx).squeeze(1)


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        pooling: PoolingMode = "last",
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.pooling = pooling

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        if self.pooling == "last":
            return _gather_last(outputs, mask)
        if self.pooling == "mean":
            return _masked_mean(outputs, mask)
        raise ValueError(f"Unknown pooling: {self.pooling}")


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 1,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: PoolingMode = "mean",
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = SwiGLUEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        if self.pooling == "last":
            return _gather_last(x, mask)
        if self.pooling == "mean":
            return _masked_mean(x, mask)
        raise ValueError(f"Unknown pooling: {self.pooling}")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent (InfoNCE) loss using in-batch negatives.
    """
    if z1.size(0) < 2:
        raise ValueError("Batch size must be >= 2 for InfoNCE.")

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9

    n = z1.size(0)
    targets = torch.arange(n, device=z.device)
    targets = torch.cat([targets + n, targets], dim=0)

    return F.cross_entropy(sim, targets)


def _apply_augmentations(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    """
    x: (T, F) single sample tensor.
    """
    out = x.clone()

    if cfg.jitter_std > 0:
        out = out + torch.randn_like(out) * cfg.jitter_std

    if cfg.scaling_std > 0:
        scale = torch.randn(1, 1, device=out.device) * cfg.scaling_std + 1.0
        out = out * scale

    if cfg.time_dropout_prob > 0:
        drop_t = torch.rand(out.size(0), device=out.device) < cfg.time_dropout_prob
        out[drop_t] = 0.0

    if cfg.feature_dropout_prob > 0:
        drop_f = torch.rand(out.size(1), device=out.device) < cfg.feature_dropout_prob
        out[:, drop_f] = 0.0

    return out


class ContrastiveTimeSeriesDataset(Dataset):
    def __init__(self, x: torch.Tensor, cfg: AugmentConfig) -> None:
        self.x = x
        self.cfg = cfg

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.x[idx]
        v1 = _apply_augmentations(sample, self.cfg)
        v2 = _apply_augmentations(sample, self.cfg)
        return v1, v2


def dataframe_to_tensor(df: pd.DataFrame, feature_cols: list[str]) -> torch.Tensor:
    df = df.sort_values(["PatientID", "Timestamp"])
    num_patients = df["PatientID"].nunique()
    x = df[feature_cols].values.reshape(num_patients, 49, len(feature_cols))
    return torch.tensor(x, dtype=torch.float32)


def create_ssl_dataloader(
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int = 64,
    shuffle: bool = True,
    augment_cfg: AugmentConfig | None = None,
) -> DataLoader:
    x = dataframe_to_tensor(df, feature_cols)
    cfg = augment_cfg or AugmentConfig()
    ds = ContrastiveTimeSeriesDataset(x, cfg)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def create_vae_dataloader(
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    x = dataframe_to_tensor(df, feature_cols)
    ds = TensorDataset(x)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_encoder(
    encoder_type: EncoderType,
    input_size: int,
    pooling: PoolingMode = "mean",
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.1,
    d_model: int = 64,
    nhead: int = 1,
    dim_feedforward: int = 128,
) -> nn.Module:
    if encoder_type == "lstm":
        return LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
        )
    if encoder_type == "transformer":
        return TransformerEncoder(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pooling=pooling,
        )
    raise ValueError(f"Unknown encoder type: {encoder_type}")


class TimeSeriesVAE(nn.Module):
    def __init__(
        self,
        encoder_type: EncoderType,
        input_size: int,
        seq_len: int,
        pooling: PoolingMode = "mean",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        d_model: int = 64,
        nhead: int = 1,
        dim_feedforward: int = 128,
        latent_dim: int = 32,
        decoder_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            encoder_type=encoder_type,
            input_size=input_size,
            pooling=pooling,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        enc_out_dim = hidden_size if encoder_type == "lstm" else d_model
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, seq_len * input_size),
        )
        self.seq_len = seq_len
        self.input_size = input_size

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x, mask)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out.view(-1, self.seq_len, self.input_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def pretrain_infonce(
    encoder: nn.Module,
    projection: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
) -> dict[str, list[float]]:
    encoder.to(device)
    projection.to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(projection.parameters()), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"loss": []}
    for _ in tqdm(range(epochs), desc="InfoNCE epochs"):
        encoder.train()
        projection.train()
        epoch_loss = 0.0

        batch_bar = tqdm(dataloader, desc="InfoNCE batches", leave=False)
        for v1, v2 in batch_bar:
            v1 = v1.to(device)
            v2 = v2.to(device)

            mask = torch.zeros(v1.size(0), v1.size(1), dtype=torch.bool, device=device)

            z1 = projection(encoder(v1, mask))
            z2 = projection(encoder(v2, mask))
            loss = nt_xent_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(projection.parameters()), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        history["loss"].append(epoch_loss / len(dataloader))

    return history


def pretrain_vae(
    vae: TimeSeriesVAE,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    beta: float = 1.0,
) -> dict[str, list[float]]:
    vae.to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}
    for _ in tqdm(range(epochs), desc="VAE epochs"):
        vae.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        batch_bar = tqdm(dataloader, desc="VAE batches", leave=False)
        for (x,) in batch_bar:
            x = x.to(device)
            mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=device)

            recon, mu, logvar = vae(x, mask)
            recon_loss = F.mse_loss(recon, x, reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}", kl=f"{kl.item():.4f}")

        history["loss"].append(epoch_loss / len(dataloader))
        history["recon"].append(epoch_recon / len(dataloader))
        history["kl"].append(epoch_kl / len(dataloader))

    return history


@torch.no_grad()
def compute_embeddings(
    encoder: nn.Module,
    df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
) -> np.ndarray:
    encoder.eval()
    x = dataframe_to_tensor(df, feature_cols).to(device)
    mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=device)
    embeddings = encoder(x, mask)
    return embeddings.cpu().numpy()


@torch.no_grad()
def compute_vae_embeddings(
    vae: TimeSeriesVAE,
    df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
) -> np.ndarray:
    vae.eval()
    x = dataframe_to_tensor(df, feature_cols).to(device)
    mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=device)
    mu, _ = vae.encode(x, mask)
    return mu.cpu().numpy()


def _safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return float("nan")


def evaluate_linear_probe(
    embeddings_train: np.ndarray,
    labels_train: np.ndarray,
    embeddings_val: np.ndarray,
    labels_val: np.ndarray,
    embeddings_test: np.ndarray,
    labels_test: np.ndarray,
    c: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(embeddings_train)
    x_val = scaler.transform(embeddings_val)
    x_test = scaler.transform(embeddings_test)

    clf = LogisticRegression(
        C=c,
        max_iter=max_iter,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    clf.fit(x_train, labels_train)

    val_scores = clf.predict_proba(x_val)[:, 1]
    test_scores = clf.predict_proba(x_test)[:, 1]

    return {
        "val": {
            "auroc": _safe_metric(roc_auc_score, labels_val, val_scores),
            "auprc": _safe_metric(average_precision_score, labels_val, val_scores),
        },
        "test": {
            "auroc": _safe_metric(roc_auc_score, labels_test, test_scores),
            "auprc": _safe_metric(average_precision_score, labels_test, test_scores),
        },
    }

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from typing import Literal

PoolingMode = Literal["last", "mean"]

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes mean pooling across time while ignoring padded positions.

    Args:
        x:    (B, T, D) tensor of sequence outputs.
        mask: (B, T) boolean mask where True indicates padding.
    """
    inv_mask = (~mask).float().unsqueeze(-1)
    summed = (x * inv_mask).sum(dim=1)
    denom = inv_mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def _gather_last(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Selects the last valid timestep per sequence.
    """
    lengths = (~mask).sum(dim=1).clamp(min=1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(2))
    return x.gather(1, idx).squeeze(1)


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for dense time-grid data.

    Input: (B, T, F) with an optional padding mask (B, T) where True indicates padding.
    Output: logits of shape (B, 1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        pooling: PoolingMode = "last",
        use_packed: bool = True,
    ) -> None:
        super().__init__()

        self.pooling = pooling
        self.use_packed = use_packed

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        if self.use_packed:
            lengths = (~mask).sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            outputs, _ = self.lstm(x)

        if self.pooling == "last":
            pooled = _gather_last(outputs, mask)
        elif self.pooling == "mean":
            pooled = _masked_mean(outputs, mask)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

        return self.classifier(pooled)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class BiLSTMClassifier(LSTMClassifier):
    """
    Convenience wrapper for a bidirectional LSTM classifier.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        pooling: PoolingMode = "last",
        use_packed: bool = True,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            pooling=pooling,
            use_packed=use_packed,
        )

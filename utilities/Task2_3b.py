from __future__ import annotations

import pandas as pd
import numpy as np
import json
import os
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from datetime import datetime

from typing import List, Tuple, Dict, Any, Union
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, average_precision_score

# Define constants and feature lists 
T_MAX = 2880.0  # 48 hours in minutes
STATIC_VARS = ['Age', 'Height', 'Weight', 'Gender']
DYNAMIC_VARS = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', 
    'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 
    'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 
    'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 
    'Urine', 'WBC'
]
ALL_FEATURES = sorted(list(set(DYNAMIC_VARS + STATIC_VARS))) # 41 Categories



###############
def set_seed(seed: int) -> None:
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_raw_to_grid(data_dir: str, outcomes_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rapidly loads and merges raw PhysioNet patient text files into a combined DataFrame grid natively using vectorized string operations and a single aggregated matrix pivot.
    """
    outcomes = pd.read_csv(outcomes_file)
    df_list = []
    
    for pid in outcomes['RecordID']:
        file_path = os.path.join(data_dir, f"{pid}.txt")
        temp_df = pd.read_csv(file_path)
        temp_df['RecordID'] = pid
        df_list.append(temp_df)
        
    master_df = pd.concat(df_list, ignore_index=True)
    
    time_parts = master_df['Time'].str.split(':', expand=True).astype(int)
    master_df['Minutes'] = time_parts[0] * 60 + time_parts[1]
    
    master_df = master_df[master_df['Parameter'].isin(ALL_FEATURES)]
    
    final_grid = master_df.pivot_table(
        index=['RecordID', 'Minutes'], 
        columns='Parameter', 
        values='Value'
    ).reset_index()
    
    return final_grid, outcomes[['RecordID', 'In-hospital_death']]

######################################
def setup_run_directory(base_save_dir: str | Path) -> tuple[Path, str]:
    """
    Creates a unique timestamped directory for the current run.
    
    Returns:
        tuple: (Path object for the directory, string name of the run)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    save_dir = Path(base_save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    return save_dir, run_name

######################################

def get_sparsity_mask(df: pd.DataFrame, static_vars: List[str]) -> pd.DataFrame:
    """Creates a boolean mask to restore sparsity while keeping static vars on the first row."""
    mask = df.notna()
    first_rows = df.groupby('PatientID').cumcount() == 0
    for col in static_vars:
        if col in mask.columns:
            mask.loc[first_rows, col] = True
    return mask

def dataframe_to_triplets(df: pd.DataFrame, feature_list: List[str]) -> List[List[Tuple[Union[int, float], int, float]]]:
    """Converts a processed grid back into a list of (t, z, v) sequences per patient."""
    z_map = {feat: i for i, feat in enumerate(feature_list)}
    
    melted = df.melt(
        id_vars=['PatientID', 'Timestamp'], 
        value_vars=feature_list, 
        var_name='Feature', 
        value_name='Value'
    )
    
    melted = melted.dropna(subset=['Value'])
    melted['FeatureID'] = melted['Feature'].map(z_map)
    melted = melted.sort_values(by=['PatientID', 'Timestamp'])
    
    dataset = []
    for pid, group in melted.groupby('PatientID', sort=False):
        dataset.append(list(group[['Timestamp', 'FeatureID', 'Value']].itertuples(index=False, name=None)))

    print(f"Converted DataFrame to {len(dataset)} patient sequences of triplets.")
    return dataset

######################################
class PhysioNetDataset(Dataset):
    def __init__(self, triplets: List[List[Tuple[Union[int, float], int, float]]], labels: pd.DataFrame) -> None:
        self.triplets = triplets
        self.labels = labels.sort_values('RecordID')['In-hospital_death'].values

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sequences are securely pre-sorted by time during dataframe_to_triplets
        seq = self.triplets[idx]
        t, z, v = zip(*seq)
        return (torch.tensor(t, dtype=torch.float32), 
                torch.tensor(z, dtype=torch.long), 
                torch.tensor(v, dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.float32))

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Handles dynamic padding and generates the attention mask."""
    t, z, v, labels = zip(*batch)
    
    t_pad = pad_sequence(t, batch_first=True, padding_value=-1)
    z_pad = pad_sequence(z, batch_first=True, padding_value=0)
    v_pad = pad_sequence(v, batch_first=True, padding_value=0)
    
    # Mask: True for padding positions (time token is -1)
    mask = (t_pad == -1)
    
    return t_pad, z_pad, v_pad, torch.stack(labels), mask

def clip_feature_values(
    df: pd.DataFrame, 
    features: list[str], 
    min_val: float = -5.0, 
    max_val: float = 5.0
) -> pd.DataFrame:
    """
    Clips specific numerical features within a user-defined range.
    
    Args:
        df: The input DataFrame (e.g., scaled_a).
        features: List of column names to apply clipping to.
        min_val: The lower bound (default -5.0).
        max_val: The upper bound (default 5.0).
        
    Returns:
        A copy of the DataFrame with clipped values.
    """
    df_clipped = df.copy()
    
    # Apply clipping only to the specified feature columns
    df_clipped[features] = df_clipped[features].clip(lower=min_val, upper=max_val)
    
    print(f"Data clipped to range: [{min_val}, {max_val}] for {len(features)} features.")
    return df_clipped


################################
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_time: int, T: float = 2880.0, tau: float = 100.0) -> None:
        super().__init__()
        self.d_time = d_time
        self.T = T
        self.tau = tau

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        I_{2k}(t) := sin( t / (T^(2k / tau)) ) [cite: 124]
        I_{2k+1}(t) := cos( t / (T^(2k / tau)) ) [cite: 124]
        """
        device = t.device
        half_dim = self.d_time // 2
        k = torch.arange(half_dim, device=device).float()
        denominators = torch.pow(self.T, (2 * k) / self.tau)
        
        args = t.unsqueeze(-1) / denominators.view(1, 1, -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class TripletEmbedding(nn.Module):
    def __init__(self, d_model: int, d_time: int, num_categories: int = 41) -> None:
        super().__init__()
        self.time_enc = SinusoidalTimeEmbedding(d_time)
        # Input: d_time + 41 (one-hot) + 1 (value) -> Output: d_model
        self.projection = nn.Linear(d_time + num_categories + 1, d_model)

    def forward(self, t: torch.Tensor, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_enc(t)
        z_onehot = F.one_hot(z, num_classes=41).float()
        v_val = v.unsqueeze(-1)
        
        combined = torch.cat([t_emb, z_onehot, v_val], dim=-1)
        return self.projection(combined)
################

class SwiGLUEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer using Pre-LN and a SwiGLU FeedForward Network.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # SwiGLU components
        self.w1 = nn.Linear(d_model, dim_feedforward)
        self.w2 = nn.Linear(d_model, dim_feedforward)
        self.w3 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        # Pre-LN Self-Attention
        src_norm = self.norm1(src)
        attn_out, _ = self.self_attn(src_norm, src_norm, src_norm, key_padding_mask=src_key_padding_mask, need_weights=False)
        src = src + self.dropout(attn_out)
        
        # Pre-LN SwiGLU FFN
        src_norm = self.norm2(src)
        gate = F.silu(self.w1(src_norm))
        ffn_inner = gate * self.w2(src_norm)
        ffn_out = self.w3(self.dropout(ffn_inner))
        
        src = src + self.dropout(ffn_out)
        return src

class TransformerModel(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        num_layers: int, 
        d_time_emb: int, 
        dim_feedforward: int = 128, 
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.embedding = TripletEmbedding(d_model, d_time_emb)
        
        encoder_layer = SwiGLUEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, t: torch.Tensor, z: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Generate initial embeddings from (t, z, v) triplets
        x = self.embedding(t, z, v)
        
        # Process sequence with self-attention 
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Masked Global Average Pooling
        mask_expanded = mask.unsqueeze(-1).float() 
        inv_mask = 1.0 - mask_expanded 
        
        # Cast to float32 BEFORE summing to prevent fp16 numerical overflow on long patient sequences
        sum_x = torch.sum(x.to(torch.float32) * inv_mask, dim=1)
        count_x = torch.clamp(torch.sum(inv_mask, dim=1), min=1e-9)
        pooled_x = (sum_x / count_x).to(x.dtype)
        
        return self.classifier(pooled_x)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)
    
################
def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
) -> Dict[str, float]:
    """
    Trains the model for one epoch on processed dataset.
    
    Args:
        model: The Transformer model with TripletEmbedding.
        dataloader: DataLoader for Set A.
        optimizer: PyTorch optimizer (e.g., Adam).
        device: 'cuda' or 'cpu'.        
    Returns:
        dict: Average loss, AuROC, and AuPRC for the epoch.
    """
    model.train()
    train_loss = 0
    all_targets = []
    all_preds = []

    for t_pad, z_pad, v_pad, labels, mask in dataloader:
        # Move tensors to device
        t_pad, z_pad, v_pad = t_pad.to(device), z_pad.to(device), v_pad.to(device)
        labels, mask = labels.to(device), mask.to(device)

        optimizer.zero_grad()

        # Forward pass: Pass triplets and the padding mask to the model 
        # The mask ensures the Transformer ignores padded 'empty' triplets
        logits = model(t_pad, z_pad, v_pad, mask)
        loss = model.compute_loss(logits.view(-1), labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track statistics
        train_loss += loss.item()
        all_targets.append(labels.detach().cpu())
        # Convert logits to probabilities for metric calculation 
        all_preds.append(torch.sigmoid(logits).detach().cpu())

    # Concatenate all batches
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # Calculate avg loss and auroc, auprc
    avg_loss = train_loss / len(dataloader)
    auroc = roc_auc_score(all_targets, all_preds) 
    auprc = average_precision_score(all_targets, all_preds) 

    return {
        "loss": avg_loss,
        "auroc": auroc,
        "auprc": auprc
    }

##############################
@torch.no_grad()
def evaluate(
    model: TransformerModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    eval_loss = 0.0
    all_targets = []
    all_preds = []

    for t_pad, z_pad, v_pad, labels, mask in dataloader:
        # Move tensors to device
        t_pad, z_pad, v_pad = t_pad.to(device), z_pad.to(device), v_pad.to(device)
        labels, mask = labels.to(device), mask.to(device)

        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(t_pad, z_pad, v_pad, mask)
            # Compute loss
            loss = model.compute_loss(logits.view(-1), labels)

        # Track statistics
        eval_loss += loss.item()
        all_targets.append(labels.detach().cpu())

        # Convert logits to probabilities for metric calculation 
        all_preds.append(torch.sigmoid(logits).detach().cpu())

    # Concatenate all batches
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # Calculate avg loss and auroc, auprc
    avg_loss = eval_loss / len(dataloader)
    auroc = roc_auc_score(all_targets, all_preds) 
    auprc = average_precision_score(all_targets, all_preds) 

    return {
        "loss": avg_loss,
        "auroc": auroc,
        "auprc": auprc
    }

#######################
def save_checkpoint(
    state_dict: dict[str, torch.Tensor],
    history: dict[str, list[dict[str, float]]],
    save_dir: str,
    model_name: str = "best_model.pt"
) -> None:
    """
    Saves the model state and training history to the specified directory.
    """
    checkpoint_dir = Path(save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model weights
    torch.save(state_dict, checkpoint_dir / model_name)
    
    # Save the training history as a formatted JSON
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"  >> Checkpoint synced to: {checkpoint_dir}")

#############

def run_training_pipeline(
    model: TransformerModel,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: str = "model_checkpoints/task_2_3b",
    epochs: int = 20
) -> dict[str, Any]:
    """
    Core training loop. Orchestrates training, evaluation, and checkpointing.
    """
    # Initialize the run directory
    save_dir, run_name = setup_run_directory(save_dir)
    
    #scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_auprc = 0.0
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        # Train and Validate
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        # Update History
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"Epoch {epoch:02d} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"Val AuROC: {val_metrics['auroc']:.4f} | Val AuPRC: {val_metrics['auprc']:.4f}")

        # Checkpointing 
        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            
            # save best model and history
            save_checkpoint(
                state_dict=model.state_dict(),
                history=history,
                save_dir=save_dir
            )
            print(f"  * Best model updated (AuPRC: {best_val_auprc:.4f})")

    return {
        "best_val_auprc": best_val_auprc,
        "history": history,
        "save_dir": save_dir,
        "run_name": run_name
    }

###############

def smooth_curve(values: list[float], window: int = 3) -> np.ndarray:
    """
    Applies a simple moving average to smooth out noise in the curves.
    """
    if window <= 1:
        return np.array(values)
    
    # Pad the start to maintain the same length as the input
    padded_values = np.pad(values, (window - 1, 0), mode='edge')
    return np.convolve(padded_values, np.ones(window)/window, mode='valid')

def plot_training_results(
    history: dict[str, list[dict[str, float]]], 
    save_path: str | None = None,
    window_size: int = 5
) -> None:
    """
    Plots Training vs. Validation metrics using only smoothed loss curves
    for maximum clarity in the final report.
    """
    epochs = np.arange(1, len(history["train"]) + 1)
    
    # Extract and Smooth Loss
    train_loss_raw = [step["loss"] for step in history["train"]]
    val_loss_raw = [step["loss"] for step in history["val"]]
    
    train_loss_smooth = smooth_curve(train_loss_raw, window=window_size)
    val_loss_smooth = smooth_curve(val_loss_raw, window=window_size)
    
    # Extract Metrics
    train_auroc = [step["auroc"] for step in history["train"]]
    val_auroc = [step["auroc"] for step in history["val"]]
    
    train_auprc = [step["auprc"] for step in history["train"]]
    val_auprc = [step["auprc"] for step in history["val"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Smoothed Loss Plot ---
    axes[0].plot(epochs, train_loss_smooth, color='royalblue', linewidth=2, label='Train Loss')
    axes[0].plot(epochs, val_loss_smooth, color='darkorange', linewidth=2, label='Val Loss')
    
    axes[0].set_title(f'Binary Cross-Entropy Loss (SMA-{window_size})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- AuROC Plot ---
    axes[1].plot(epochs, train_auroc, color='forestgreen', linestyle='--', label='Train AuROC')
    axes[1].plot(epochs, val_auroc, color='forestgreen', linewidth=2, label='Val AuROC')
    axes[1].set_title('Area Under ROC Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AuROC')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # --- AuPRC Plot ---
    axes[2].plot(epochs, train_auprc, color='crimson', linestyle='--', label='Train AuPRC')
    axes[2].plot(epochs, val_auprc, color='crimson', linewidth=2, label='Val AuPRC')
    axes[2].set_title('Area Under PR Curve')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AuPRC')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cleaned plots saved to: {save_path}")
    
    plt.show()

#######################
def run_final_test(
    model: TransformerModel,
    test_loader: DataLoader,
    checkpoint_path: str,
    device: torch.device
) -> dict[str, float]:
    """
    Loads the best weights from a checkpoint and evaluates on the unseen Set C.
    """
    print(f"Loading best model weights from: {checkpoint_path}")
    
    # Load state dictionary
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    print("Starting final evaluation on Set C...")
    
    # Run the evaluation
    test_results = evaluate(model, test_loader, device)
    
    # Format and display the results
    print("\n" + "="*30)
    print("FINAL TEST RESULTS (SET C)")
    print("="*30)
    print(f"AuROC: {test_results['auroc']:.4f}")
    print(f"AuPRC: {test_results['auprc']:.4f}")
    print(f"Loss:  {test_results['loss']:.4f}")
    print("="*30)
    
    return test_results

##############
def report_test_results(results: dict[str, float], model_name: str = "Transformer (Task 2.3b)") -> None:
    """
    Prints a formatted report of the test set performance and generates 
    a LaTeX table row.
    """
    print("\n" + "="*45)
    print(f" FINAL PERFORMANCE REPORT: {model_name}")
    print("="*45)
    
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 28)
    print(f"{'AuROC':<15} | {results['auroc']:.4f}")
    print(f"{'AuPRC':<15} | {results['auprc']:.4f}")
    print(f"{'Test Loss':<15} | {results['loss']:.4f}")
    print("-" * 45)

    # Table Generation
    latex_row = (
        f"{model_name} & {results['auroc']:.4f} & "
        f"{results['auprc']:.4f} & {results['loss']:.4f} \\\\"
    )
    
    print(latex_row)
    print("="*45 + "\n")

#######
def clean_checkpoint_dir(
    base_dir: str | Path, 
    keep_run_name: str | None = None
) -> None:
    """
    Deletes all subdirectories in the base_dir except for the one specified.
    
    Args:
        base_dir: The parent directory (e.g., 'model_checkpoints/task_2_3b/')
        keep_run_name: The name of the folder to preserve (e.g., 'run_20260326_2015')
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Directory {base_dir} does not exist. Nothing to clean.")
        return

    if keep_run_name is None:
        print("⚠️ No 'keep_run_name' provided. Skipping cleanup to prevent total data loss.")
        return

    print(f"Cleaning directory: {base_path}")
    print(f"Keeping only: {keep_run_name}")

    # Iterate through all items in the directory
    deleted_count = 0
    for item in base_path.iterdir():
        # delete directories we don't want to keep
        if item.is_dir():
            if item.name != keep_run_name:
                try:
                    shutil.rmtree(item)
                    print(f"  [Deleted] {item.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  [Error] Could not delete {item.name}: {e}")
            else:
                print(f"  [Preserved] {item.name} (Best Model)")

    print(f"\nCleanup complete. Removed {deleted_count} old run directories.")


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
from utilities.data_processor import DataPreprocessor

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
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
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
        
        sum_x = torch.sum(x * inv_mask, dim=1)
        count_x = torch.clamp(torch.sum(inv_mask, dim=1), min=1e-9)
        pooled_x = sum_x / count_x
        
        return self.classifier(pooled_x)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)
    
################
def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    scaler: Any = None
) -> Dict[str, float]:
    """
    Trains the model for one epoch on processed dataset.
    
    Args:
        model: The Transformer model with TripletEmbedding.
        dataloader: DataLoader for Set A.
        optimizer: PyTorch optimizer (e.g., Adam).
        device: 'cuda' or 'cpu'.
        scaler: GradScaler for mixed precision.
        
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
        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(t_pad, z_pad, v_pad, mask)
            loss = model.compute_loss(logits.view(-1), labels)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_auprc = 0.0
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        # Train and Validate
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
        val_metrics = evaluate(model, val_loader, device)
        
        # Update History
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"Epoch {epoch:02d} | Train Loss: {train_metrics['loss']:.4f} | "
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
        "save_dir": save_dir
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
    a LaTeX table row for the assignment report.
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
    
    print("\n[LaTeX Table Row for your Report]")
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

######################################

# Enforce reproducibility before any execution logic
SEED = 42
set_seed(SEED)

rnd_generator = torch.Generator()
rnd_generator.manual_seed(SEED)

###############

# Initialization for model and hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"run_{timestamp}"
save_directory = f"model_checkpoints/task_2_3b/{run_name}"
epochs = 2

model = TransformerModel(
    d_model=64,
    nhead=4,
    num_layers=3,
    d_time_emb=16,
    dim_feedforward=128,
    dropout=0.1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

######################################
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

##############

raw_a, labels_a = load_raw_to_grid('ml4h_data/p1/set-a/', 'ml4h_data/p1/Outcomes-a.txt')
raw_b, labels_b = load_raw_to_grid('ml4h_data/p1/set-b/', 'ml4h_data/p1/Outcomes-b.txt')
raw_c, labels_c = load_raw_to_grid('ml4h_data/p1/set-c/', 'ml4h_data/p1/Outcomes-c.txt')

# RecordID -> PatientID
# Minutes  -> Timestamp
raw_a = raw_a.rename(columns={'RecordID': 'PatientID', 'Minutes': 'Timestamp'})
raw_b = raw_b.rename(columns={'RecordID': 'PatientID', 'Minutes': 'Timestamp'})
raw_c = raw_c.rename(columns={'RecordID': 'PatientID', 'Minutes': 'Timestamp'})

# Capture Sparsity Masks
mask_a = get_sparsity_mask(raw_a, STATIC_VARS)
mask_b = get_sparsity_mask(raw_b, STATIC_VARS)
mask_c = get_sparsity_mask(raw_c, STATIC_VARS)

# Fit and Transform using DataPreprocessor 
processor = DataPreprocessor(impute_outliers=True, impute_missing=True)
scaled_a = processor.fit_transform(raw_a)
scaled_b = processor.transform(raw_b)
scaled_c = processor.transform(raw_c)

# Restore Sparsity
scaled_a = scaled_a.where(mask_a)
scaled_b = scaled_b.where(mask_b)
scaled_c = scaled_c.where(mask_c)

######################################
# Create Loaders
train_triplets = dataframe_to_triplets(scaled_a, ALL_FEATURES)
train_loader = DataLoader(PhysioNetDataset(train_triplets, labels_a), 
                          batch_size=32, shuffle=True, collate_fn=collate_fn, generator=rnd_generator)

val_triplets = dataframe_to_triplets(scaled_b, ALL_FEATURES)
val_loader = DataLoader(PhysioNetDataset(val_triplets, labels_b), 
                        batch_size=32, shuffle=False, collate_fn=collate_fn)

test_triplets = dataframe_to_triplets(scaled_c, ALL_FEATURES)
test_loader = DataLoader(PhysioNetDataset(test_triplets, labels_c), 
                         batch_size=32, shuffle=False, collate_fn=collate_fn)

######################################
# Train the Model
training_results = run_training_pipeline(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    save_dir=save_directory,
    epochs=epochs
)
current_save_dir = training_results["save_dir"]
current_run_name = training_results["run_name"]

######################################

# Plot Metrics
plot_training_results(training_results["history"], save_path=current_save_dir / "training_curves.png")

######################################

# Test on the Test Set
best_model_path = current_save_dir / "best_model.pt"
test_results = run_final_test(
    model=model,
    test_loader=test_loader,
    checkpoint_path=str(best_model_path),
    device=device
)

######################################

# Report the metric
report_test_results(test_results, model_name="Transformer (Task 2.3b)")

######################################

# Clean checkpoint directory
clean_checkpoint_dir(base_dir="model_checkpoints/task_2_3b", keep_run_name=current_run_name)




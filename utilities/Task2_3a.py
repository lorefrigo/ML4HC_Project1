from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from utilities.preprocessing import clip_feature_values
from utilities.models import SwiGLUEncoderLayer
from utilities.training_functions import run_training_pipeline, run_final_test, set_seed
from utilities.plotting import plot_training_results, report_test_results
from utilities.dir_manager import clean_checkpoint_dir

###### CONSTANTS ######

PREPROCESSED_DIR = "output/preprocessed_imputed_datasets"
CLIP = 5.0
BATCH_SIZE = 64
EPOCHS = 2  

###### CLASSES & FUNCTIONS ######

class SimpleTransformer(nn.Module):
    """
    Transformer classifier for tabular / aggregated time-series inputs.

    Takes a plain (B, T, F) feature tensor — no triplet encoding or
    sinusoidal time embedding — and otherwise mirrors TripletTransformer:

        Linear projection  →  Pre-LN SwiGLU TransformerEncoder  →
        Masked Global Average Pooling  →  Linear classifier head

    Args:
        num_features:    Number of input features per time-step F.
        d_model:         Transformer hidden dimension.
        nhead:           Number of attention heads.
        num_layers:      Number of SwiGLUEncoderLayer blocks.
        dim_feedforward: Inner dim of the SwiGLU FFN (default 128).
        dropout:         Dropout probability (default 0.1).
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # (B, T, F) -> (B, T, d_model)
        self.input_projection = nn.Linear(num_features, d_model)

        encoder_layer = SwiGLUEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T, F)  — batch of padded feature sequences.
            mask: (B, T) bool — True marks padding positions.

        Returns:
            Logits of shape (B, 1).
        """
        x = self.input_projection(x)                                     # (B, T, d_model)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)       # (B, T, d_model)

        # Masked Global Average Pooling
        inv_mask = (1.0 - mask.unsqueeze(-1).float())                    # (B, T, 1)
        sum_x    = torch.sum(x.to(torch.float32) * inv_mask, dim=1)     # (B, d_model)
        count_x  = torch.clamp(inv_mask.sum(dim=1), min=1e-9)           # (B, 1)
        pooled   = (sum_x / count_x).to(x.dtype)                        # (B, d_model)

        return self.classifier(pooled)                                   # (B, 1)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


def check_missing_values(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Reports missing value counts and percentages per column.

    Args:
        df:   Input DataFrame.
        name: Label for the dataset (used in printed output).

    Returns:
        A summary DataFrame with columns ['missing_count', 'missing_pct'],
        filtered to only rows where missing_count > 0.
    """
    total_rows = len(df)
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / total_rows * 100).round(2)
    summary = pd.DataFrame({
        'missing_count': missing_count,
        'missing_pct':   missing_pct
    })
    summary = summary[summary['missing_count'] > 0].sort_values('missing_count', ascending=False)
    if summary.empty:
        print(f"[{name}] No missing values found. ({total_rows} rows)")
    else:
        print(f"[{name}] {len(summary)} columns with missing values (out of {df.shape[1]}):")
        print(summary.to_string())
        print()
    return summary


def verify_patient_rows(df: pd.DataFrame, name: str, expected_rows: int = 49) -> bool:
    """
    Verifies that every patient in the dataset has exactly `expected_rows` rows.

    Args:
        df:            Input DataFrame with a 'PatientID' column.
        name:          Label for the dataset (used in printed output).
        expected_rows: Expected number of rows per patient (default 49).

    Returns:
        True if all patients have exactly `expected_rows` rows, False otherwise.
    """
    counts = df.groupby('PatientID').size()
    bad = counts[counts != expected_rows]

    if bad.empty:
        print(f"[{name}] ✓ All {len(counts)} patients have exactly {expected_rows} rows.")
        return True
    else:
        print(f"[{name}] ✗ {len(bad)} / {len(counts)} patients do NOT have {expected_rows} rows:")
        print(bad.value_counts().sort_index().to_string())
        print()
        return False


def create_dataloader(df: pd.DataFrame, feature_cols: list[str], batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Converts a pandas DataFrame (with exactly 49 rows per patient) into a PyTorch DataLoader.
    """
    # Ensure data is sorted by Patient and Time
    df = df.sort_values(['PatientID', 'Timestamp'])
    
    num_patients = df['PatientID'].nunique()
    expected_rows = 49
    num_features = len(feature_cols)
    
    # Reshape features to (Num_Patients, Time_Steps, Num_Features)
    x_data = df[feature_cols].values.reshape(num_patients, expected_rows, num_features)
    
    # Extract 1 label per patient (preserves the exact sorted order)
    y_data = df.drop_duplicates(subset=['PatientID'])['Label'].values
    
    # Convert arrays to PyTorch Tensors
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)  # shape (B)
    
    # The datasets are completely dense (no padding needed because everyone is 49 hours)
    # We provide a dummy padding mask of all 'False' 
    mask_tensor = torch.zeros((num_patients, expected_rows), dtype=torch.bool)
    
    dataset = TensorDataset(x_tensor, mask_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_and_preprocess_dataset(file_path: str | Path, name: str, clip_val: float):
    # Load the data
    df = pd.read_parquet(file_path)
    
    # Exclude metadata
    exclude = ['PatientID', 'Timestamp', 'Label', 'AgeBin', 'RecordID']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Clipping
    df = clip_feature_values(df, feature_cols, min_val=-clip_val, max_val=clip_val)
    
    # Run validations and reporting
    check_missing_values(df, name)
    verify_patient_rows(df, name)
    
    # Type casting
    df[feature_cols] = df[feature_cols].astype(float)
    
    return df, feature_cols

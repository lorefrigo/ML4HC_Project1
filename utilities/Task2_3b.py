from __future__ import annotations

import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Any, Union
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utilities.models import SwiGLUEncoderLayer
from utilities.preprocessing import DataPreprocessor, clip_feature_values

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

def preprocess_pipeline(
    df: pd.DataFrame, 
    processor: DataPreprocessor, 
    is_train: bool = False, 
    clip_val: float = 5.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Standardized preprocessing pipeline for Task 2.3b.
    Hides renaming, masking, scaling, and clipping logic from the notebook.
    """
    # Rename columns to match pipeline expectations
    df = df.rename(columns={'RecordID': 'PatientID', 'Minutes': 'Timestamp'})
    
    # Capture Sparsity Mask (uses the global STATIC_VARS from Task2_3b.py)
    mask = get_sparsity_mask(df, STATIC_VARS)
    
    # Fit and Transform (train) or Transform only (val/test)
    if is_train:
        scaled_df = processor.fit_transform(df)
    else:
        scaled_df = processor.transform(df)
        
    # Restore original sparsity (True NaNs)
    scaled_df = scaled_df.where(mask)
    
    # Define valid columns and apply clipping
    exclude = ['PatientID', 'Timestamp', 'Label', 'AgeBin', 'RecordID']
    valid_cols = [col for col in scaled_df.columns if col not in exclude]
    scaled_df = clip_feature_values(scaled_df, valid_cols, min_val=-clip_val, max_val=clip_val)
    
    return scaled_df, valid_cols

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

class TripletTransformer(nn.Module):
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
        
        sum_x = torch.sum(x.to(torch.float32) * inv_mask, dim=1)
        count_x = torch.clamp(torch.sum(inv_mask, dim=1), min=1e-9)
        pooled_x = (sum_x / count_x).to(x.dtype)
        
        return self.classifier(pooled_x)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)
    


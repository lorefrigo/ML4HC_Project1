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
from utilities.dir_manager import clean_checkpoint_dir
from utilities.plotting import plot_training_results, report_test_results
from utilities.plotting import plot_training_results
from utilities.training_functions import run_final_test, run_training_pipeline, set_seed
from utilities.models import SwiGLUEncoderLayer, SinusoidalTimeEmbedding, TripletEmbedding, TripletTransformer
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
ALL_FEATURES = sorted(list(set(DYNAMIC_VARS + STATIC_VARS))) 


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
    valid_cols = [str(col) for col in scaled_df.columns if col not in exclude]
    scaled_df = clip_feature_values(scaled_df, valid_cols, min_val=-clip_val, max_val=clip_val)
    
    return scaled_df, valid_cols

    

if __name__ == "__main__":
    # Initialization for model and hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir_task_2_3b = "model_checkpoints/task_2_3b"
    epochs = 4
    CLIP = 5.0
    SEED = 42
    BATCH_SIZE = 32
    set_seed(SEED)
    rnd_generator = torch.Generator().manual_seed(SEED)

    print("\n[STEP 1/5] Initializing TripletTransformer Model...")
    model = TripletTransformer(
        d_model=64,
        nhead=1,
        num_layers=2,
        d_time_emb=4,
        dim_feedforward=176,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=2
        )
    
    # build datasets from files
    print("\n[STEP 2/5] Loading raw PhysiONet data (Sets A, B, and C)...")
    raw_a, labels_a = load_raw_to_grid('ml4h_data/p1/set-a/', 'ml4h_data/p1/Outcomes-a.txt')
    raw_b, labels_b = load_raw_to_grid('ml4h_data/p1/set-b/', 'ml4h_data/p1/Outcomes-b.txt')
    raw_c, labels_c = load_raw_to_grid('ml4h_data/p1/set-c/', 'ml4h_data/p1/Outcomes-c.txt')
    print("      Data sets loaded successfully.")

    # Instantiate DataPreprocessor 
    processor = DataPreprocessor(impute_outliers=True, impute_missing=False, encode_cat=True)

    # preprocess data
    print("\n[STEP 3/5] Preprocessing and scaling data...")
    scaled_a, valid_cols = preprocess_pipeline(raw_a, processor, is_train=True, clip_val=CLIP)
    scaled_b, _ = preprocess_pipeline(raw_b, processor, is_train=False, clip_val=CLIP)
    scaled_c, _ = preprocess_pipeline(raw_c, processor, is_train=False, clip_val=CLIP)
    print(f"      Preprocessing complete. Valid features: {len(valid_cols)}")

    # Create Loaders
    print("\n[STEP 4/5] Converting dataframes to sequences of triplets and creating DataLoaders...")
    train_triplets = dataframe_to_triplets(scaled_a, valid_cols)
    train_loader = DataLoader(PhysioNetDataset(train_triplets, labels_a), 
                            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, generator=rnd_generator)

    val_triplets = dataframe_to_triplets(scaled_b, valid_cols)
    val_loader = DataLoader(PhysioNetDataset(val_triplets, labels_b), 
                            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, generator=rnd_generator)

    test_triplets = dataframe_to_triplets(scaled_c, valid_cols)
    test_loader = DataLoader(PhysioNetDataset(test_triplets, labels_c), 
                            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, generator=rnd_generator)
    
    # Train the Model
    print("\n[STEP 5/5] Starting Training Pipeline...")
    print(f"      Running for {epochs} epochs. This may take a few minutes...")
    training_results = run_training_pipeline(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=base_dir_task_2_3b,
        epochs=epochs,
        scheduler=scheduler,
        data_format="triplet"
    )

    current_save_dir_2_3b = training_results["save_dir"]
    current_run_name_2_3b = training_results["run_name"]

    # Plot Metrics
    plot_training_results(training_results["history"], save_path=current_save_dir_2_3b / "training_curves.png")

    # Test on the Test Set
    best_model_path = current_save_dir_2_3b / "best_model.pt"
    test_results = run_final_test(
        model=model,
        test_loader=test_loader,
        checkpoint_path=str(best_model_path),
        device=device,
        data_format="triplet"
    )

    # Report the metric
    report_test_results(test_results, model_name="Transformer (Task 2.3b)")

    # Clean checkpoint directory
    clean_checkpoint_dir(base_dir=base_dir_task_2_3b, keep_run_name=current_run_name_2_3b)
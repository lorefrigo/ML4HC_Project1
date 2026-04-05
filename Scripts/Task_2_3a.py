from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from utilities.preprocessing import clip_feature_values
from utilities.models import SwiGLUEncoderLayer, SimpleTransformer
from utilities.training_functions import run_training_pipeline, run_final_test, set_seed, LinearDropoutScheduler, StepDropoutScheduler, ConstantDropoutScheduler
from utilities.plotting import plot_training_results, report_test_results
from utilities.dir_manager import clean_checkpoint_dir


###### CLASSES & FUNCTIONS ######



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


def create_dataloader(df: pd.DataFrame, feature_cols: list[str], batch_size: int = 32, shuffle: bool = True, generator: torch.Generator | None = None) -> DataLoader:
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

def load_and_preprocess_dataset(file_path: str | Path, name: str, clip_val: float) -> tuple[pd.DataFrame, list[str]]:
    # Load the data
    df = pd.read_parquet(file_path)
    
    # Exclude metadata
    exclude = ['PatientID', 'Timestamp', 'Label', 'AgeBin', 'RecordID', 'ICUType']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Clipping
    df = clip_feature_values(df, feature_cols, min_val=-clip_val, max_val=clip_val)
    
    # Run validations and reporting
    check_missing_values(df, name)
    verify_patient_rows(df, name)
    
    # Type casting
    df[feature_cols] = df[feature_cols].astype(float)
    
    return df, feature_cols



if __name__ == "__main__":

    # Constants
    PREPROCESSED_DIR = "output/preprocessed_imputed_datasets"
    base_dir_task_2_3a = "model_checkpoints/task_2_3a"
    CLIP = 5.0
    EPOCHS = 100
    BATCH_SIZE = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    set_seed(SEED)
    rnn_generator = torch.Generator()
    rnn_generator.manual_seed(SEED)

    # Load and preprocess datasets
    df_a, FEATURE_COLS = load_and_preprocess_dataset(f"{PREPROCESSED_DIR}/preprocessed_set_a.parquet", "Set A", CLIP)
    df_b, _ = load_and_preprocess_dataset(f"{PREPROCESSED_DIR}/preprocessed_set_b.parquet", "Set B", CLIP)
    df_c, _ = load_and_preprocess_dataset(f"{PREPROCESSED_DIR}/preprocessed_set_c.parquet", "Set C", CLIP)

    # Combine Set A and Set B for a unique split
    df_combined = pd.concat([df_a, df_b], ignore_index=True)
    patient_ids = df_combined['PatientID'].unique()
    
    # Shuffle IDs and split 
    np.random.seed(42)  
    np.random.shuffle(patient_ids)
    split_idx = int(0.7 * len(patient_ids))
    train_ids = patient_ids[:split_idx]
    val_ids   = patient_ids[split_idx:]
    
    df_train = df_combined[df_combined['PatientID'].isin(train_ids)]
    df_val   = df_combined[df_combined['PatientID'].isin(val_ids)]
    
    # Create DataLoaders
    train_loader = create_dataloader(df_train, FEATURE_COLS, batch_size=BATCH_SIZE, shuffle=True, generator=rnn_generator)
    val_loader   = create_dataloader(df_val, FEATURE_COLS, batch_size=BATCH_SIZE, shuffle=False, generator=rnn_generator)
    test_loader  = create_dataloader(df_c, FEATURE_COLS, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Created train_loader: {len(train_loader)} batches")
    print(f"Created val_loader:   {len(val_loader)} batches")
    print(f"Created test_loader:  {len(test_loader)} batches")

    # Model instantiation
    NUM_FEATURES  = len(FEATURE_COLS)
    D_MODEL       = 24
    NHEAD         = 3
    NUM_LAYERS    = 1
    DIM_FFORWARD  = 16
    DROPOUT_START       = 0.8
    DROPOUT_END         = 0.6

    model = SimpleTransformer(
        num_features    = NUM_FEATURES,
        d_model         = D_MODEL,
        nhead           = NHEAD,
        num_layers      = NUM_LAYERS,
        dim_feedforward = DIM_FFORWARD,
        dropout         = DROPOUT_START,
        swiglu_layer    = False,
        loss            = "BCEWithLogitsLoss",
        #gamma_plus      = 0.0,
        #gamma_minus     = 2.0,
        #m               = 0.02,
    ).to(device)

    # Selective weight decay: exclude biases and LayerNorm
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 1e-2},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )
    
    # Define dropout schedule: linear decay 
    dropout_sched = LinearDropoutScheduler(start_rate=DROPOUT_START, end_rate=DROPOUT_END, total_epochs=EPOCHS//2)

    # Train model and save checkpoints
    results = run_training_pipeline(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=base_dir_task_2_3a,
        epochs=EPOCHS,
        monitor="auprc",
        scheduler=scheduler,
        dropout_scheduler=dropout_sched,
        patience=10,
        data_format="dense"
    )

    # get directory
    current_save_dir_2_3a = results["save_dir"]
    current_run_name_2_3a = results["run_name"]

    # plot training and validation curves
    plot_training_results(results["history"], save_path=current_save_dir_2_3a / "training_curves.png")

    test_results = run_final_test(
    model=model,
    test_loader=test_loader,
    checkpoint_path=f"{current_save_dir_2_3a}/best_model.pt",
    device=device,
    data_format="dense"
)

    # Evaluate on test set and report results
    report_test_results(test_results, model_name="Simple Transformer (Task 2.3a)")

    # Clean checkpoint directory from all previous runs except the current one
    clean_checkpoint_dir(base_dir=base_dir_task_2_3a, keep_run_name=current_run_name_2_3a)

import torch
import numpy as np
from pathlib import Path
from chronos import ChronosBoltPipeline
from ML4HC_Project1.Scripts.Task_2_3a import load_and_preprocess_dataset, create_dataloader
from ML4HC_Project1.Scripts.Task_1 import (process_physionet_set, STATIC_VARS)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utilities.training_functions import set_seed, run_training_pipeline, run_final_test
from utilities.plotting import plot_training_results, report_test_results
from utilities.dir_manager import clean_checkpoint_dir



# ==========================================
# Classes and Definitions
# ==========================================

class SwiGLU(nn.Module):
    """
    SwiGLU Activation: (xW + b) * silu(xV + c)
    Standard implementation that chunks the input tensor into two halves.
    """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)

class MLPAggregator(nn.Module):
    def __init__(self, num_channels=37, embed_dim=768, n_hidden=1, hidden_dim=512, dropout=0.1):
        """
        A 'smarter' channel aggregator using multiple hidden layers and SwiGLU.
        
        Args:
            num_channels: 37 dynamic variables.
            embed_dim: 768 (for Chronos Bolt Small).
            n_hidden: Number of SwiGLU blocks.
            hidden_dim: The internal dimensionality.
            dropout: Dropout rate for regularization.
        """
        super(MLPAggregator, self).__init__()
        
        self.flatten = nn.Flatten()
        
        layers = []
        input_size = num_channels * embed_dim
        
        for i in range(n_hidden):
            # Project to 2x hidden_dim for SwiGLU 
            layers.append(nn.Linear(input_size, hidden_dim * 2))
            layers.append(SwiGLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            input_size = hidden_dim 
            
        self.hidden_layers = nn.Sequential(*layers)
        
        # Final projection to a single logit for binary classification
        self.output_head = nn.Linear(input_size, 1)
        
    def forward(self, x, mask=None):
        # x shape: (Batch, 37, 768)
        x = self.flatten(x) # (Batch, 37 * 768)
        x = self.hidden_layers(x) # (Batch, hidden_dim)
        return self.output_head(x) # (Batch, 1)

    def compute_loss(self, logits, targets):
        """
        Computes Binary Cross Entropy with Logits.
        """
        return F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1))

def extract_chronos_embeddings(dataloader, chronos_model, aggregator, device=torch.device("cpu")):
    """
    Extracts embeddings using Chronos and aggregates them using the provided aggregator.
    
    Args:
        dataloader: The PyTorch DataLoader.
        chronos_model: The loaded Chronos pipeline.
        aggregator: A function or nn.Module that takes a tensor of shape (B, C, D) 
                    and returns (B, D).
        device: The device to run inference on.
    """
    all_patient_embeddings = []
    all_labels = []
    
    encoder = chronos_model.model.to(device)
    encoder.eval()

    if isinstance(aggregator, nn.Module):
        aggregator.to(device)
        aggregator.eval()

    with torch.no_grad():
        for batch_data, _, labels in dataloader:
            # batch_data shape from dataloader: (Batch, Time, Channels)
            # We need (Batch, Channels, Time) so each channel is a univariate sequence.
            batch_data = batch_data.transpose(1, 2)
            B, C, T = batch_data.shape
            
            # Flatten for Chronos univariate processing: (B * C, T)
            flat_input = batch_data.reshape(B * C, T).to(device)
            
            # Extract Embeddings: May return a tuple/ModelOutput (hidden_states, ...)
            outputs = encoder.encode(flat_input)
            embeddings = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            # If the model returns the full sequence (B*C, S, D), pool to get a single vector per sequence.
            if embeddings.dim() == 3:
                embeddings = embeddings.mean(dim=1)
            
            # Unflatten back to Patient-Channel view: (B, C, D)
            unflattened_embeddings = embeddings.view(B, C, -1)
            
            # Apply the aggregator (BxCxD -> BxD)
            patient_level_rep = aggregator(unflattened_embeddings)
            
            all_patient_embeddings.append(patient_level_rep.cpu())
            all_labels.append(labels)

    return torch.cat(all_patient_embeddings, dim=0), torch.cat(all_labels, dim=0)

def mean_aggregator(x):
    """Mean Aggregator: x shape (B, C, D) -> returns (B, D)"""
    return x.mean(dim=1)

def identity_aggregator(x):
    """Identity Aggregator: returns x (B, C, D)"""
    return x 


# ==========================================
# Main Execution logic
# ==========================================

if __name__ == "__main__":

    # Constants
    PREPROCESSED_DIR = "output/preprocessed_imputed_datasets"
    CLIP = 5.0
    BATCH_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    rnd_generator = torch.Generator()
    EPOCHS = 5 

    set_seed(SEED)
    SAVE_DIR = "model_checkpoints/task_4_3"

    # Load the Chronos Bolt Small model
    print(f"Loading Chronos model on {DEVICE}...")
    chronos_model = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny",
        device_map=DEVICE,
        torch_dtype=torch.float32,
    )

    # Identify dynamic variables by removing STATIC_VARS
    df_a, ALL_COLS = load_and_preprocess_dataset(f"{PREPROCESSED_DIR}/preprocessed_set_a.parquet", "Set A", CLIP)
    df_b, _ = load_and_preprocess_dataset(f"{PREPROCESSED_DIR}/preprocessed_set_b.parquet", "Set B", CLIP)
    df_c, _ = load_and_preprocess_dataset(f"{PREPROCESSED_DIR}/preprocessed_set_c.parquet", "Set C", CLIP)

    DYNAMIC_FEATURES = [col for col in ALL_COLS if col not in STATIC_VARS]

    # Re-create dataloaders using only dynamic features for Chronos 
    train_loader = create_dataloader(df_a, DYNAMIC_FEATURES, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader = create_dataloader(df_b, DYNAMIC_FEATURES, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = create_dataloader(df_c, DYNAMIC_FEATURES, batch_size=BATCH_SIZE, shuffle=False)

    ##################
    print("Extracting training embeddings and aggregating with mean...")
    X_train_emb, y_train = extract_chronos_embeddings(train_loader, chronos_model, identity_aggregator)
    X_train_mean_emb = mean_aggregator(X_train_emb)
    
    print("Extracting test embeddings and aggregating with mean...")
    X_test_emb, y_test = extract_chronos_embeddings(test_loader, chronos_model, identity_aggregator)
    X_test_mean_emb = mean_aggregator(X_test_emb)

    print(f"Extraction complete. Feature matrix size: {X_train_emb.shape}")


    # Convert torch tensors to numpy arrays for Scikit-Learn
    X_train = X_train_mean_emb.numpy()
    y_train_np = y_train.numpy()
    X_test = X_test_mean_emb.numpy()
    y_test_np = y_test.numpy()

    # Logistic Regression model
    logreg = LogisticRegression(max_iter=1000, random_state=42)

    # Train on the Chronos embeddings
    print("Training Linear Probe on Chronos embeddings...")
    logreg.fit(X_train, y_train_np)

    # Evaluate on Test Set C
    y_probs = logreg.predict_proba(X_test)[:, 1]

    # Report metrics 
    auroc = roc_auc_score(y_test_np, y_probs) 
    auprc = average_precision_score(y_test_np, y_probs) 

    print("-" * 35)
    print(f"Chronos Linear Probe (Mean Aggregation):")
    print(f"Test AuROC: {auroc:.4f}")
    print(f"Test AuPRC: {auprc:.4f}")
    print("-" * 35)


    # get model embedding dimension
    num_extracted_channels = X_train_emb.shape[1]
    embed_dim = X_train_emb.shape[2]

    # Smart Aggregator (MLP)
    smart_aggregator = MLPAggregator(
        num_channels=num_extracted_channels, 
        embed_dim=embed_dim, 
        n_hidden=5,
        dropout=0.4, 
        hidden_dim=512
    ).to(DEVICE)

    print("\nExtracting Validation Embeddings for MLP Aggregator/Identity...")
    X_val_raw, y_val = extract_chronos_embeddings(valid_loader, chronos_model, identity_aggregator)

    # Create DataLoaders for the extracted features with dummy masks for compatibility
    # The training pipeline expects (data, mask, labels) during iteration.
    train_masks = torch.ones(X_train_emb.shape[0], 1)
    train_ds = TensorDataset(X_train_emb, train_masks, y_train.float())
    fast_train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=rnd_generator)

    val_masks = torch.ones(X_val_raw.shape[0], 1)
    eval_ds = TensorDataset(X_val_raw, val_masks, y_val.float())
    fast_eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, generator=rnd_generator)

    test_masks = torch.ones(X_test_emb.shape[0], 1)
    test_ds = TensorDataset(X_test_emb, test_masks, y_test.float())
    fast_test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, generator=rnd_generator)

    
    optimizer = torch.optim.Adam(smart_aggregator.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Initialize Scheduler: ReduceLROnPlateau
    # # Reduce the learning rate if validation loss doesn't improve for patience+1 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=2
    )

    print("\nStarting end-to-end training of MLP Aggregator head with ReduceLROnPlateau...")
    train_results = run_training_pipeline(
        model=smart_aggregator,
        optimizer=optimizer,
        train_loader=fast_train_loader,
        val_loader=fast_eval_loader,
        device=torch.device(DEVICE),
        scheduler=scheduler,  
        save_dir=SAVE_DIR,
        epochs=EPOCHS,
        data_format="dense"
    )

    # Plot results
    plot_training_results(train_results["history"], save_path=Path(train_results["save_dir"]) / "training_curves.png")

    # Final test evaluation
    checkpoint_path = Path(train_results["save_dir"]) / "best_model.pt"
    test_results = run_final_test(
        model=smart_aggregator,
        test_loader=fast_test_loader,
        checkpoint_path=str(checkpoint_path),
        device=torch.device(DEVICE),
        data_format="dense"
    )

    # Report results using formatted utility
    report_test_results(test_results, model_name="Chronos-MLP-Aggregator")
    
    # Cleanup directory preserving only the best run
    clean_checkpoint_dir(SAVE_DIR, keep_run_name=train_results["run_name"])
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, average_precision_score

from utilities.dir_manager import setup_run_directory, save_checkpoint

def set_seed(seed: int) -> None:
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _train_triplet_epoch(
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

def _train_dense_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
) -> Dict[str, float]:
    """Trains the model for one epoch on dense dataset."""
    model.train()
    train_loss = 0
    all_targets = []
    all_preds = []

    for x, mask, labels in dataloader:
        x, mask, labels = x.to(device), mask.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(x, mask)
        loss = model.compute_loss(logits.view(-1), labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        all_targets.append(labels.detach().cpu())
        all_preds.append(torch.sigmoid(logits).detach().cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    return {
        "loss": train_loss / len(dataloader),
        "auroc": roc_auc_score(all_targets, all_preds),
        "auprc": average_precision_score(all_targets, all_preds)
    }

def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    data_format: str = "dense"
) -> Dict[str, float]:
    """Wrapper that routes to the specific training function based on data_format."""
    if data_format == "triplet":
        return _train_triplet_epoch(model, dataloader, optimizer, device)
    elif data_format == "dense":
        return _train_dense_epoch(model, dataloader, optimizer, device)
    else:
        raise ValueError(f"Unknown data_format: {data_format}")

@torch.no_grad()
def _evaluate_triplet(
    model: nn.Module,
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

@torch.no_grad()
def _evaluate_dense(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    eval_loss = 0.0
    all_targets = []
    all_preds = []

    for x, mask, labels in dataloader:
        x, mask, labels = x.to(device), mask.to(device), labels.to(device)

        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(x, mask)
            loss = model.compute_loss(logits.view(-1), labels)

        eval_loss += loss.item()
        all_targets.append(labels.detach().cpu())
        all_preds.append(torch.sigmoid(logits).detach().cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    return {
        "loss": eval_loss / len(dataloader),
        "auroc": roc_auc_score(all_targets, all_preds),
        "auprc": average_precision_score(all_targets, all_preds)
    }

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    data_format: str = "dense"
) -> Dict[str, float]:
    """Wrapper that routes to the specific evaluation function based on data_format."""
    if data_format == "triplet":
        return _evaluate_triplet(model, dataloader, device)
    elif data_format == "dense":
        return _evaluate_dense(model, dataloader, device)
    else:
        raise ValueError(f"Unknown data_format: {data_format}")

def run_training_pipeline(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: str = "model_checkpoints/task_2_3b",
    epochs: int = 20,
    data_format: str = "dense"
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
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, data_format=data_format)
        val_metrics = evaluate(model, val_loader, device, data_format=data_format)
        
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

def run_final_test(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_path: str,
    device: torch.device,
    data_format: str = "dense"
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
    test_results = evaluate(model, test_loader, device, data_format=data_format)
    
    # Format and display the results
    print("\n" + "="*30)
    print("FINAL TEST RESULTS (SET C)")
    print("="*30)
    print(f"AuROC: {test_results['auroc']:.4f}")
    print(f"AuPRC: {test_results['auprc']:.4f}")
    print(f"Loss:  {test_results['loss']:.4f}")
    print("="*30)
    
    return test_results
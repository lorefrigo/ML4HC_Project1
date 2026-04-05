import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, average_precision_score

from utilities.dir_manager import setup_run_directory, save_checkpoint

from typing import Dict, Any, Optional, Protocol, cast

class DropoutScheduler(Protocol):
    def __call__(self, epoch: int) -> float:
        ...

class ConstantDropoutScheduler:
    def __init__(self, drop_rate: float):
        self.drop_rate = drop_rate
    def __call__(self, epoch: int) -> float:
        return self.drop_rate

class LinearDropoutScheduler:
    def __init__(self, start_rate: float, end_rate: float, total_epochs: int):
        self.start_rate = start_rate
        self.end_rate = end_rate
        self.total_epochs = total_epochs
        
    def __call__(self, epoch: int) -> float:
        if epoch >= self.total_epochs:
            return self.end_rate
        pct = epoch / self.total_epochs
        return self.start_rate + pct * (self.end_rate - self.start_rate)

class StepDropoutScheduler:
    def __init__(self, start_rate: float, end_rate: float, total_epochs: int, num_steps: int):
        self.start_rate = start_rate
        self.end_rate = end_rate
        self.total_epochs = total_epochs
        self.num_steps = num_steps
        self.step_size = max(1, total_epochs // num_steps)
        
    def __call__(self, epoch: int) -> float:
        step = epoch // self.step_size
        if step >= self.num_steps:
            return self.end_rate
        
        if self.num_steps > 1:
            delta = (self.end_rate - self.start_rate) / (self.num_steps - 1)
            return self.start_rate + step * delta
        return self.end_rate

def set_dropout(model: nn.Module, drop_rate: float) -> None:
    """
    Recursively sets the dropout rate for all nn.Dropout and 
    nn.MultiheadAttention modules in the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            setattr(module, 'p', drop_rate)
        elif isinstance(module, nn.MultiheadAttention):
            setattr(module, 'dropout', drop_rate)

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
    device = torch.device(device)
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
    device = torch.device(device)
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
    scheduler: Any = None,
    dropout_scheduler: Optional[DropoutScheduler] = None,
    save_dir: str = "model_checkpoints/task_2_3b",
    epochs: int = 20,
    data_format: str = "dense",
    patience: Optional[int] = None,
    monitor: str = "loss"
) -> dict[str, Any]:
    """
    Core training loop. Orchestrates training, evaluation, and checkpointing.
    Supports early stopping if patience is provided.
    """
    device = torch.device(device)
    # Initialize the run directory
    save_dir, run_name = setup_run_directory(save_dir)
    
    # Initialize trackers
    is_lower_better = monitor == "loss"  
    best_val_monitor = float("inf") if is_lower_better else -float("inf")
    best_val_auprc = -float("inf")
    early_stop_counter = 0
    
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        # Update dropout if a scheduler is provided
        if dropout_scheduler is not None:
            # We know it's not None here, cast for the type checker
            sched = cast(DropoutScheduler, dropout_scheduler)
            current_drop_rate = sched(epoch - 1)
            set_dropout(model, current_drop_rate)
            if epoch % 5 == 0 or epoch == 1:
                print(f"--- Epoch {epoch}: Setting dropout rate to {current_drop_rate:.4f} ---")

        # Train and Validate
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, data_format=data_format)
        val_metrics = evaluate(model, val_loader, device, data_format=data_format)
        
        # Update History
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"Epoch {epoch:02d} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"Val AuROC: {val_metrics['auroc']:.4f} | Val AuPRC: {val_metrics['auprc']:.4f}")

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau typically monitors validation loss, but is now configured for monitor metric
                scheduler.step(val_metrics[monitor])
            else:
                scheduler.step()

        # Checkpointing (based on AuPRC)
        current_val_auprc = val_metrics["auprc"]
        if current_val_auprc > best_val_auprc:
            best_val_auprc = current_val_auprc
            # save best model and history
            save_checkpoint(
                state_dict=model.state_dict(),
                history=history,
                save_dir=save_dir
            )
            print(f"  * Best model updated (AuPRC: {best_val_auprc:.4f})")

        # Early Stopping & Scheduler Monitor (based on 'monitor')
        current_val_monitor = val_metrics[monitor]
        has_improved = (current_val_monitor < best_val_monitor) if is_lower_better else (current_val_monitor > best_val_monitor)

        if has_improved:
            best_val_monitor = current_val_monitor
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if isinstance(patience, int) and early_stop_counter >= patience:
                print(f"\nEarly stopping triggered: No improvement in {monitor} for {patience} epochs.")
                break

    return {
        f"best_val_{monitor}": best_val_monitor,
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
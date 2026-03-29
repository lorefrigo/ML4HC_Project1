import json
import shutil
import torch
from datetime import datetime
from pathlib import Path


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
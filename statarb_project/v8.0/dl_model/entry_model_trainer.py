import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

# Allow running as a standalone script
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

sys.path.append('/storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0')
from dataset import StatArbDataset, StatArbSubset
from entry_quality_preprocessor import load_remove_pairs
from lightweight_entry_model import LightweightEntryQualityModel


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to: {seed}")


def create_train_val_split(dataset: StatArbDataset, val_ratio: float = 0.2, random_seed: Optional[int] = None):
    """Split dataset into train and validation subsets."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - val_ratio))
    train_indices = indices[:split_idx].tolist()
    val_indices = indices[split_idx:].tolist()
    
    train_dataset = StatArbSubset(dataset, train_indices)
    val_dataset = StatArbSubset(dataset, val_indices)
    
    return train_dataset, val_dataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    data_dir: str,
    remove_pairs_file: Optional[str],
    lookback: int = 64,
    batch_size: int = 256,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 5,
    max_files: Optional[int] = None,
    max_samples_per_file: Optional[int] = None,
    model_out: str = "entry_quality_gru.pt",
    val_ratio: float = 0.2,
    random_seed: Optional[int] = 42,
) -> None:
    # Set global random seed if provided
    if random_seed is not None:
        set_global_seed(random_seed)
    
    remove_pairs = load_remove_pairs(remove_pairs_file)

    # Load full dataset
    print("Loading full dataset...")
    full_dataset = StatArbDataset(
        data_dir=data_dir,
        remove_pairs=remove_pairs,
        lookback=lookback,
        per_sample_norm=False,
        max_files=max_files,
        max_samples_per_file=max_samples_per_file,
    )
    
    # Print dataset statistics
    full_dataset.print_statistics()
    
    if len(full_dataset) == 0:
        raise RuntimeError("No training samples found. Check data directory and remove-pairs filter.")
    
    # Get feature dimension
    C = full_dataset[0][0].shape[-1]
    
    # Split into train/val
    print(f"\nSplitting dataset: {1-val_ratio:.1%} train, {val_ratio:.1%} validation")
    train_dataset, val_dataset = create_train_val_split(full_dataset, val_ratio=val_ratio, random_seed=random_seed)
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    def collate(batch):
        xs, ys = zip(*batch)
        X = torch.stack(xs, dim=0)
        Y = torch.tensor(ys, dtype=torch.float32)  # Binary labels as float for BCE
        return X, Y

    # Create data loaders
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # Model
    device = get_device()
    model = LightweightEntryQualityModel(input_dim=C, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    model.to(device)

    # Optim and loss (BCE for binary classification)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    # Training loop with validation
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_n = 0
        
        for X, Y in train_dl:
            X = X.to(device)
            Y = Y.to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(X).squeeze(-1)  # [batch_size, 1] -> [batch_size]
            loss = loss_fn(pred, Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            bs = X.size(0)
            train_loss += loss.item() * bs
            train_n += bs
            global_step += 1

        train_loss = train_loss / max(train_n, 1)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_n = 0
        
        with torch.no_grad():
            for X, Y in val_dl:
                X = X.to(device)
                Y = Y.to(device)
                pred = model(X).squeeze(-1)  # [batch_size, 1] -> [batch_size]
                loss = loss_fn(pred, Y)
                
                bs = X.size(0)
                val_loss += loss.item() * bs
                val_n += bs
        
        val_loss = val_loss / max(val_n, 1)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Calculate validation metrics: precision, recall, PR-AUC
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X, Y in val_dl:
                X = X.to(device)
                Y = Y.to(device)
                pred = model(X).squeeze(-1)
                
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(Y.cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # Calculate binary predictions and metrics
        val_pred_binary = (val_preds > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = (val_pred_binary == val_targets).mean()
        
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_targets, val_pred_binary, average='binary', zero_division=0
        )
        
        # Calculate PR-AUC
        pr_auc = average_precision_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.0
        
        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, acc={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, pr_auc={pr_auc:.4f} (train_n={train_n:,}, val_n={val_n:,})")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation loss: {best_val_loss:.6f}")

    # Save best model
    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": C,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lookback": lookback,
        },
        "best_val_loss": best_val_loss,
        "training_info": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "random_seed": random_seed,
        }
    }, out_path)
    print(f"Saved best model to {out_path} (val_loss={best_val_loss:.6f})")


def main():
    ap = argparse.ArgumentParser(description="Train GRU binary classification model for stat-arb entry quality")
    ap.add_argument("--data-dir", type=str, default='/storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0/data', help="Directory with ml_data_*.json files")
    ap.add_argument("--remove-pairs-file", type=str, default="/storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0/remove_pair_list.txt")
    ap.add_argument("--lookback", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max-files", type=int, default=20)
    ap.add_argument("--max-samples-per-file", type=int, default=None)
    ap.add_argument("--model-out", type=str, default="/storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0/dl_model/output/entry_quality_gru.pt")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio")
    ap.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    train(
        data_dir=args.data_dir,
        remove_pairs_file=args.remove_pairs_file,
        lookback=args.lookback,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        max_files=args.max_files,
        max_samples_per_file=args.max_samples_per_file,
        model_out=args.model_out,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()

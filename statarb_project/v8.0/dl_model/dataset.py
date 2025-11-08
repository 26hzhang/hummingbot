from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append('/storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0')
from entry_quality_preprocessor import (
    extract_features_from_sample,
    iter_json_samples,
    list_data_files,
)

class StatArbDataset(Dataset):
    """
    Dataset that loads all samples into memory at initialization.
    Provides dataset statistics.
    """

    def __init__(
        self,
        data_dir: str,
        remove_pairs: Optional[set] = None,
        lookback: int = 64,
        per_sample_norm: bool = True,
        max_files: Optional[int] = None,
        max_samples_per_file: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.remove_pairs = remove_pairs or set()
        self.lookback = lookback
        self.per_sample_norm = per_sample_norm
        self.max_files = max_files
        self.max_samples_per_file = max_samples_per_file
        
        # Load all data at initialization
        self.samples = []
        self.targets = []
        self.pairs = []

        self._load_all_data()
        self._compute_statistics()

    def _load_all_data(self) -> None:
        """Load all data into memory at initialization."""
        files: List[Path] = list_data_files(self.data_dir)
        if self.max_files is not None:
            files = files[: self.max_files]
            
        print(f"Loading data from {len(files)} files...")
        
        for fp in files:
            count = 0
            for sample in iter_json_samples(fp):
                out = extract_features_from_sample(
                    sample,
                    lookback=self.lookback,
                    per_sample_norm=self.per_sample_norm,
                )
                if out is None:
                    continue
                x, y, pair = out
                if pair in self.remove_pairs:
                    continue
                    
                self.samples.append(torch.from_numpy(x))
                self.targets.append(y)
                self.pairs.append(pair)
                count += 1
                
                if self.max_samples_per_file is not None and count >= self.max_samples_per_file:
                    break

        print(f"Loaded {len(self.samples)} valid samples")
    
    def _compute_statistics(self) -> None:
        """Compute dataset statistics."""
        if not self.targets:
            self.stats = {}
            return
            
        targets_array = np.array(self.targets)
        unique_pairs = list(set(self.pairs))
        
        self.stats = {
            'total_samples': len(self.targets),
            'unique_pairs': len(unique_pairs),
            'pair_list': sorted(unique_pairs),
            'target_stats': {
                'mean': float(np.mean(targets_array)),
                'std': float(np.std(targets_array)),
                'min': float(np.min(targets_array)),
                'max': float(np.max(targets_array)),
            },
            'pair_counts': {pair: self.pairs.count(pair) for pair in unique_pairs},
            'feature_shape': self.samples[0].shape if self.samples else None,
        }
    
    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        return self.stats
    
    def print_statistics(self) -> None:
        """Print formatted dataset statistics."""
        stats = self.stats
        print("\n=== Dataset Statistics ===")
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Unique pairs: {stats['unique_pairs']}")
        print(f"Feature shape: {stats['feature_shape']}")
        
        target_stats = stats['target_stats']
        print(f"\nTarget (profit_percentage) statistics:")
        print(f"  Mean: {target_stats['mean']:.4f}")
        print(f"  Std:  {target_stats['std']:.4f}")
        print(f"  Min:  {target_stats['min']:.4f}")
        print(f"  Max:  {target_stats['max']:.4f}")
        print("="*30)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], torch.tensor([self.targets[idx]], dtype=torch.float32)


class StatArbSubset(Dataset):
    """Lightweight subset wrapper for train/val split."""
    
    def __init__(self, dataset: StatArbDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]
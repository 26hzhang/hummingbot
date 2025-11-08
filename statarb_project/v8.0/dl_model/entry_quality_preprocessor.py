import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _to_np(arr: List[float]) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (on the left) or truncate 1D array to target_len."""
    L = seq.shape[0]
    if L == target_len:
        return seq
    if L > target_len:
        return seq[-target_len:]
    # pad on the left with the first value
    pad_val = seq[0] if L > 0 else 0.0
    pad = np.full((target_len - L,), pad_val, dtype=seq.dtype)
    return np.concatenate([pad, seq], axis=0)


def _stack_and_align(features: List[np.ndarray], target_len: int) -> np.ndarray:
    """Ensure each 1D feature is length target_len, then stack to [target_len, C]."""
    processed = [_pad_or_truncate(f.astype(np.float32), target_len) for f in features]
    x = np.stack(processed, axis=1)  # [T, C]
    return x


def _per_sample_standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Channel-wise standardization per sample over time (mean/std across T)."""
    # x: [T, C]
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def load_remove_pairs(path: Optional[str]) -> set:
    pairs = set()
    if not path:
        return pairs
    p = Path(path)
    if not p.exists():
        return pairs
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or '-' not in line:
            continue
        pairs.add(line)
    return pairs


def extract_features_from_sample(
    sample: Dict,
    lookback: int = 64,
    per_sample_norm: bool = True,
    clamp_target: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[np.ndarray, int, str]]:
    """
    Extract time series features from a trading sample for binary classification.

    Args:
        sample: JSON sample containing historical data and target variables
        lookback: Fixed sequence length for features
        per_sample_norm: Whether to normalize features per sample
        clamp_target: Not used for binary classification, kept for compatibility

    Returns:
        Tuple of (features[T,C], binary_target, trading_pair) or None if invalid
        - binary_target: 0 if profit < 0.1%, 1 if profit > 1%
        
    Feature channels (C=8):
        0: Asset1 close price log returns
        1: Asset2 close price log returns  
        2: Price ratio (asset2/asset1) log returns
        3: Z-score history (raw values)
        4: Beta coefficient history (raw values)
        5: Alpha coefficient history (raw values)
        6: Asset1 volume log returns
        7: Asset2 volume log returns
    """
    
    # 1. Extract and validate target variable
    target_vars = sample.get("target_variables", {})
    profit_percentage = target_vars.get("profit_percentage", None)
    if profit_percentage is None or (isinstance(profit_percentage, float) and 
                                   (math.isnan(profit_percentage) or math.isinf(profit_percentage))):
        return None
    
    # Convert to binary classification: 0 if profit < 0.1%, 1 if profit > 1%
    # Debug: Print profit distribution
    if profit_percentage < 0.1:
        binary_target = 0
    elif profit_percentage >= 0.1:
        binary_target = 1
    else:
        return None  # Skip samples in between thresholds (0.1% <= profit <= 1.0%)

    # 2. Extract historical time series data
    asset1_close_history = _to_np(sample.get("asset1_close_history", []))
    asset2_close_history = _to_np(sample.get("asset2_close_history", []))
    asset1_volume_history = _to_np(sample.get("asset1_volume_history", []))
    asset2_volume_history = _to_np(sample.get("asset2_volume_history", []))
    zscore_history = _to_np(sample.get("zscore_history", []))
    beta_history = _to_np(sample.get("beta_history", []))
    alpha_history = _to_np(sample.get("alpha_history", []))

    # 3. Validate minimum data length
    min_required_length = 2  # Need at least 2 points to compute returns
    all_series = [asset1_close_history, asset2_close_history, 
                  asset1_volume_history, asset2_volume_history, zscore_history, beta_history, alpha_history]
    
    if min(len(series) for series in all_series) < min_required_length:
        return None

    # 4. Define log returns calculation function
    def compute_log_returns(price_series: np.ndarray) -> np.ndarray:
        """Compute log returns: log(price_t / price_{t-1})"""
        if len(price_series) < 2:
            return np.zeros((1,), dtype=np.float32)
        price_ratios = price_series[1:] / np.clip(price_series[:-1], 1e-12, None)
        return _safe_log(price_ratios)

    # 5. Compute feature time series
    # Price-based features
    asset1_price_returns = compute_log_returns(asset1_close_history)
    asset2_price_returns = compute_log_returns(asset2_close_history)
    
    # Price ratio log returns
    price_ratio = asset2_close_history / np.clip(asset1_close_history, 1e-12, None)
    price_ratio_returns = compute_log_returns(price_ratio)
    
    # Volume-based features 
    asset1_volume_returns = compute_log_returns(asset1_volume_history)
    asset2_volume_returns = compute_log_returns(asset2_volume_history)

    # 6. Align all feature sequences to common length
    feature_sequences = [
        asset1_price_returns,      # Channel 0
        asset2_price_returns,      # Channel 1
        price_ratio_returns,       # Channel 2
        zscore_history,            # Channel 3 (raw)
        beta_history,              # Channel 4 (raw)
        alpha_history,             # Channel 5 (raw)
        asset1_volume_returns,     # Channel 6
        asset2_volume_returns,     # Channel 7
    ]
    
    sequence_lengths = [len(seq) for seq in feature_sequences]
    min_common_length = min(sequence_lengths) if all(sequence_lengths) else 0
    
    if min_common_length == 0:
        return None
        
    # Truncate all sequences to common length (keep most recent)
    aligned_sequences = [seq[-min_common_length:] for seq in feature_sequences]

    # 7. Stack and pad/truncate to fixed lookback length
    feature_matrix = _stack_and_align(aligned_sequences, lookback)  # Shape: [T, C]
    
    # 8. Apply per-sample normalization if requested
    if per_sample_norm:
        feature_matrix = _per_sample_standardize(feature_matrix)

    # 9. Extract trading pair information
    trading_pair = sample.get("trade_info", {}).get("pair", "UNKNOWN-UNKNOWN")

    # 10. Return binary classification result
    return feature_matrix.astype(np.float32), binary_target, trading_pair


def iter_json_samples(json_path: Path) -> Iterable[Dict]:
    """Iterate JSON array samples from a file path. Note: loads entire file."""
    data = json.loads(json_path.read_text())
    for item in data:
        yield item


def list_data_files(data_dir: str) -> List[Path]:
    p = Path(data_dir)
    files = sorted(p.glob("ml_data_*.json"))
    return files


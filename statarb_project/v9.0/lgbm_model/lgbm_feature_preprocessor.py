"""
V9.0 LightGBM Feature Preprocessor
基于v8.0的entry_quality_preprocessor.py扩展，针对LightGBM优化特征工程
"""
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_np(arr: List[float]) -> np.ndarray:
    """Convert list to numpy array with float32 dtype"""
    return np.asarray(arr, dtype=np.float32)


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Safe logarithm with epsilon clipping"""
    return np.log(np.clip(x, eps, None))


def _compute_ohlcv_indicators(open_prices: np.ndarray, high_prices: np.ndarray, 
                             low_prices: np.ndarray, close_prices: np.ndarray, 
                             volumes: np.ndarray) -> Dict[str, float]:
    """
    基于完整OHLCV数据计算技术指标特征
    
    Args:
        open_prices: 开盘价历史
        high_prices: 最高价历史  
        low_prices: 最低价历史
        close_prices: 收盘价历史
        volumes: 成交量历史
        
    Returns:
        技术指标特征字典
    """
    if len(close_prices) < 5:
        return {}
    
    features = {}
    
    # 基础价格统计特征
    features['close_mean'] = float(np.mean(close_prices))
    features['close_std'] = float(np.std(close_prices))
    features['close_latest'] = float(close_prices[-1])
    features['close_min'] = float(np.min(close_prices))
    features['close_max'] = float(np.max(close_prices))
    features['close_range'] = features['close_max'] - features['close_min']
    
    # 价格变异系数（CV）
    features['close_cv'] = features['close_std'] / features['close_mean'] if features['close_mean'] > 0 else 0
    
    # OHLC相关特征
    if len(open_prices) == len(close_prices):
        # 日内波动幅度
        daily_ranges = (high_prices - low_prices) / close_prices
        features['avg_daily_range'] = float(np.mean(daily_ranges))
        features['daily_range_std'] = float(np.std(daily_ranges))
        
        # 收盘vs开盘比例
        oc_ratios = close_prices / np.clip(open_prices, 1e-12, None)
        features['avg_oc_ratio'] = float(np.mean(oc_ratios))
        features['oc_ratio_std'] = float(np.std(oc_ratios))
        
        # 上影线和下影线
        upper_shadows = (high_prices - np.maximum(open_prices, close_prices)) / close_prices
        lower_shadows = (np.minimum(open_prices, close_prices) - low_prices) / close_prices
        features['avg_upper_shadow'] = float(np.mean(upper_shadows))
        features['avg_lower_shadow'] = float(np.mean(lower_shadows))
    
    # 收益率特征
    if len(close_prices) >= 2:
        returns = np.diff(close_prices) / close_prices[:-1]
        returns = returns[~np.isnan(returns)]  # 移除NaN
        if len(returns) > 0:
            features['returns_mean'] = float(np.mean(returns))
            features['returns_std'] = float(np.std(returns))
            features['returns_skew'] = float(pd.Series(returns).skew()) if len(returns) > 2 else 0
            features['returns_kurt'] = float(pd.Series(returns).kurtosis()) if len(returns) > 2 else 0
            
            # 正负收益率比例
            positive_returns = returns[returns > 0]
            features['positive_return_ratio'] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    
    # 移动平均特征
    if len(close_prices) >= 5:
        ma5 = np.mean(close_prices[-5:])
        ma10 = np.mean(close_prices[-min(10, len(close_prices)):])
        ma20 = np.mean(close_prices[-min(20, len(close_prices)):])
        
        features['ma5'] = float(ma5)
        features['ma10'] = float(ma10) 
        features['ma20'] = float(ma20)
        features['ma5_ratio'] = features['close_latest'] / ma5 if ma5 > 0 else 1
        features['ma10_ratio'] = features['close_latest'] / ma10 if ma10 > 0 else 1
        features['ma20_ratio'] = features['close_latest'] / ma20 if ma20 > 0 else 1
        
        # MA趋势
        if len(close_prices) >= 10:
            ma5_prev = np.mean(close_prices[-10:-5])
            ma10_prev = np.mean(close_prices[-min(15, len(close_prices)):-5])
            features['ma5_trend'] = (ma5 - ma5_prev) / ma5_prev if ma5_prev > 0 else 0
            features['ma10_trend'] = (ma10 - ma10_prev) / ma10_prev if ma10_prev > 0 else 0
    
    # 成交量特征
    if len(volumes) > 0:
        features['volume_mean'] = float(np.mean(volumes))
        features['volume_std'] = float(np.std(volumes))
        features['volume_latest'] = float(volumes[-1])
        features['volume_ratio'] = features['volume_latest'] / features['volume_mean'] if features['volume_mean'] > 0 else 1
        features['volume_cv'] = features['volume_std'] / features['volume_mean'] if features['volume_mean'] > 0 else 0
        
        # 成交量趋势
        if len(volumes) >= 10:
            recent_volume = np.mean(volumes[-5:])
            prev_volume = np.mean(volumes[-10:-5]) 
            features['volume_trend'] = (recent_volume - prev_volume) / prev_volume if prev_volume > 0 else 0
    
    return features


def extract_lgbm_features_from_sample(
    sample: Dict,
    lookback: int = 64,
    use_technical_indicators: bool = True,
    clamp_target: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[np.ndarray, float, str]]:
    """
    从交易样本中提取LightGBM特征（标量特征向量）
    
    Args:
        sample: JSON样本，包含历史数据和目标变量
        lookback: 历史数据回望期数（用于计算统计特征）
        use_technical_indicators: 是否使用技术指标特征
        clamp_target: 目标值裁剪范围
        
    Returns:
        Tuple of (features_vector, target_value, trading_pair) or None if invalid
        - features_vector: 1D特征向量，适合LightGBM训练
        - target_value: 连续目标值（profit_percentage）
        
    特征维度说明（约80-100维）：
        - 基础统计特征：双资产的价格/成交量统计（均值、标准差、最新值等）
        - 价格比值特征：价格比序列的统计特征
        - Kalman参数特征：Z-score、Beta、Alpha的统计特征
        - 技术指标特征：移动平均、波动率、偏度、峰度等
        - 交叉特征：不同资产间的相关性等
    """
    
    # 1. 提取和验证目标变量
    target_vars = sample.get("target_variables", {})
    profit_percentage = target_vars.get("profit_percentage", None)
    if profit_percentage is None or (isinstance(profit_percentage, float) and 
                                   (math.isnan(profit_percentage) or math.isinf(profit_percentage))):
        return None
    
    # 目标值裁剪（增强稳健性）
    if clamp_target is not None:
        profit_percentage = np.clip(profit_percentage, clamp_target[0], clamp_target[1])
    
    # 2. 提取历史时序数据（完整OHLCV数据）
    asset1_open_history = _to_np(sample.get("asset1_open_history", []))
    asset1_high_history = _to_np(sample.get("asset1_high_history", []))
    asset1_low_history = _to_np(sample.get("asset1_low_history", []))
    asset1_close_history = _to_np(sample.get("asset1_close_history", []))
    asset1_volume_history = _to_np(sample.get("asset1_volume_history", []))
    
    asset2_open_history = _to_np(sample.get("asset2_open_history", []))
    asset2_high_history = _to_np(sample.get("asset2_high_history", []))
    asset2_low_history = _to_np(sample.get("asset2_low_history", []))
    asset2_close_history = _to_np(sample.get("asset2_close_history", []))
    asset2_volume_history = _to_np(sample.get("asset2_volume_history", []))
    
    zscore_history = _to_np(sample.get("zscore_history", []))
    beta_history = _to_np(sample.get("beta_history", []))
    alpha_history = _to_np(sample.get("alpha_history", []))
    
    # 3. 验证数据长度
    min_required_length = 5  # LightGBM需要足够数据计算统计特征
    all_series = [asset1_close_history, asset2_close_history, 
                  asset1_volume_history, asset2_volume_history, 
                  zscore_history, beta_history, alpha_history]
    
    if min(len(series) for series in all_series if len(series) > 0) < min_required_length:
        return None
    
    # 4. 初始化特征字典
    features = {}
    
    # 5. Asset1 OHLCV技术指标特征
    if len(asset1_close_history) >= min_required_length:
        asset1_tech = _compute_ohlcv_indicators(
            asset1_open_history, asset1_high_history, 
            asset1_low_history, asset1_close_history, asset1_volume_history
        )
        for key, val in asset1_tech.items():
            features[f'asset1_{key}'] = val
    
    # 6. Asset2 OHLCV技术指标特征
    if len(asset2_close_history) >= min_required_length:
        asset2_tech = _compute_ohlcv_indicators(
            asset2_open_history, asset2_high_history,
            asset2_low_history, asset2_close_history, asset2_volume_history
        )
        for key, val in asset2_tech.items():
            features[f'asset2_{key}'] = val
    
    # 7. 价格比值特征
    if len(asset1_close_history) > 0 and len(asset2_close_history) > 0:
        min_len = min(len(asset1_close_history), len(asset2_close_history))
        asset1_aligned = asset1_close_history[-min_len:]
        asset2_aligned = asset2_close_history[-min_len:]
        
        # 计算价格比值序列
        price_ratio = asset2_aligned / np.clip(asset1_aligned, 1e-12, None)
        
        # 价格比值的基础统计特征
        features['price_ratio_mean'] = float(np.mean(price_ratio))
        features['price_ratio_std'] = float(np.std(price_ratio))
        features['price_ratio_latest'] = float(price_ratio[-1])
        features['price_ratio_cv'] = features['price_ratio_std'] / features['price_ratio_mean'] if features['price_ratio_mean'] > 0 else 0
        
        # 价格比值收益率
        if len(price_ratio) >= 2:
            ratio_returns = np.diff(price_ratio) / price_ratio[:-1]
            ratio_returns = ratio_returns[~np.isnan(ratio_returns)]
            if len(ratio_returns) > 0:
                features['price_ratio_returns_mean'] = float(np.mean(ratio_returns))
                features['price_ratio_returns_std'] = float(np.std(ratio_returns))
                if len(ratio_returns) > 2:
                    features['price_ratio_returns_skew'] = float(pd.Series(ratio_returns).skew())
        
        # 价格比值移动平均
        if len(price_ratio) >= 5:
            ratio_ma5 = np.mean(price_ratio[-5:])
            features['price_ratio_ma5'] = float(ratio_ma5)
            features['price_ratio_ma5_ratio'] = features['price_ratio_latest'] / ratio_ma5 if ratio_ma5 > 0 else 1
    
    # 8. Kalman参数统计特征
    for param_name, param_history in [('zscore', zscore_history), 
                                      ('beta', beta_history), 
                                      ('alpha', alpha_history)]:
        if len(param_history) >= min_required_length:
            features[f'{param_name}_mean'] = float(np.mean(param_history))
            features[f'{param_name}_std'] = float(np.std(param_history))
            features[f'{param_name}_latest'] = float(param_history[-1])
            features[f'{param_name}_min'] = float(np.min(param_history))
            features[f'{param_name}_max'] = float(np.max(param_history))
            features[f'{param_name}_range'] = features[f'{param_name}_max'] - features[f'{param_name}_min']
            
            # 最近趋势
            if len(param_history) >= 10:
                recent_trend = np.polyfit(range(10), param_history[-10:], 1)[0]
                features[f'{param_name}_trend'] = float(recent_trend)
    
    # 9. 交叉特征
    if len(asset1_close_history) >= 5 and len(asset2_close_history) >= 5:
        min_len = min(len(asset1_close_history), len(asset2_close_history))
        
        # 价格相关性
        try:
            corr_matrix = np.corrcoef(asset1_close_history[-min_len:], asset2_close_history[-min_len:])
            features['price_correlation'] = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0
        except:
            features['price_correlation'] = 0
        
        # 成交量相关性
        if len(asset1_volume_history) >= min_len and len(asset2_volume_history) >= min_len:
            try:
                vol_corr_matrix = np.corrcoef(asset1_volume_history[-min_len:], asset2_volume_history[-min_len:])
                features['volume_correlation'] = float(vol_corr_matrix[0, 1]) if not np.isnan(vol_corr_matrix[0, 1]) else 0
            except:
                features['volume_correlation'] = 0
        else:
            features['volume_correlation'] = 0
            
        # 价格收益率相关性
        if len(asset1_close_history) >= 6 and len(asset2_close_history) >= 6:
            returns1 = np.diff(asset1_close_history[-min_len:]) / asset1_close_history[-min_len:-1]
            returns2 = np.diff(asset2_close_history[-min_len:]) / asset2_close_history[-min_len:-1]
            try:
                returns_corr = np.corrcoef(returns1, returns2)[0, 1]
                features['returns_correlation'] = float(returns_corr) if not np.isnan(returns_corr) else 0
            except:
                features['returns_correlation'] = 0
    
    # 10. 市场状态特征
    features['zscore_abs_latest'] = abs(features.get('zscore_latest', 0))
    features['beta_stability'] = 1 / (1 + features.get('beta_std', 1))  # Beta稳定性指标
    
    # 11. 当前入场信号特征（从sample中获取）
    features['signal_strength'] = float(sample.get('signal_strength', 0))
    features['current_zscore'] = float(sample.get('current_zscore', 0))
    features['current_beta'] = float(sample.get('current_beta', 1))
    features['current_alpha'] = float(sample.get('current_alpha', 0))
    features['current_asset1_price'] = float(sample.get('current_asset1_price', 0))
    features['current_asset2_price'] = float(sample.get('current_asset2_price', 0))
    
    # 信号类型编码 (categorical -> numerical)
    signal_type = sample.get('signal_type', '')
    features['signal_type_long'] = 1.0 if signal_type == 'long_spread' else 0.0
    features['signal_type_short'] = 1.0 if signal_type == 'short_spread' else 0.0
    
    # 12. 时间特征（如果有datetime信息）
    if 'datetime_history' in sample and len(sample['datetime_history']) > 0:
        try:
            from datetime import datetime
            dt_str = sample['datetime_history'][-1]
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            features['hour_of_day'] = dt.hour
            features['day_of_week'] = dt.weekday()
        except:
            pass  # 忽略时间解析错误
    
    # 13. 转换为特征向量
    feature_keys = sorted(features.keys())  # 确保特征顺序一致
    feature_vector = np.array([features[key] for key in feature_keys], dtype=np.float32)
    
    # 14. 处理NaN和Inf值
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 15. 提取交易对信息
    trading_pair = sample.get("trade_info", {}).get("pair", "UNKNOWN-UNKNOWN")
    
    return feature_vector, float(profit_percentage), trading_pair


def iter_json_samples(json_path: Path) -> Iterable[Dict]:
    """迭代JSON数组样本（复用v8.0逻辑）"""
    data = json.loads(json_path.read_text())
    for item in data:
        yield item


def get_training_data_file(data_dir: str) -> Path:
    """获取训练数据文件路径"""
    p = Path(data_dir)
    combined_file = p / "all_pairs_ml_training_data.json"
    
    if not combined_file.exists():
        raise FileNotFoundError(f"训练数据文件不存在: {combined_file}")
    
    return combined_file


def load_remove_pairs(path: Optional[str]) -> set:
    """加载需要移除的交易对列表（复用v8.0逻辑）"""
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


def extract_feature_names_from_sample(sample: Dict, lookback: int = 64) -> List[str]:
    """
    从样本中提取特征名称列表（用于模型训练时的特征名设定）
    
    通过实际执行特征提取获取准确的特征名称列表
    """
    result = extract_lgbm_features_from_sample(sample, lookback=lookback)
    if result is None:
        return []
    
    # 执行一次特征提取获取实际的特征名称
    features, _, _ = result
    
    # 从结果中推导特征名称（这需要与extract_lgbm_features_from_sample的逻辑保持一致）
    dummy_sample_result = extract_lgbm_features_from_sample(sample, lookback=lookback)
    if dummy_sample_result is None:
        return []
    
    # 由于我们无法直接从特征向量获取名称，我们需要重新构建
    # 这是一个简化的实现，实际中建议直接在特征提取函数中返回特征名
    feature_names = []
    
    # 根据extract_lgbm_features_from_sample的逻辑构建特征名称
    # Asset1 OHLCV特征
    ohlcv_feature_names = [
        'close_mean', 'close_std', 'close_latest', 'close_min', 'close_max', 'close_range', 'close_cv',
        'avg_daily_range', 'daily_range_std', 'avg_oc_ratio', 'oc_ratio_std', 
        'avg_upper_shadow', 'avg_lower_shadow',
        'returns_mean', 'returns_std', 'returns_skew', 'returns_kurt', 'positive_return_ratio',
        'ma5', 'ma10', 'ma20', 'ma5_ratio', 'ma10_ratio', 'ma20_ratio', 
        'ma5_trend', 'ma10_trend',
        'volume_mean', 'volume_std', 'volume_latest', 'volume_ratio', 'volume_cv', 'volume_trend'
    ]
    
    for name in ohlcv_feature_names:
        feature_names.extend([f'asset1_{name}', f'asset2_{name}'])
    
    # 价格比值特征
    ratio_features = [
        'price_ratio_mean', 'price_ratio_std', 'price_ratio_latest', 'price_ratio_cv',
        'price_ratio_returns_mean', 'price_ratio_returns_std', 'price_ratio_returns_skew',
        'price_ratio_ma5', 'price_ratio_ma5_ratio'
    ]
    feature_names.extend(ratio_features)
    
    # Kalman参数特征
    kalman_params = ['zscore', 'beta', 'alpha']
    kalman_suffixes = ['mean', 'std', 'latest', 'min', 'max', 'range', 'trend']
    for param in kalman_params:
        for suffix in kalman_suffixes:
            feature_names.append(f'{param}_{suffix}')
    
    # 交叉特征
    feature_names.extend(['price_correlation', 'volume_correlation', 'returns_correlation'])
    
    # 市场状态特征
    feature_names.extend(['zscore_abs_latest', 'beta_stability'])
    
    # 当前信号特征
    signal_features = [
        'signal_strength', 'current_zscore', 'current_beta', 'current_alpha',
        'current_asset1_price', 'current_asset2_price',
        'signal_type_long', 'signal_type_short'
    ]
    feature_names.extend(signal_features)
    
    # 时间特征
    feature_names.extend(['hour_of_day', 'day_of_week'])
    
    # 过滤掉可能不存在的特征（根据数据可用性）
    # 返回排序后的特征名称以确保一致性
    return sorted(feature_names)
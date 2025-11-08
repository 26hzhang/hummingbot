"""
V9.0 LightGBM Dataset Loader
基于v8.0的dataset.py扩展，针对LightGBM优化数据加载
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 导入v9.0的特征预处理器
from lgbm_feature_preprocessor import (
    extract_lgbm_features_from_sample,
    extract_feature_names_from_sample,
    get_training_data_file,
    load_remove_pairs,
)
from lgbm_data_loader import iter_json_samples_efficient


class LightGBMStatArbDataset:
    """
    为LightGBM优化的统计套利数据集加载器
    
    与v8.0的主要区别：
    1. 加载为pandas DataFrame而非PyTorch Tensor
    2. 特征为标量向量而非时序数据
    3. 支持特征名称管理
    4. 内存效率优化
    """

    def __init__(
        self,
        data_file: str,
        remove_pairs: Optional[set] = None,
        lookback: int = 64,
        use_technical_indicators: bool = True,
        clamp_target: Optional[Tuple[float, float]] = None,
        max_samples_per_file: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        """
        初始化LightGBM数据集
        
        Args:
            data_file: 训练数据文件路径
            remove_pairs: 需要排除的交易对集合
            lookback: 历史数据回望期
            use_technical_indicators: 是否使用技术指标
            clamp_target: 目标值裁剪范围
            max_samples_per_file: 最大样本数限制
            random_state: 随机种子
        """
        self.data_file = data_file
        self.remove_pairs = remove_pairs or set()
        self.lookback = lookback
        self.use_technical_indicators = use_technical_indicators
        self.clamp_target = clamp_target
        self.max_samples_per_file = max_samples_per_file
        self.random_state = random_state
        
        # 数据存储
        self.features_df = None
        self.targets = None
        self.pairs = None
        self.feature_names = None
        
        # 统计信息
        self.stats = {}
        
        # 加载数据
        self._load_all_data()
        self._compute_statistics()

    def _load_all_data(self) -> None:
        """从all_pairs_ml_training_data.json加载数据"""
        # 使用指定的训练数据文件
        training_file = Path(self.data_file)
        if not training_file.exists():
            print(f"❌ 训练文件不存在: {training_file}")
            return
        file_size_mb = training_file.stat().st_size / (1024 * 1024)
        
        print(f"Loading LightGBM data from training file...")
        print(f"   文件: {training_file.name} ({file_size_mb:.1f}MB)")
        
        all_features = []
        all_targets = []
        all_pairs = []
        feature_names_extracted = False
        
        count = 0
        # 使用高效迭代器处理大文件
        sample_iter = iter_json_samples_efficient(training_file, max_samples=self.max_samples_per_file)
        
        for sample in sample_iter:
            # 提取LightGBM特征
            result = extract_lgbm_features_from_sample(
                sample,
                lookback=self.lookback,
                use_technical_indicators=self.use_technical_indicators,
                clamp_target=self.clamp_target,
            )
            if result is None:
                continue
                
            features, target, pair = result
            
            # 过滤交易对
            if pair in self.remove_pairs:
                continue
            
            # 提取特征名称（只需要一次）
            if not feature_names_extracted:
                self.feature_names = extract_feature_names_from_sample(
                    sample, lookback=self.lookback
                )
                if len(self.feature_names) != len(features):
                    # 如果特征名称数量不匹配，使用默认名称
                    self.feature_names = [f'feature_{i}' for i in range(len(features))]
                feature_names_extracted = True
            
            all_features.append(features)
            all_targets.append(target)
            all_pairs.append(pair)
            count += 1
            
            # 显示进度
            if count % 1000 == 0:
                print(f"   已加载 {count:,} 个样本...")
            
            if self.max_samples_per_file is not None and count >= self.max_samples_per_file:
                break
        
        if all_features:
            # 转换为numpy数组和DataFrame
            features_array = np.vstack(all_features)
            self.features_df = pd.DataFrame(features_array, columns=self.feature_names)
            self.targets = np.array(all_targets)
            self.pairs = np.array(all_pairs)
        else:
            # 空数据集
            self.features_df = pd.DataFrame()
            self.targets = np.array([])
            self.pairs = np.array([])
            self.feature_names = []

        print(f"Loaded {len(all_features)} valid samples with {len(self.feature_names)} features")
    
    def _compute_statistics(self) -> None:
        """计算数据集统计信息"""
        if self.features_df.empty:
            self.stats = {
                'total_samples': 0,
                'unique_pairs': 0,
                'pair_list': [],
                'num_features': 0,
            }
            return
            
        unique_pairs = list(set(self.pairs))
        
        self.stats = {
            'total_samples': len(self.targets),
            'unique_pairs': len(unique_pairs),
            'pair_list': sorted(unique_pairs),
            'num_features': len(self.feature_names),
            'target_stats': {
                'mean': float(np.mean(self.targets)),
                'std': float(np.std(self.targets)),
                'min': float(np.min(self.targets)),
                'max': float(np.max(self.targets)),
                'median': float(np.median(self.targets)),
                'q25': float(np.percentile(self.targets, 25)),
                'q75': float(np.percentile(self.targets, 75)),
            },
            'pair_counts': {pair: list(self.pairs).count(pair) for pair in unique_pairs},
            'feature_stats': {
                'feature_names': self.feature_names,
                'feature_means': self.features_df.mean().to_dict(),
                'feature_stds': self.features_df.std().to_dict(),
            }
        }
    
    def get_statistics(self) -> Dict:
        """返回数据集统计信息"""
        return self.stats
    
    def print_statistics(self) -> None:
        """打印格式化的数据集统计信息"""
        stats = self.stats
        print("\n=== LightGBM Dataset Statistics ===")
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Unique pairs: {stats['unique_pairs']}")
        print(f"Number of features: {stats['num_features']}")
        
        if 'target_stats' in stats:
            target_stats = stats['target_stats']
            print(f"\nTarget (profit_percentage) statistics:")
            print(f"  Mean: {target_stats['mean']:.4f}")
            print(f"  Std:  {target_stats['std']:.4f}")
            print(f"  Median: {target_stats['median']:.4f}")
            print(f"  Min:  {target_stats['min']:.4f}")
            print(f"  Max:  {target_stats['max']:.4f}")
            print(f"  Q25:  {target_stats['q25']:.4f}")
            print(f"  Q75:  {target_stats['q75']:.4f}")
        print("=" * 35)
    
    def get_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        获取完整数据集
        
        Returns:
            Tuple of (features_df, targets, pairs)
        """
        return self.features_df.copy(), self.targets.copy(), self.pairs.copy()
    
    def get_numpy_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取numpy格式数据（用于sklearn兼容）
        
        Returns:
            Tuple of (X, y)
        """
        return self.features_df.values, self.targets
    
    def train_val_split(self, 
                       val_ratio: float = 0.2, 
                       stratify_by_pairs: bool = True,
                       shuffle: bool = True) -> Tuple[Tuple[pd.DataFrame, np.ndarray], Tuple[pd.DataFrame, np.ndarray]]:
        """
        划分训练集和验证集
        
        Args:
            val_ratio: 验证集比例
            stratify_by_pairs: 是否按交易对分层采样
            shuffle: 是否随机打乱
            
        Returns:
            ((X_train, y_train), (X_val, y_val))
        """
        if self.features_df.empty:
            return (pd.DataFrame(), np.array([])), (pd.DataFrame(), np.array([]))
        
        stratify = self.pairs if stratify_by_pairs and len(set(self.pairs)) > 1 else None
        
        train_idx, val_idx = train_test_split(
            range(len(self.targets)),
            test_size=val_ratio,
            random_state=self.random_state,
            shuffle=shuffle,
            stratify=stratify
        )
        
        X_train = self.features_df.iloc[train_idx]
        y_train = self.targets[train_idx]
        X_val = self.features_df.iloc[val_idx]
        y_val = self.targets[val_idx]
        
        print(f"Train/Val split: {len(train_idx):,} / {len(val_idx):,} samples")
        
        return (X_train, y_train), (X_val, y_val)
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names.copy()
    
    def filter_by_pairs(self, allowed_pairs: List[str]) -> 'LightGBMStatArbDataset':
        """
        按交易对过滤数据集
        
        Args:
            allowed_pairs: 允许的交易对列表
            
        Returns:
            过滤后的新数据集实例
        """
        if self.features_df.empty:
            return self
        
        mask = np.isin(self.pairs, allowed_pairs)
        
        # 创建新实例
        new_dataset = LightGBMStatArbDataset.__new__(LightGBMStatArbDataset)
        new_dataset.data_file = self.data_file
        new_dataset.remove_pairs = self.remove_pairs
        new_dataset.lookback = self.lookback
        new_dataset.use_technical_indicators = self.use_technical_indicators
        new_dataset.clamp_target = self.clamp_target
        new_dataset.random_state = self.random_state
        
        # 过滤数据
        new_dataset.features_df = self.features_df.iloc[mask].reset_index(drop=True)
        new_dataset.targets = self.targets[mask]
        new_dataset.pairs = self.pairs[mask]
        new_dataset.feature_names = self.feature_names.copy()
        
        # 重新计算统计信息
        new_dataset._compute_statistics()
        
        print(f"Filtered dataset: {np.sum(mask):,} / {len(mask):,} samples kept")
        
        return new_dataset
    
    def get_top_features_by_variance(self, top_n: int = 50) -> List[str]:
        """
        按方差选择最重要的特征
        
        Args:
            top_n: 返回前N个特征
            
        Returns:
            特征名称列表
        """
        if self.features_df.empty:
            return []
        
        variances = self.features_df.var()
        top_features = variances.nlargest(top_n).index.tolist()
        
        return top_features
    
    def remove_low_variance_features(self, threshold: float = 0.01) -> 'LightGBMStatArbDataset':
        """
        移除低方差特征
        
        Args:
            threshold: 方差阈值
            
        Returns:
            特征过滤后的新数据集
        """
        if self.features_df.empty:
            return self
        
        variances = self.features_df.var()
        high_var_features = variances[variances > threshold].index.tolist()
        
        # 创建新实例
        new_dataset = LightGBMStatArbDataset.__new__(LightGBMStatArbDataset)
        new_dataset.data_file = self.data_file
        new_dataset.remove_pairs = self.remove_pairs
        new_dataset.lookback = self.lookback
        new_dataset.use_technical_indicators = self.use_technical_indicators
        new_dataset.clamp_target = self.clamp_target
        new_dataset.random_state = self.random_state
        
        # 过滤特征
        new_dataset.features_df = self.features_df[high_var_features]
        new_dataset.targets = self.targets.copy()
        new_dataset.pairs = self.pairs.copy()
        new_dataset.feature_names = high_var_features
        
        # 重新计算统计信息
        new_dataset._compute_statistics()
        
        print(f"Removed low variance features: {len(high_var_features)} / {len(self.feature_names)} features kept")
        
        return new_dataset
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.targets)
    
    def __getitem__(self, idx) -> Tuple[pd.Series, float]:
        """获取单个样本"""
        if self.features_df.empty:
            raise IndexError("Dataset is empty")
        
        return self.features_df.iloc[idx], self.targets[idx]
    
    def summary(self) -> str:
        """返回数据集摘要字符串"""
        stats = self.stats
        
        summary_str = f"""
LightGBM StatArb Dataset Summary:
- Total samples: {stats['total_samples']:,}
- Number of features: {stats.get('num_features', 0)}
- Unique trading pairs: {stats['unique_pairs']}
- Target range: [{stats.get('target_stats', {}).get('min', 'N/A'):.4f}, {stats.get('target_stats', {}).get('max', 'N/A'):.4f}]
- Target mean: {stats.get('target_stats', {}).get('mean', 'N/A'):.4f}
- Data file: {self.data_file}
"""
        return summary_str.strip()


def create_dataset_from_config(config: Dict) -> LightGBMStatArbDataset:
    """
    从配置字典创建数据集
    
    Args:
        config: 数据集配置
        
    Returns:
        数据集实例
    """
    remove_pairs = set()
    if 'remove_pairs_file' in config and config['remove_pairs_file']:
        remove_pairs = load_remove_pairs(config['remove_pairs_file'])
    
    dataset = LightGBMStatArbDataset(
        data_file=config['data_file'],
        remove_pairs=remove_pairs,
        lookback=config.get('lookback', 64),
        use_technical_indicators=config.get('use_technical_indicators', True),
        clamp_target=config.get('clamp_target', None),
        max_samples_per_file=config.get('max_samples_per_file', None),
        random_state=config.get('random_state', 42),
    )
    
    return dataset
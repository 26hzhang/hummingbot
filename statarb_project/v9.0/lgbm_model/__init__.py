"""
V9.0 LightGBM Model Package
统计套利LightGBM入场质量评估模型
"""

from .lgbm_entry_model import LightGBMEntryQualityModel, LightGBMBinaryClassifier
from .lgbm_feature_preprocessor import (
    extract_lgbm_features_from_sample,
    extract_feature_names_from_sample,
    iter_json_samples,
    get_training_data_file,
    load_remove_pairs,
)
from .lgbm_dataset import LightGBMStatArbDataset, create_dataset_from_config

__version__ = "9.0.0"
__author__ = "AI Assistant"

__all__ = [
    # 模型类
    "LightGBMEntryQualityModel",
    "LightGBMBinaryClassifier",
    
    # 特征处理函数
    "extract_lgbm_features_from_sample",
    "extract_feature_names_from_sample",
    "iter_json_samples", 
    "get_training_data_file",
    "load_remove_pairs",
    
    # 数据集类
    "LightGBMStatArbDataset",
    "create_dataset_from_config",
]
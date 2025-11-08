#!/usr/bin/env python3
"""
V9.0快速测试 - 使用少量样本验证功能
"""

import sys
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent / "lgbm_model"))

from lgbm_model.lgbm_dataset import create_dataset_from_config


def quick_test():
    """快速测试数据加载和特征提取"""
    print("🚀 V9.0 LightGBM快速功能测试")
    print("=" * 50)
    
    # 配置：限制样本数量进行快速测试
    dataset_config = {
        'data_file': "/Users/zhanghao/GitHub/hummingbot/statarb_project/v8.0/all_pairs_ml_training_data.json",
        'remove_pairs_file': None,
        'lookback': 64,
        'use_technical_indicators': True,
        'clamp_target': None,
        'max_samples_per_file': 100,  # 只取100个样本进行快速测试
        'random_state': 42,
    }
    
    try:
        print("📊 创建数据集...")
        dataset = create_dataset_from_config(dataset_config)
        
        print("📈 数据集统计:")
        dataset.print_statistics()
        
        if len(dataset) > 0:
            print("\n✅ 数据加载成功！")
            print(f"📊 特征数量: {len(dataset.get_feature_names())}")
            
            # 测试训练/验证分割
            (X_train, y_train), (X_val, y_val) = dataset.train_val_split(val_ratio=0.2)
            print(f"📈 训练集: {len(X_train)} 样本")
            print(f"📉 验证集: {len(X_val)} 样本")
            
            # 显示特征名称示例
            feature_names = dataset.get_feature_names()
            print(f"\n🏷️ 特征名称示例 (前10个):")
            for i, name in enumerate(feature_names[:10]):
                print(f"  {i+1:2d}. {name}")
            
            print("\n🎉 快速测试通过！系统功能正常。")
            return True
        else:
            print("\n❌ 没有加载到有效样本")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
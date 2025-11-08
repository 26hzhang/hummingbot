#!/usr/bin/env python3
"""
V9.0快速训练示例 - 使用少量样本训练LightGBM模型
"""

import sys
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent / "lgbm_model"))

from lgbm_model.lgbm_trainer import train_lgbm_model


def quick_train():
    """使用少量样本快速训练LightGBM模型"""
    print("🚀 V9.0 LightGBM快速训练示例")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "models"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 训练参数 - 使用少量样本和简单参数进行快速测试
        train_lgbm_model(
            data_file="/Users/zhanghao/GitHub/hummingbot/statarb_project/v8.0/all_pairs_ml_training_data.json",
            remove_pairs_file=None,
            lookback=64,
            val_ratio=0.2,
            max_samples_per_file=1000,  # 只用1000个样本进行快速训练
            model_out=str(output_dir / "lgbm_quick_test.joblib"),
            random_state=42,
            # LightGBM超参数 - 快速训练设置
            n_estimators=100,  # 减少树的数量
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            early_stopping_rounds=20,  # 减少早停轮数
            verbose_eval=10,  # 更频繁的日志
        )
        
        print("\n🎉 快速训练完成！")
        print(f"📁 模型保存路径: {output_dir / 'lgbm_quick_test.joblib'}")
        return True
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_train()
    sys.exit(0 if success else 1)
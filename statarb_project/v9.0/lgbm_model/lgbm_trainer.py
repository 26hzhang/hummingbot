"""
V9.0 LightGBM Model Trainer
基于v8.0的entry_model_trainer.py重写，使用LightGBM替代GRU模型
"""
import argparse
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入v9.0 LightGBM组件
from lgbm_dataset import create_dataset_from_config
from lgbm_entry_model import LightGBMEntryQualityModel


def set_global_seed(seed: int) -> None:
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Global random seed set to: {seed}")


def train_lgbm_model(
    data_file: str,
    remove_pairs_file: Optional[str] = None,
    lookback: int = 64,
    val_ratio: float = 0.2,
    max_samples_per_file: Optional[int] = None,
    model_out: str = "lgbm_entry_model.joblib",
    random_state: int = 42,
    # LightGBM 超参数
    n_estimators: int = 500,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    feature_fraction: float = 0.8,
    bagging_fraction: float = 0.8,
    bagging_freq: int = 5,
    min_child_samples: int = 20,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 50,
    clamp_target: Optional[tuple] = None,
) -> None:
    """
    训练LightGBM入场质量预测模型
    
    Args:
        data_file: 训练数据文件路径
        remove_pairs_file: 需要排除的交易对文件
        lookback: 历史数据回望期
        val_ratio: 验证集比例
        max_samples_per_file: 最大样本数限制
        model_out: 模型输出路径
        random_state: 随机种子
        n_estimators: LightGBM树的数量
        learning_rate: 学习率
        num_leaves: 叶子节点数
        feature_fraction: 特征采样比例
        bagging_fraction: 样本采样比例
        bagging_freq: bagging频率
        min_child_samples: 叶子节点最小样本数
        reg_alpha: L1正则化
        reg_lambda: L2正则化
        early_stopping_rounds: 早停轮数
        verbose_eval: 训练日志频率
        clamp_target: 目标值裁剪范围
    """
    # 设置随机种子
    set_global_seed(random_state)
    
    print("=" * 80)
    print("🚀 LightGBM Entry Quality Model Training")
    print("=" * 80)
    
    # 创建数据集配置
    dataset_config = {
        'data_file': data_file,
        'remove_pairs_file': remove_pairs_file,
        'lookback': lookback,
        'use_technical_indicators': True,
        'clamp_target': clamp_target,
        'max_samples_per_file': max_samples_per_file,
        'random_state': random_state,
    }
    
    # 加载数据集
    print("📊 Loading dataset...")
    start_time = time.time()
    dataset = create_dataset_from_config(dataset_config)
    load_time = time.time() - start_time
    
    # 打印数据集统计
    dataset.print_statistics()
    print(f"Data loading time: {load_time:.2f} seconds")
    
    if len(dataset) == 0:
        raise RuntimeError("No training samples found. Check data directory and remove-pairs filter.")
    
    # 获取特征名称
    feature_names = dataset.get_feature_names()
    print(f"Number of features: {len(feature_names)}")
    
    # 数据划分
    print(f"\n📈 Splitting dataset: {1-val_ratio:.1%} train, {val_ratio:.1%} validation")
    (X_train, y_train), (X_val, y_val) = dataset.train_val_split(
        val_ratio=val_ratio, 
        stratify_by_pairs=True,
        shuffle=True
    )
    
    print(f"Train samples: {len(X_train):,}")
    print(f"Val samples: {len(X_val):,}")
    print(f"Train target mean: {y_train.mean():.4f}")
    print(f"Val target mean: {y_val.mean():.4f}")
    
    # 创建和配置LightGBM模型
    print(f"\n🤖 Creating LightGBM model...")
    model = LightGBMEntryQualityModel(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        min_child_samples=min_child_samples,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        verbose=-1,
    )
    
    # 训练模型
    print(f"\n🏋️ Training LightGBM model...")
    train_start_time = time.time()
    
    model.fit(
        X=X_train.values,
        y=y_train,
        feature_names=feature_names,
        eval_set=[(X_val.values, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    
    train_time = time.time() - train_start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # 模型评估
    print(f"\n📊 Evaluating model...")
    
    # 训练集预测
    y_train_pred = model.predict(X_train.values)
    train_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred),
        'correlation': np.corrcoef(y_train, y_train_pred)[0, 1]
    }
    
    # 验证集预测
    y_val_pred = model.predict(X_val.values)
    val_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'mae': mean_absolute_error(y_val, y_val_pred),
        'r2': r2_score(y_val, y_val_pred),
        'correlation': np.corrcoef(y_val, y_val_pred)[0, 1]
    }
    
    # 打印评估结果
    print(f"\n📈 Training Metrics:")
    print(f"  RMSE: {train_metrics['rmse']:.6f}")
    print(f"  MAE:  {train_metrics['mae']:.6f}")
    print(f"  R²:   {train_metrics['r2']:.6f}")
    print(f"  Corr: {train_metrics['correlation']:.6f}")
    
    print(f"\n📉 Validation Metrics:")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  MAE:  {val_metrics['mae']:.6f}")
    print(f"  R²:   {val_metrics['r2']:.6f}")
    print(f"  Corr: {val_metrics['correlation']:.6f}")
    
    # 特征重要性分析
    print(f"\n🔍 Top 20 Feature Importance:")
    feature_importance_df = model.get_feature_importance(importance_type='gain')
    print(feature_importance_df.head(20).to_string(index=False))
    
    # 保存模型
    print(f"\n💾 Saving model...")
    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save_model(out_path)
    
    # 保存训练信息
    training_info = {
        'dataset_stats': dataset.get_statistics(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'feature_importance': feature_importance_df.head(50).to_dict('records'),
        'training_config': {
            'data_file': data_file,
            'remove_pairs_file': remove_pairs_file,
            'lookback': lookback,
            'val_ratio': val_ratio,
            'max_samples_per_file': max_samples_per_file,
            'random_state': random_state,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'early_stopping_rounds': early_stopping_rounds,
            'clamp_target': clamp_target,
        },
        'training_time_seconds': train_time,
        'load_time_seconds': load_time,
    }
    
    info_path = out_path.with_suffix('.json')
    import json
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2, default=str)
    
    print(f"✅ Model saved to: {out_path}")
    print(f"📄 Training info saved to: {info_path}")
    print(f"📊 Final validation R²: {val_metrics['r2']:.6f}")
    print(f"🔗 Final validation correlation: {val_metrics['correlation']:.6f}")
    
    # 推理速度测试
    print(f"\n⚡ Testing inference speed...")
    test_sample = X_val.iloc[0].values.reshape(1, -1)
    
    # 预热
    for _ in range(10):
        model.predict(test_sample)
    
    # 测试推理时间
    inference_times = []
    for _ in range(100):
        start = time.time()
        model.predict(test_sample)
        inference_times.append(time.time() - start)
    
    avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
    print(f"Average inference time: {avg_inference_time:.2f}ms per prediction")
    
    if avg_inference_time < 10:
        print("✅ Inference speed meets real-time requirements (<10ms)")
    else:
        print("⚠️ Inference speed may be too slow for real-time trading")
    
    print("\n🎉 Training completed successfully!")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Train LightGBM model for stat-arb entry quality prediction")
    
    # 数据相关参数
    parser.add_argument("--data-file", type=str, 
                       default="/Users/zhanghao/GitHub/hummingbot/statarb_project/v8.0/all_pairs_ml_training_data.json",
                       help="Path to the training data JSON file")
    parser.add_argument("--remove-pairs-file", type=str, default=None,
                       help="File containing pairs to remove from training")
    parser.add_argument("--lookback", type=int, default=64,
                       help="Historical lookback period for features")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="Validation set ratio")
    parser.add_argument("--max-samples-per-file", type=int, default=None,
                       help="Maximum samples to load from training file (for testing)")
    parser.add_argument("--clamp-target-min", type=float, default=None,
                       help="Minimum target value (clipping)")
    parser.add_argument("--clamp-target-max", type=float, default=None,
                       help="Maximum target value (clipping)")
    
    # 模型相关参数
    parser.add_argument("--model-out", type=str, required=True,
                       help="Output path for trained model")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # LightGBM超参数
    parser.add_argument("--n-estimators", type=int, default=500,
                       help="Number of boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                       help="Learning rate")
    parser.add_argument("--num-leaves", type=int, default=31,
                       help="Number of leaves in tree")
    parser.add_argument("--feature-fraction", type=float, default=0.8,
                       help="Feature sampling ratio")
    parser.add_argument("--bagging-fraction", type=float, default=0.8,
                       help="Sample bagging ratio")
    parser.add_argument("--bagging-freq", type=int, default=5,
                       help="Bagging frequency")
    parser.add_argument("--min-child-samples", type=int, default=20,
                       help="Minimum samples in leaf")
    parser.add_argument("--reg-alpha", type=float, default=0.1,
                       help="L1 regularization")
    parser.add_argument("--reg-lambda", type=float, default=0.1,
                       help="L2 regularization")
    parser.add_argument("--early-stopping-rounds", type=int, default=50,
                       help="Early stopping rounds")
    parser.add_argument("--verbose-eval", type=int, default=50,
                       help="Verbose evaluation frequency")
    
    args = parser.parse_args()
    
    # 处理目标裁剪参数
    clamp_target = None
    if args.clamp_target_min is not None or args.clamp_target_max is not None:
        clamp_target = (
            args.clamp_target_min if args.clamp_target_min is not None else -np.inf,
            args.clamp_target_max if args.clamp_target_max is not None else np.inf
        )
    
    # 训练模型
    train_lgbm_model(
        data_file=args.data_file,
        remove_pairs_file=args.remove_pairs_file,
        lookback=args.lookback,
        val_ratio=args.val_ratio,
        max_samples_per_file=args.max_samples_per_file,
        model_out=args.model_out,
        random_state=args.random_seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        min_child_samples=args.min_child_samples,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
        clamp_target=clamp_target,
    )


if __name__ == "__main__":
    main()
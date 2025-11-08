"""
V9.0 LightGBM Entry Quality Model
基于LightGBM的轻量级入场质量评估模型，替代v8.0的GRU模型
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


class LightGBMEntryQualityModel:
    """
    基于LightGBM的入场质量评估模型包装器
    
    功能特点：
    1. 轻量级：模型文件小，推理速度快（<1ms）
    2. 可解释性：支持特征重要性分析
    3. 稳健性：对缺失值和异常值有较好的鲁棒性
    4. 易部署：无需深度学习框架，仅需lightgbm
    """
    
    def __init__(self, 
                 objective: str = 'regression',
                 metric: str = 'rmse',
                 boosting_type: str = 'gbdt',
                 num_leaves: int = 31,
                 learning_rate: float = 0.1,
                 feature_fraction: float = 0.8,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 5,
                 verbose: int = -1,
                 random_state: int = 42,
                 n_estimators: int = 100,
                 **kwargs):
        """
        初始化LightGBM模型
        
        Args:
            objective: 目标函数（'regression' 或 'binary'）
            metric: 评估指标
            boosting_type: 提升类型
            num_leaves: 叶子节点数
            learning_rate: 学习率
            feature_fraction: 特征采样比例
            bagging_fraction: 样本采样比例
            bagging_freq: bagging频率
            verbose: 日志级别
            random_state: 随机种子
            n_estimators: 树的数量
            **kwargs: 其他LightGBM参数
        """
        self.params = {
            'objective': objective,
            'metric': metric,
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': verbose,
            'random_state': random_state,
            'n_estimators': n_estimators,
            **kwargs
        }
        
        self.model = None
        self.feature_names = None
        self.feature_importances_ = None
        self.is_trained = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            eval_set: Optional[List[tuple]] = None,
            early_stopping_rounds: Optional[int] = None,
            verbose_eval: Optional[int] = None) -> 'LightGBMEntryQualityModel':
        """
        训练模型
        
        Args:
            X: 训练特征 [n_samples, n_features]
            y: 训练标签 [n_samples]
            feature_names: 特征名称列表
            eval_set: 验证集 [(X_val, y_val)]
            early_stopping_rounds: 早停轮数
            verbose_eval: 训练日志频率
            
        Returns:
            训练好的模型实例
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if eval_set is not None:
            for i, (X_val, y_val) in enumerate(eval_set):
                val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)
                valid_sets.append(val_data)
                valid_names.append(f'valid_{i}')
        
        # 训练模型
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds) if early_stopping_rounds else None,
                lgb.log_evaluation(verbose_eval) if verbose_eval else None,
            ]
        )
        
        # 保存特征重要性
        self.feature_importances_ = self.model.feature_importance(importance_type='gain')
        self.is_trained = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        模型预测
        
        Args:
            X: 输入特征 [n_samples, n_features]
            
        Returns:
            预测结果 [n_samples]
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用fit()方法")
        
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        return predictions
    
    def predict_single(self, features: np.ndarray) -> float:
        """
        单样本预测（实盘使用）
        
        Args:
            features: 单个样本特征向量 [n_features]
            
        Returns:
            预测的入场质量评分
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        prediction = self.predict(features)[0]
        return float(prediction)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型 ('gain', 'split', 'cover')
            
        Returns:
            特征重要性DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        importances = self.model.feature_importance(importance_type=importance_type)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save_model(self, file_path: Union[str, Path]) -> None:
        """
        保存模型到文件
        
        Args:
            file_path: 保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importances_': self.feature_importances_,
            'params': self.params,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        print(f"✅ 模型已保存至: {file_path}")
    
    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> 'LightGBMEntryQualityModel':
        """
        从文件加载模型
        
        Args:
            file_path: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
        
        model_data = joblib.load(file_path)
        
        # 创建模型实例
        instance = cls(**model_data['params'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.feature_importances_ = model_data['feature_importances_']
        instance.is_trained = model_data['is_trained']
        
        return instance
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'num_features': len(self.feature_names),
            'num_trees': self.model.num_trees(),
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'feature_names': self.feature_names,
            'params': self.params,
        }
    
    def plot_feature_importance(self, top_n: int = 20, figsize: tuple = (10, 8)):
        """
        绘制特征重要性图
        
        Args:
            top_n: 显示前N个重要特征
            figsize: 图片大小
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            importance_df = self.get_feature_importance().head(top_n)
            
            plt.figure(figsize=figsize)
            sns.barplot(data=importance_df, x='importance', y='feature', orient='h')
            plt.title(f'Top {top_n} Feature Importance')
            plt.xlabel('Importance (Gain)')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("⚠️ 需要安装matplotlib和seaborn才能绘制图表")
            return self.get_feature_importance().head(top_n)
    
    def evaluate_on_validation(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        在验证集上评估模型
        
        Args:
            X_val: 验证集特征
            y_val: 验证集标签
            
        Returns:
            评估指标字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        y_pred = self.predict(X_val)
        
        # 计算回归指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred),
            'correlation': np.corrcoef(y_val, y_pred)[0, 1]
        }
        
        return metrics


class LightGBMBinaryClassifier(LightGBMEntryQualityModel):
    """
    LightGBM二分类模型（如果需要二分类而非回归）
    """
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        """
        初始化二分类模型
        
        Args:
            threshold: 分类阈值
            **kwargs: 传递给父类的参数
        """
        kwargs['objective'] = 'binary'
        kwargs['metric'] = 'binary_logloss'
        super().__init__(**kwargs)
        self.threshold = threshold
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        probabilities = self.model.predict(X, num_iteration=self.model.best_iteration)
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        probabilities = self.predict_proba(X)
        return (probabilities > self.threshold).astype(int)
    
    def predict_single_proba(self, features: np.ndarray) -> float:
        """单样本概率预测"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        probability = self.predict_proba(features)[0]
        return float(probability)
    
    def evaluate_on_validation(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """二分类评估"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        y_pred_proba = self.predict_proba(X_val)
        y_pred = self.predict(X_val)
        
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, average_precision_score)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_pred_proba),
            'pr_auc': average_precision_score(y_val, y_pred_proba)
        }
        
        return metrics
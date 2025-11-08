# V8.0 Statistical Arbitrage Deep Learning Entry Quality Filter

## 📋 项目背景

### 问题描述
基于V7.0的Kalman滤波统计套利策略已实现基本功能，但在实盘交易中存在以下痛点：
1. **入场时机优化不足**: 仅依赖Z-Score阈值触发，无法识别高质量入场机会
2. **Beta稳定性过滤有限**: 现有Beta波动率过滤机制较为简单
3. **市场状态感知缺乏**: 未考虑更复杂的市场微观结构特征
4. **实盘适配性要求**: 需要轻量级、快速响应的智能过滤机制

### 核心需求
- **实盘导向**: 模型仅在Kalman产生入场信号时调用，不干扰主策略逻辑
- **特征纯净性**: 严格使用t-1时刻及之前的历史信息，避免数据泄漏  
- **轻量级推理**: 模型结构简单，推理时间<10ms，适合实盘频繁调用
- **质量评估**: 输出0-1评分，表示该入场点的预期质量

## 🎯 设计目标

### 核心定位
**入场质量过滤器**: 不是信号生成器，而是对Kalman信号的智能增强过滤

### 工作流程
```
市场数据更新 → Kalman Filter计算Z-Score → 触发入场阈值? 
    ↓(是)
提取t-1时刻特征向量 → DL模型评估入场质量 → 质量评分>阈值?
    ↓(是)                                        ↓(否)
执行入场交易                                    跳过本次入场
```

### 技术架构
- **特征层**: 基于历史价格、成交量、技术指标的时序特征
- **模型层**: 轻量级GRU/LSTM序列建模，捕获时序模式
- **集成层**: 无缝集成到V7.0 StatArbKalmanStrategy

## 📊 数据基础

### 现有数据资产
- **数据源**: V8.0已收集的ML训练数据 (`all_pairs_ml_training_data.json`, 4.8GB+)
- **样本规模**: 多交易对历史回测数据，包含完整的入场-出场生命周期
- **特征维度**: 64期历史OHLCV + Kalman参数(alpha/beta/zscore)
- **标签质量**: 基于实际盈亏的入场质量标签

### 特征工程重点
```python
# 核心特征类别 (t-1时刻纯净特征)
features = {
    'price_features': {
        'asset1_ohlcv_history': [64],  # 价格序列特征  
        'asset2_ohlcv_history': [64],  # 价格序列特征
        'price_ratio_history': [64],   # 价格比值序列
    },
    'kalman_features': {
        'zscore_history': [64],        # Z-Score时序
        'beta_history': [64],          # Beta参数演化
        'alpha_history': [64],         # Alpha参数演化
    },
    'technical_features': {
        'volatility_indicators': [64], # 波动率指标
        'momentum_indicators': [64],   # 动量指标  
        'volume_patterns': [64],       # 成交量模式
    }
}
```

## 🏗️ 实现方案

### 核心文件架构

#### 1. `entry_quality_preprocessor.py`
**职责**: 数据预处理和特征工程
**扩展**: 基于现有`statarb_ml_data_collector.py`
```python
class EntryQualityPreprocessor:
    """入场质量数据预处理器"""
    
    def __init__(self, base_data_path, lookback_periods=64):
        # 加载现有ML数据
        # 重构特征提取逻辑
    
    def extract_t_minus_1_features(self, sample):
        """提取严格的t-1时刻特征"""
        # 确保无数据泄漏的特征提取
        
    def generate_quality_labels(self, samples):  
        """基于盈亏生成入场质量标签"""
        # profit_percentage -> quality_score [0,1]
```

#### 2. `lightweight_entry_model.py` 
**职责**: 轻量级深度学习模型定义
```python
class LightweightEntryQualityModel(nn.Module):
    """轻量级入场质量评估模型"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        # 简单GRU/LSTM架构
        # 优化推理速度
        
    def forward(self, sequence_features):
        # 输入: [batch, seq_len, features]
        # 输出: [batch, 1] 质量评分
```

#### 3. `entry_model_trainer.py`
**职责**: 模型训练和验证框架  
```python  
class EntryModelTrainer:
    """入场质量模型训练器"""
    
    def __init__(self, model, data_loader):
        # 训练配置和优化器设置
        
    def train_entry_quality_model(self):
        # 训练循环
        # 验证和早停
        # 模型保存和版本控制
```

#### 4. `kalman_ml_strategy.py`
**职责**: 集成到V7.0策略的ML增强版本
```python
class StatArbKalmanMLStrategy(StatArbKalmanStrategy):
    """ML增强的Kalman统计套利策略"""
    
    def __init__(self):
        super().__init__()
        self.entry_quality_model = self._load_quality_model()
        
    def _check_spread_entry_conditions(self, zscore):
        """重写入场条件检查，增加ML质量过滤"""
        # 1. 原有Kalman逻辑
        signal = super()._check_spread_entry_conditions(zscore)  
        if signal is None:
            return None
            
        # 2. ML质量评估  
        t_minus_1_features = self._extract_current_features()
        quality_score = self.entry_quality_model.predict(t_minus_1_features)
        
        # 3. 质量阈值过滤
        if quality_score < self.params.ml_quality_threshold:
            return None  # 跳过低质量入场
            
        return signal
```

## 🚀 开发计划

### Phase 1: 数据准备 (1-2天)
1. **数据重构**: 修改特征提取逻辑，确保t-1时刻特征纯净性
2. **标签生成**: 基于历史盈亏数据生成入场质量标签
3. **数据集切分**: 训练/验证/测试集划分，避免时间泄漏

### Phase 2: 模型开发 (2-3天)  
1. **架构设计**: 实现轻量级GRU/LSTM模型
2. **训练框架**: 建立完整的训练、验证、评估pipeline
3. **超参优化**: 网格搜索最优模型配置

### Phase 3: 集成测试 (1-2天)
1. **策略集成**: 将模型集成到V7.0策略中
2. **回测验证**: 对比ML增强前后的策略表现
3. **性能优化**: 确保推理速度满足实盘要求

## 📈 预期效果

### 性能指标
- **推理延迟**: <10ms per prediction
- **模型大小**: <5MB (便于部署)
- **准确性**: 质量评分与实际盈亏相关性>0.6

### 策略改进  
- **入场精度**: 预期提升20-30%的入场成功率
- **风险控制**: 减少低质量入场导致的损失
- **夏普比率**: 预期提升10-20%

## 🔧 技术栈

### 深度学习框架
- **PyTorch**: 主要建模框架
- **PyTorch Lightning**: 训练流程管理
- **ONNX**: 模型部署优化

### 数据处理
- **Pandas**: 数据操作和特征工程
- **NumPy**: 数值计算
- **Scikit-learn**: 数据预处理和评估

### 集成测试
- **Backtrader**: 基于现有V7.0回测框架
- **Matplotlib**: 可视化分析

## ⚠️ 风险控制

### 过拟合防护
- **时间切分**: 严格按时间顺序切分数据集
- **交叉验证**: 多交易对交叉验证
- **正则化**: Dropout和权重衰减

### 实盘适配
- **模型简化**: 避免过于复杂的架构
- **特征稳定**: 选择稳定、可解释的特征
- **降级策略**: ML失败时自动回退到原始Kalman策略

## 📝 成功标准

### 技术指标
1. 模型推理速度 < 10ms
2. 质量评分与盈亏相关性 > 0.6  
3. 集成后策略Sharpe比率提升 > 10%

### 工程质量
1. 代码复用现有V7.0架构 > 80%
2. 单元测试覆盖率 > 90%
3. 文档完整性和可维护性

---

**项目负责人**: AI Assistant  
**创建时间**: 2025-09-01  
**版本**: V8.0  
**状态**: 设计阶段

## 🧪 训练与使用步骤（本版本实现）

本次提交已在 `dl_model/` 下实现了一个基于 PyTorch 的回归版 GRU 模型，用于直接预测入场后的 `profit_percentage`（单位：百分比）。不做数据切片（不滑窗），每个样本的一段历史数组即为一条序列样本。

### 目录与文件
- `entry_quality_preprocessor.py`: 样本级特征构建与清洗
  - 仅使用历史数组（t-1 及更早），不会使用 `current_*` 字段作为输入
  - 特征通道（默认 8 维）：
    - f0: asset1 收盘价对数收益
    - f1: asset2 收盘价对数收益
    - f2: 价差比（asset2/asset1）的对数收益
    - f3: zscore 历史（原值）
    - f4: beta 历史（原值）
    - f5: alpha 历史（原值）
    - f6: asset1 成交量对数变化
    - f7: asset2 成交量对数变化
  - 每条样本的序列长度将被左侧填充/截断到固定 `lookback`（默认 64）
  - 采用“样本内标准化”：对每个样本在时间维度做通道级标准化（均值/标准差）
- `dataset.py`: 可迭代数据集，按文件顺序逐个文件加载 JSON 数组，逐条样本产出张量
- `lightweight_entry_model.py`: 轻量 GRU 回归模型（2 层 GRU + 小 MLP 头）
- `entry_model_trainer.py`: 训练脚本（命令行）

### 训练前准备
- 数据目录：`/storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0/data`（包含 `ml_data_*.json` 文件）
- 剔除交易对列表：`/storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0/remove_pair_list.txt`
  - 训练时会自动过滤这些对（不进入训练样本）

### 训练命令示例

- 使用全部数据（注意：首次可先用 `--max-files` 限制）：
  - 命令：
    - `python /storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0/dl_model/entry_model_trainer.py \
       --data-dir /storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0/data \
       --remove-pairs-file /storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0/remove_pair_list.txt \
       --lookback 64 \
       --batch-size 256 \
       --hidden-dim 64 \
       --num-layers 2 \
       --dropout 0.1 \
       --lr 1e-3 \
       --epochs 5 \
       --max-files 10 \
       --max-samples-per-file 2000 \
       --model-out /storage/tianzichen/sicheng/hummingbot/stat_arb_project/analysis/v8.0/dl_model/entry_quality_gru.pt`

- 参数说明：
  - `--data-dir`: ML样本 JSON 文件目录
  - `--remove-pairs-file`: 需要剔除的交易对列表（每行如 `NEAR-FET`）
  - `--lookback`: 固定序列长度（自动左填充/截断）
  - `--batch-size`: 批大小
  - `--hidden-dim`/`--num-layers`/`--dropout`: GRU 模型超参
  - `--lr`/`--epochs`: 学习率与训练轮数
  - `--max-files`/`--max-samples-per-file`: 首次试跑可限制读取数据量，逐步放开
  - `--model-out`: 模型权重保存路径

### 训练细节说明
- 目标：回归 `profit_percentage`（默认会在 [-5, 5] 范围内裁剪，以增强稳健性）
- 标准化：每条样本在时间维度做通道级标准化（不需要全量统计，避免大文件两遍扫描）
- 优化器与损失：AdamW + HuberLoss（delta=0.5），对异常值更稳健
- 设备：自动检测 CUDA；若有 GPU 会自动使用

### 输出
- `entry_quality_gru.pt`：包含 `state_dict` 与模型配置（输入维度/GRU结构/lookback）

### 后续接入（下一步）
- 在 `kalman_ml_strategy.py` 中加载上述权重，对每次入场信号触发时提取 t-1 特征（与本预处理一致），模型打分为预测收益；据此与阈值比较过滤低质量入场。
- 如需要，我们会提供一个 `infer.py` 脚本用于离线评估预测-真实收益的相关与分位收益曲线。

### 注意事项
- 大文件：当前数据集使用 JSON 数组，读取时会以“逐文件加载+释放”的方式工作；首次建议通过 `--max-files`/`--max-samples-per-file` 限制规模，逐步扩容。
- 字段过滤：预处理明确不将 `current_*` 字段作为模型输入，避免泄露。
- 交易对过滤：训练期自动剔除 `remove_pair_list.txt` 中的对。

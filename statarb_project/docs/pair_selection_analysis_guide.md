# 统计套利配对选择分析完整指南

## 概述

统计套利（Statistical Arbitrage）是一种基于数理统计分析的量化交易策略，核心思想是识别长期具有协整关系的资产对，在其价格偏离长期均衡时进行反向交易，待价格回归均衡时获利。

`pair_selection_analysis.py` 是一个完整的统计套利配对选择分析工具，提供从数据加载到结果输出的全流程自动化分析功能。

## 理论基础

### 协整理论

#### 1. 协整的定义

如果两个时间序列都是非平稳的，但它们的线性组合是平稳的，则称这两个序列存在协整关系。对于价格序列 P1(t) 和 P2(t)，如果存在系数 α 和 β，使得：

```
Spread(t) = ln(P2(t)) - α - β * ln(P1(t))
```

该价差序列是平稳的，则 P1 和 P2 存在协整关系。

#### 2. 平稳性检验

使用增强迪基-福勒检验(ADF Test)检验价差序列的平稳性：

- H0: 序列存在单位根（非平稳）
- H1: 序列不存在单位根（平稳）
- 拒绝H0（p-value < 0.05）则认为序列平稳，存在协整关系

### 滚动窗口分析方法

#### In-Sample (IS) 和 Out-of-Sample (OS) 框架

本分析采用严格的IS/OS验证框架：

**In-Sample 阶段 (如，前300个K线)**:

- 估计协整参数（α, β）
- 使用线性回归：ln(P2) = α + β * ln(P1) + ε
- 计算拟合优度 (R-squared)
- 检验IS期间价差的平稳性

**Out-of-Sample 阶段 (如，后300个K线)**:

- 使用IS估计的参数计算OS期间的价差
- 验证OS价差的平稳性
- 评估参数的样本外稳定性

#### 滚动窗口机制

使用600K线的滑动窗口（300K IS + 300K OS），每次滑动一个窗口长度：

```
Window 1: [0-299] IS + [300-599] OS
Window 2: [600-899] IS + [900-1199] OS
...
```

### 核心指标与评估标准

#### 1. 协整检验指标

**IS协整率**: IS期间通过协整检验的窗口比例

- 阈值: p-value < 0.1

**OS协整率**: OS期间通过协整检验的窗口比例

- 阈值: p-value < 0.2（更宽松，考虑样本外效应）

**双重协整率**: 同时通过IS和OS协整检验的窗口比例

#### 2. 参数稳定性指标

**Beta系数变异系数 (CV)**:

```
CV = std(β) / mean(β)
```

衡量协整系数的稳定性，CV越小表示关系越稳定。

**R-squared**: 线性关系的拟合优度

- 反映两个资产价格的相关程度
- 高R²表示强线性关系

#### 3. 均值回归特性

**半衰期 (Half-Life)**: 价差回归到均值一半距离所需时间

- 使用ArbitrageLab库计算
- 半衰期越短，均值回归越快，交易机会越频繁

**半衰期稳定性**:

```
Stability = |IS_HalfLife - OS_HalfLife| / IS_HalfLife
```

### 综合评分模型

结合多个维度构建综合评分：

```
Overall_Score = IS_Rate * 0.3 + OS_Rate * 0.5 + Stability_Score * 0.2
```

其中:

- `Stability_Score = 1 / (1 + Beta_CV)`
- OS协整率权重最高（0.5），体现样本外表现的重要性
- IS协整率权重次之（0.3），保证基础协整关系
- 参数稳定性权重最低（0.2），作为补充指标

## 功能模块详解

### 1. DataLoader 类

**功能**: 负责加载和预处理交易对数据

**主要方法**:

- `__init__(data_dir)`: 初始化数据目录，自动扫描可用交易对
- `load_pair_data(coin1, coin2, start_date, end_date, silent)`: 加载两个币种的对齐数据

**数据处理特性**:

- 自动时区对齐和时间排序
- 内连接确保数据时间戳一致性
- 支持日期范围筛选
- 返回包含价格和成交量的清洁DataFrame

**输入格式要求**:

```
数据文件格式: {COIN}USDT_5m.csv
必需列: open_time, close, volume
```

### 2. RollingCointegrationAnalyzer 类

**功能**: 执行滚动窗口协整分析的核心分析引擎

#### 初始化参数

- `window_size`: 单个分析窗口大小（默认300，实际使用600做IS+OS）
- `step_size`: 滚动步长（默认等于window_size）

#### 核心分析方法

**`analyze_pair_rolling(data, coin1, coin2, silent)`**

- 执行完整的滚动协整分析
- 返回详细结果和统计摘要
- 可选择静默模式（批量分析时使用）

**分析流程**:

1. 提取对数价格序列
2. 滚动窗口分割（IS + OS）
3. IS期间参数估计
4. OS期间参数验证
5. 统计指标计算
6. 可视化图表生成（非静默模式）

#### 内部方法详解

**`_estimate_cointegration(log_p1, log_p2)`**

- 使用线性回归估计协整参数
- 计算Alpha(截距)和Beta(斜率)系数
- 执行ADF平稳性检验
- 计算R-squared和半衰期

**`_validate_cointegration(log_p1, log_p2, alpha, beta)`**

- 使用IS估计参数计算OS价差
- 验证OS期间价差平稳性
- 计算OS期间半衰期

**`_calculate_summary(results_df, coin1, coin2, silent)`**

- 汇总所有滚动窗口结果
- 计算关键统计指标
- 生成综合评分

**`_plot_results(results_df, coin1, coin2)`**

- 生成四宫格可视化图表
- 展示p-value时间序列、R-squared、Beta稳定性等

### 3. 批量分析功能

#### 多线程分析架构

**`analyze_single_pair(args)`**

- 单配对分析的包装函数
- 支持多线程并行执行
- 错误处理和结果标准化
- **参数更新**: 现在接收start_date参数用于数据筛选

**`batch_cointegration_analysis(data_loader, output_dir, start_date, max_workers, window_size)`**

- 批量分析所有可能的交易对组合
- 使用ThreadPoolExecutor实现并行处理
- 实时进度显示和结果收集
- **参数更新**: 新增start_date参数，默认为"2023-01-01"

#### 结果处理与输出

**输出内容**:

- 成功分析结果的汇总表格
- 失败分析的详细错误记录
- TOP 10最佳配对展示
- 带时间戳的CSV文件保存

**排序标准**: 按综合评分（overall_score）降序排列

## 输出指标详解

### 主要评估指标

| 指标名称                    | 含义           | 计算方法                   |
| --------------------------- | -------------- | -------------------------- |
| `is_cointegration_rate`   | IS协整通过率   | IS p-value < 0.1的窗口比例 |
| `os_cointegration_rate`   | OS协整通过率   | OS p-value < 0.2的窗口比例 |
| `both_cointegration_rate` | 双重协整通过率 | IS和OS同时通过的窗口比例   |
| `beta_mean`               | Beta系数均值   | 所有窗口Beta的平均值       |
| `beta_cv`                 | Beta变异系数   | Beta标准差/Beta均值        |
| `avg_r_squared`           | 平均拟合优度   | 所有窗口R²的平均值        |
| `avg_is_half_life`        | IS平均半衰期   | IS期间半衰期的平均值       |
| `avg_os_half_life`        | OS平均半衰期   | OS期间半衰期的平均值       |
| `half_life_stability`     | 半衰期稳定性   | IS和OS半衰期的相对差异     |
| `overall_score`           | 综合评分       | 加权综合评价指标           |

### 输出文件结构

**成功分析结果** (`cointegration_analysis_summary_YYYYMMDD_HHMMSS.csv`):

- 包含所有成功分析配对的完整统计指标
- 按综合评分降序排列
- 直接用于配对筛选和策略开发

**失败记录** (`failed_pairs_YYYYMMDD_HHMMSS.csv`):

- 记录分析失败的配对及原因
- 便于问题诊断和数据质量评估

## 使用指南

### 环境准备

#### 依赖安装

确保安装以下Python包：

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels tqdm
pip install arbitragelab  # 用于半衰期计算
```

#### 数据准备

**数据格式要求**：

- 文件命名：`{SYMBOL}USDT_5m.csv`
- 必需列：`open_time`, `close`, `volume`
- 时间格式：支持多种时间戳格式（自动解析）

**示例数据结构**：

```
BTCUSDT_5m.csv:
open_time,open,high,low,close,volume
2023-01-01 00:00:00,42000.1,42100.5,41900.0,42050.2,123.45
...
```

### 基础使用

#### 1. 命令行界面 (推荐)

该工具支持完整的命令行界面，使用方便且功能强大：

**查看帮助信息**：

```bash
python pair_selection_analysis.py --help
```

**单配对分析**：

```bash
# 基础单配对分析
python pair_selection_analysis.py single_coint --coin_pair BTC-ETH

# 自定义参数的单配对分析
python pair_selection_analysis.py single_coint \
    --coin_pair BTC-ETH \
    --start_date 2023-01-01 \
    --window_size 600 \
    --step_size 300 \
    --is_pvalue_threshold 0.05 \
    --os_pvalue_threshold 0.15 \
    --data_dir /path/to/your/data \
    --output_dir /path/to/output
```

**批量配对分析**：

```bash
# 基础批量分析
python pair_selection_analysis.py batch_coint

# 自定义参数的批量分析
python pair_selection_analysis.py batch_coint \
    --start_date 2023-01-01 \
    --max_workers 8 \
    --window_size 600 \
    --step_size 150 \
    --is_pvalue_threshold 0.05 \
    --os_pvalue_threshold 0.1 \
    --data_dir /path/to/your/data \
    --output_dir /path/to/output
# 高频分析示例（更小步长，更密集分析）
python pair_selection_analysis.py single_coint \
    --coin_pair BTC-ETH \
    --window_size 300 \
    --step_size 100 \
    --is_pvalue_threshold 0.01 \
    --os_pvalue_threshold 0.05

# 宽松阈值批量分析（发现更多潜在配对）
python pair_selection_analysis.py batch_coint \
    --is_pvalue_threshold 0.2 \
    --os_pvalue_threshold 0.3 \
    --max_workers 12
```

**命令行参数说明**：

- `mode`: 必选，选择 `single_coint` 或 `batch_coint`
- `--coin_pair`: 币种对格式 COIN1-COIN2 (如 BTC-ETH)，单配对模式必需
- `--start_date`: 数据起始日期，默认 2023-01-01
- `--max_workers`: 批量分析并行线程数，默认 8
- `--window_size`: 分析窗口大小，默认 600
- `--step_size`: 滚动步长，默认等于window_size
- `--is_pvalue_threshold`: IS阶段协整检验p-value阈值，默认 0.1
- `--os_pvalue_threshold`: OS阶段协整检验p-value阈值，默认 0.2
- `--data_dir`: 数据目录路径
- `--output_dir`: 输出目录路径

#### 2. Python API 调用

**单配对分析**：

```python
from pathlib import Path
from pair_selection_analysis import DataLoader, RollingCointegrationAnalyzer, single_cointegration_analysis

# 初始化数据加载器
data_loader = DataLoader(data_dir="/path/to/your/data")
output_dir = Path("/path/to/output")

# 执行单配对分析
result = single_cointegration_analysis(
    data_loader=data_loader,
    coin_pair="BTC-ETH",
    output_dir=output_dir,
    start_date="2023-01-01",
    window_size=600
)
```

**批量配对分析**：

```python
from pair_selection_analysis import DataLoader, batch_cointegration_analysis

# 设置参数
data_loader = DataLoader(data_dir="/path/to/your/data")
output_dir = Path("/path/to/output")

# 执行批量分析
summary_results, failed_results = batch_cointegration_analysis(
    data_loader=data_loader,
    output_dir=output_dir,
    start_date="2023-01-01",  # 新增参数：数据起始日期
    max_workers=8,           # 并行线程数
    window_size=600          # 每个窗口大小(实际使用一半做IS，一半做OS)
)

# 查看Top 10结果
print(summary_results.head(10))
```

**传统API调用**：

```python
from pathlib import Path
from pair_selection_analysis import DataLoader, RollingCointegrationAnalyzer

# 初始化数据加载器
data_loader = DataLoader(data_dir="/path/to/your/data")

# 加载配对数据
data = data_loader.load_pair_data('BTC', 'ETH', start_date='2023-01-01')

# 执行协整分析（使用默认参数）
analyzer = RollingCointegrationAnalyzer(window_size=300)
result = analyzer.analyze_pair_rolling(data, 'BTC', 'ETH')

# 或者自定义所有参数
analyzer_custom = RollingCointegrationAnalyzer(
    window_size=400,
    step_size=200,
    is_pvalue_threshold=0.05,
    os_pvalue_threshold=0.15
)
result_custom = analyzer_custom.analyze_pair_rolling(data, 'BTC', 'ETH')

# 查看结果
print(result['summary'])
# 结果包含可视化图表
```

### 参数详细说明

#### 1. 新增分析参数

**`step_size` (滚动步长)**:
- 控制滑动窗口的步进大小
- 默认值: None (等于window_size，即非重叠窗口)
- 较小的步长提供更密集的分析，但计算量更大
- 示例: `--step_size 150` 表示每次滑动150个数据点

**`is_pvalue_threshold` (IS阶段p-value阈值)**:
- IS (In-Sample) 期间协整检验的显著性阈值
- 默认值: 0.1 (90%置信度)
- 较小的值意味着更严格的协整要求
- 建议范围: 0.01-0.2

**`os_pvalue_threshold` (OS阶段p-value阈值)**:
- OS (Out-of-Sample) 期间协整检验的显著性阈值  
- 默认值: 0.2 (80%置信度，考虑样本外衰减效应)
- 通常比IS阈值更宽松，因为样本外性能普遍较差
- 建议范围: 0.05-0.3

#### 2. 参数使用建议

**保守策略 (高质量配对)**:
```bash
python pair_selection_analysis.py batch_coint \
    --is_pvalue_threshold 0.01 \
    --os_pvalue_threshold 0.05
```

**探索性策略 (发现更多候选)**:
```bash
python pair_selection_analysis.py batch_coint \
    --is_pvalue_threshold 0.15 \
    --os_pvalue_threshold 0.25
```

**高频分析策略 (密集采样)**:
```bash
python pair_selection_analysis.py single_coint \
    --coin_pair BTC-ETH \
    --window_size 300 \
    --step_size 50
```

### 高级配置

#### 1. 参数调优

**窗口大小与阈值组合**：

```python
# 短周期高精度分析
analyzer = RollingCointegrationAnalyzer(
    window_size=200,
    step_size=100,
    is_pvalue_threshold=0.01,
    os_pvalue_threshold=0.05
)

# 长周期稳健分析
analyzer = RollingCointegrationAnalyzer(
    window_size=800,
    step_size=400,
    is_pvalue_threshold=0.05,
    os_pvalue_threshold=0.1
)

# 探索性宽松分析
analyzer = RollingCointegrationAnalyzer(
    window_size=400,
    step_size=200,
    is_pvalue_threshold=0.2,
    os_pvalue_threshold=0.3
)
```

#### 2. 数据筛选

**时间范围限制**：

```python
# 分析特定时间段
summary_results, failed_results = batch_cointegration_analysis(
    data_loader=data_loader,
    output_dir=output_dir,
    start_date="2023-01-01",  # 可以调整起始日期
    max_workers=8,
    window_size=600
)

# 或在单配对分析中指定时间范围
data = data_loader.load_pair_data(
    'BTC', 'ETH', 
    start_date='2023-01-01', 
    end_date='2024-06-30'
)
```

**币种筛选**：

```python
# 手动指定分析的币种列表
target_coins = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
# 可以在batch_cointegration_analysis函数中添加筛选逻辑
```

### 结果解读

#### 关键指标解读

**高质量配对特征**：

- `both_cointegration_rate > 0.5`: 双重协整通过率超过50%
- `beta_cv < 0.1`: Beta系数变异系数低于10%
- `avg_r_squared > 0.7`: 平均拟合优度超过70%
- `overall_score > 0.6`: 综合评分超过0.6

**示例优质配对**：

```
pair: BTC-ETH
both_cointegration_rate: 0.73
avg_r_squared: 0.85
beta_cv: 0.05
overall_score: 0.75
```

#### 可视化图表解读

生成的四宫格图表包含：

1. **P-value时间序列图**：

   - 蓝线：IS期间p-value
   - 红线：OS期间p-value
   - 虚线：显著性阈值
   - 低于阈值表示协整关系显著
2. **R-squared时间序列图**：

   - 反映线性关系强度的稳定性
   - 越接近1表示关系越强
3. **Beta系数稳定性图**：

   - 显示协整系数的时间变化
   - 波动越小表示关系越稳定
4. **IS vs OS散点图**：

   - 横轴：IS期间协整强度
   - 纵轴：OS期间协整强度
   - 颜色：R-squared值
   - 右上角的点表示双重协整性好

## 实际应用场景

### 1. 策略开发流程

```python
# 第一步：批量筛选
summary_results, _ = batch_cointegration_analysis(
    data_loader, output_dir, start_date="2023-01-01", max_workers=8, window_size=600
)

# 第二步：按评分排序选择前20个配对
top_pairs = summary_results.head(20)

# 第三步：对Top配对做详细单独分析
for _, row in top_pairs.iterrows():
    coin1, coin2 = row['pair'].split('-')
    data = data_loader.load_pair_data(coin1, coin2)
    detailed_result = analyzer.analyze_pair_rolling(data, coin1, coin2)
    # 保存详细分析结果用于策略参数设定
```

### 2. 定期监控

```python
import schedule
import datetime

def daily_analysis():
    """每日定时分析任务"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d')
    output_dir = Path(f'/data/analysis/{timestamp}')
  
    summary_results, failed_results = batch_cointegration_analysis(
        data_loader, output_dir, start_date="2023-01-01", max_workers=8, window_size=600
    )
  
    # 发送结果报告
    send_analysis_report(summary_results)

# 每日凌晨2点执行
schedule.every().day.at("02:00").do(daily_analysis)
```

### 3. 回测验证

```python
def backtest_pairs(pair_list, start_date, end_date):
    """对筛选出的配对进行回测验证"""
    backtest_results = []
  
    for pair in pair_list:
        coin1, coin2 = pair.split('-')
        data = data_loader.load_pair_data(coin1, coin2, start_date, end_date)
    
        # 执行协整分析
        result = analyzer.analyze_pair_rolling(data, coin1, coin2, silent=True)
    
        # 模拟交易策略
        strategy_result = simulate_pairs_trading(data, result)
        backtest_results.append(strategy_result)
  
    return backtest_results
```

## 性能特征与优化

### 计算复杂度

- 时间复杂度: O(n²) 其中n为币种数量
- 空间复杂度: O(m) 其中m为单个配对的数据量

### 并行化效果

- 支持多线程并行处理
- 理论加速比接近线程数
- 适合多核CPU环境

### 内存管理

- 流式数据处理，避免内存溢出
- 及时释放临时变量
- 适合大规模配对分析

### 性能优化建议

#### 硬件配置

**推荐配置**：

- CPU：多核处理器（8核或以上）
- 内存：16GB以上
- 存储：SSD硬盘（提高数据加载速度）

#### 参数调优

**平衡分析质量与速度**：

```python
# 快速筛选（初选）
batch_cointegration_analysis(..., start_date="2023-01-01", window_size=300, max_workers=12)

# 精细分析（复选）
batch_cointegration_analysis(..., start_date="2023-01-01", window_size=600, max_workers=8)
```

#### 分批处理

```python
def batch_analysis_by_chunks(data_loader, chunk_size=50):
    """分批处理大量配对"""
    all_coins = [pair.replace('USDT', '') for pair in data_loader.pairs]
  
    for i in range(0, len(all_coins), chunk_size):
        chunk_coins = all_coins[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}: {len(chunk_coins)} coins")
    
        # 处理当前批次
        # ... 分析代码 ...
```

## 故障排除

### 常见错误

**数据文件不存在**：

```
Files not found for BTCUSDT or ETHUSDT
```

解决：检查数据文件路径和命名格式

**内存不足**：

```
MemoryError during analysis
```

解决：减少max_workers或增加系统内存

**协整检验失败**：

```
ADF test failed
```

解决：检查数据质量，确保有足够的数据点

### 数据质量检查

```python
def validate_data_quality(data_loader):
    """验证数据质量"""
    for pair in data_loader.pairs:
        file_path = data_loader.data_dir / f"{pair}_5m.csv"
        df = pd.read_csv(file_path)
    
        print(f"{pair}: {len(df)} records, "
              f"null values: {df.isnull().sum().sum()}, "
              f"date range: {df['open_time'].min()} to {df['open_time'].max()}")
```

## 统计显著性标准

### 协整检验显著性水平

- IS期间: p-value < 0.1 (90%置信度)
- OS期间: p-value < 0.2 (80%置信度，考虑样本外衰减)

### 最低数据要求

- 每个配对至少需要1200个5分钟K线数据点
- 确保有足够的样本进行IS/OS分割

### 稳健性检验

通过滚动窗口分析，评估协整关系在不同时间段的稳定性，避免过拟合问题。

## 理论优势与最佳实践

### 理论优势

1. **样本外验证**: 严格的IS/OS框架避免数据挖掘偏差
2. **动态分析**: 滚动窗口捕捉关系的时变特性
3. **多维评估**: 综合考虑协整性、稳定性和均值回归特性
4. **统计严谨性**: 基于经典计量经济学理论和检验方法
5. **实用性**: 直接服务于统计套利策略的配对选择

### 最佳实践

1. **数据准备阶段**：确保数据完整性和时间对齐
2. **参数选择**：从较小窗口开始测试，逐步调优
3. **结果验证**：对Top配对进行人工复核和回测验证
4. **定期更新**：建立定期分析机制，跟踪配对关系变化
5. **风险控制**：结合其他风险指标，不仅依赖协整分析

该分析框架为统计套利策略提供了科学、严谨的配对筛选工具，结合了严谨的统计方法和高效的工程实现，是进行大规模配对分析的理想工具，有助于提高交易策略的稳健性和盈利能力。

# Statistical Arbitrage Controller (StatArb) 统计套利控制器文档

## 概述 (Overview)

本文档详细解释了 `controllers/generic/stat_arb.py` 中实现的统计套利控制器的完整工作流程。这是一个基于Strategy V2框架的现代化配对交易（Pairs Trading）控制器，通过分析两个资产之间的协整关系来进行自动化统计套利交易。

## 核心概念 (Core Concepts)

### 统计套利 (Statistical Arbitrage)

- **定义**: 一种市场中性策略，通过同时做多一个资产和做空另一个相关资产来获利
- **理论基础**: 利用两个资产价格的长期统计关系，当价差偏离历史均值时进行交易
- **风险特点**: 相对低风险，市场中性，依赖统计模型而非市场方向判断

### 关键统计指标

- **Z-Score**: 标准化价差，用于判断当前价差偏离历史均值的程度
- **Alpha (α)**: 线性回归的截距，表示对冲资产的独立价格水平
- **Beta (β)**: 线性回归的斜率，表示两个资产的相关性强度
- **Spread**: 实际价差与理论价差的百分比差异

## 配置架构 (Configuration Architecture)

### StatArbConfig 类配置参数详解

**文件位置**: `controllers/generic/stat_arb.py:16-85`

#### 基础配置

```python
# 控制器标识
controller_type: str = "generic"        # 控制器类型
controller_name: str = "stat_arb"       # 控制器名称

# 交易对配置
connector_pair_dominant: ConnectorPair = ConnectorPair(
    connector_name="binance_perpetual", 
    trading_pair="SOL-USDT"
)  # 主导资产交易对
connector_pair_hedge: ConnectorPair = ConnectorPair(
    connector_name="binance_perpetual", 
    trading_pair="POPCAT-USDT"
)  # 对冲资产交易对

# 数据源配置
interval: str = "1m"                    # K线时间间隔
candles_config: List[CandlesConfig]     # K线数据源配置（自动生成）
```

#### 统计分析参数

```python
lookback_period: int = 300              # 回看周期，用于统计计算
entry_threshold: Decimal = 2.0          # Z-score入场阈值
```

#### 风险控制参数

```python
take_profit: Decimal = 0.0008           # 单笔止盈 (0.08%)
tp_global: Decimal = 0.01               # 全局止盈 (1%)
sl_global: Decimal = 0.05               # 全局止损 (5%)
```

#### 交易执行参数

```python
min_amount_quote: Decimal = 10          # 最小交易金额（USDT）
quoter_spread: Decimal = 0.0001         # 报价价差 (0.01%)
quoter_cooldown: int = 30               # 冷却时间（秒）
quoter_refresh: int = 10                # 订单刷新时间（秒）
```

#### 仓位管理参数

```python
max_orders_placed_per_side: int = 2     # 每边最大挂单数
max_orders_filled_per_side: int = 2     # 每边最大持仓数
max_position_deviation: Decimal = 0.1   # 最大仓位偏差 (10%)
pos_hedge_ratio: Decimal = 1.0          # 对冲比例
```

#### 杠杆与仓位模式

```python
leverage: int = 20                      # 杠杆倍数
position_mode: PositionMode = HEDGE     # 仓位模式（允许对冲）
```

## 控制器架构与初始化 (Controller Architecture & Initialization)

### StatArb 控制器类

**文件位置**: `controllers/generic/stat_arb.py:87-153`

#### 初始化过程 (__init__)

**文件位置**: `controllers/generic/stat_arb.py:98-153`

1. **理论仓位计算**:

```python
# 根据对冲比例分配总资金
theoretical_dominant_quote = total_amount * (1 / (1 + pos_hedge_ratio))
theoretical_hedge_quote = total_amount * (pos_hedge_ratio / (1 + pos_hedge_ratio))
```

2. **处理数据字典初始化**:

```python
processed_data = {
    "dominant_price": None,              # 主导资产当前价格
    "hedge_price": None,                 # 对冲资产当前价格
    "spread": None,                      # 当前价差
    "z_score": None,                     # 当前Z分数
    "hedge_ratio": None,                 # 对冲比率
    "position_dominant": Decimal("0"),   # 主导资产仓位
    "position_hedge": Decimal("0"),      # 对冲资产仓位
    "active_orders_dominant": [],        # 主导资产活跃订单
    "active_orders_hedge": [],           # 对冲资产活跃订单
    "pair_pnl": Decimal("0"),           # 配对总盈亏
    "signal": 0                         # 交易信号
}
```

3. **K线数据源自动配置**:

```python
# 为两个交易对自动生成K线配置
max_records = lookback_period + 20  # 额外20个数据点确保安全
candles_config = [
    CandlesConfig(
        connector=dominant.connector_name,
        trading_pair=dominant.trading_pair,
        interval=interval,
        max_records=max_records
    ),
    CandlesConfig(
        connector=hedge.connector_name, 
        trading_pair=hedge.trading_pair,
        interval=interval,
        max_records=max_records
    )
]
```

4. **永续合约配置**:

```python
# 仅在永续合约模式下设置杠杆和仓位模式
if "_perpetual" in connector_name:
    connector.set_position_mode(position_mode)
    connector.set_leverage(trading_pair, leverage)
```

## 核心交易逻辑 (Core Trading Logic)

### 主要执行流程 (determine_executor_actions)

**文件位置**: `controllers/generic/stat_arb.py:154-196`

#### 执行优先级决策树

```python
def determine_executor_actions() -> List[ExecutorAction]:
    actions = []
  
    # 优先级1: 全局风险控制
    if pair_pnl_pct > tp_global or pair_pnl_pct < -sl_global:
        # 触发全局止盈止损 -> 平掉所有仓位
        for position in positions_held:
            actions.extend(get_executors_to_reduce_position(position))
        return actions
  
    # 优先级2: 信号驱动交易
    elif signal != 0:
        actions.extend(get_executors_to_quote())  # 开新仓
        actions.extend(get_executors_to_reduce_position_on_opposite_signal())  # 平反向仓
  
    # 优先级3: 仓位维护
    actions.extend(get_executors_to_keep_position())  # 冷却期管理
    actions.extend(get_executors_to_refresh())        # 订单刷新
  
    return actions
```

### 数据更新与信号生成 (update_processed_data)

**文件位置**: `controllers/generic/stat_arb.py:361-463`

#### 统计分析流程

1. **获取价差和Z分数**:

```python
spread, z_score = get_spread_and_z_score()
```

2. **交易信号生成**:

```python
entry_threshold = float(config.entry_threshold)  # 通常为2.0

if z_score > entry_threshold:
    # 价差过高，预期回归：做多主导，做空对冲
    signal = 1
    dominant_side, hedge_side = TradeType.BUY, TradeType.SELL

elif z_score < -entry_threshold:
    # 价差过低，预期回归：做空主导，做多对冲  
    signal = -1
    dominant_side, hedge_side = TradeType.SELL, TradeType.BUY
else:
    # 价差在正常范围，无交易信号
    signal = 0
```

3. **仓位状态计算**:

```python
# 获取当前仓位信息
positions_dominant = [pos for pos in positions_held 
                     if pos.connector_name == dominant.connector_name 
                     and pos.trading_pair == dominant.trading_pair 
                     and pos.side == dominant_side]

positions_hedge = [pos for pos in positions_held 
                  if pos.connector_name == hedge.connector_name 
                  and pos.trading_pair == hedge.trading_pair 
                  and pos.side == hedge_side]

# 计算配对盈亏百分比
pair_pnl_pct = (dominant_pnl + hedge_pnl) / (dominant_position + hedge_position)
```

4. **仓位不平衡检查**:

```python
# 计算仓位不平衡
imbalance_scaled = position_dominant - position_hedge * pos_hedge_ratio
imbalance_scaled_pct = imbalance_scaled / position_dominant

# 风险控制过滤器
if imbalance_scaled_pct > max_position_deviation:
    filter_connector_pair = connector_pair_dominant  # 暂停主导资产交易
elif imbalance_scaled_pct < -max_position_deviation:
    filter_connector_pair = connector_pair_hedge     # 暂停对冲资产交易
```

### 统计分析核心算法 (get_spread_and_z_score)

**文件位置**: `controllers/generic/stat_arb.py:464-562`

#### 算法实现步骤

1. **数据获取与验证**:

```python
# 获取K线数据
dominant_df = market_data_provider.get_candles_df(
    connector_name=dominant.connector_name,
    trading_pair=dominant.trading_pair,
    interval=interval,
    max_records=max_records
)

hedge_df = market_data_provider.get_candles_df(
    connector_name=hedge.connector_name,
    trading_pair=hedge.trading_pair,
    interval=interval,
    max_records=max_records
)

# 数据充足性检查
min_length = min(len(dominant_prices), len(hedge_prices))
if min_length < lookback_period:
    logger.warning(f"数据不足. 需要: {lookback_period}, 可用: {min_length}")
    return None
```

2. **收益率计算与标准化**:

```python
# 提取收盘价
dominant_prices = dominant_df['close'].values[-lookback_period:]
hedge_prices = hedge_df['close'].values[-lookback_period:]

# 计算百分比收益率
dominant_pct_change = np.diff(dominant_prices) / dominant_prices[:-1]
hedge_pct_change = np.diff(hedge_prices) / hedge_prices[:-1]

# 转换为累积收益率
dominant_cum_returns = np.cumprod(dominant_pct_change + 1)
hedge_cum_returns = np.cumprod(hedge_pct_change + 1)

# 标准化（起始值为1）
dominant_cum_returns = dominant_cum_returns / dominant_cum_returns[0]
hedge_cum_returns = hedge_cum_returns / hedge_cum_returns[0]
```

3. **线性回归分析**:

```python
# 执行线性回归
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(
    dominant_cum_returns.reshape(-1, 1), 
    hedge_cum_returns
)

alpha = reg.intercept_    # 截距
beta = reg.coef_[0]      # 斜率（对冲比例）

# 保存回归参数
processed_data.update({
    "alpha": alpha,
    "beta": beta,
})
```

4. **价差与Z分数计算**:

```python
# 计算理论价格和实际价差
y_pred = alpha + beta * dominant_cum_returns
spread_pct = (hedge_cum_returns - y_pred) / y_pred * 100

# 计算Z分数
mean_spread = np.mean(spread_pct)
std_spread = np.std(spread_pct)

if std_spread == 0:
    logger.warning("价差标准差为零，无法计算Z分数")
    return None

current_spread = spread_pct[-1]
current_z_score = (current_spread - mean_spread) / std_spread

return current_spread, current_z_score
```

## 交易执行机制 (Trading Execution Mechanism)

### 新仓位开立 (get_executors_to_quote)

**文件位置**: `controllers/generic/stat_arb.py:268-329`

#### 主导资产执行器创建

```python
# 检查开仓条件
if (dominant_gap > 0 and                                    # 有资金缺口
    filter_connector_pair != connector_pair_dominant and    # 未被风控过滤
    len(executors_dominant_placed) < max_orders_placed and  # 未超订单限制
    len(executors_dominant_filled) < max_orders_filled):    # 未超持仓限制

    # 计算限价单价格
    if trade_type_dominant == TradeType.BUY:
        price = min_price_dominant * (1 - quoter_spread)
    else:
        price = max_price_dominant * (1 + quoter_spread)

    # 创建仓位执行器配置
    dominant_executor_config = PositionExecutorConfig(
        timestamp=current_time,
        connector_name=connector_pair_dominant.connector_name,
        trading_pair=connector_pair_dominant.trading_pair,
        side=trade_type_dominant,
        entry_price=price,
        amount=min_amount_quote / dominant_price,
        triple_barrier_config=triple_barrier_config,
        leverage=leverage,
    )
  
    actions.append(CreateExecutorAction(
        controller_id=config.id, 
        executor_config=dominant_executor_config
    ))
```

#### 对冲资产执行器创建

对冲资产的执行器创建逻辑与主导资产类似，但交易方向相反：

- 当signal=1时：主导资产BUY，对冲资产SELL
- 当signal=-1时：主导资产SELL，对冲资产BUY

### 反向仓位平仓 (get_executors_to_reduce_position_on_opposite_signal)

**文件位置**: `controllers/generic/stat_arb.py:198-228`

#### 信号反转处理

```python
def get_executors_to_reduce_position_on_opposite_signal():
    # 确定要平仓的方向
    if signal == 1:  # 当前信号：做多主导/做空对冲
        dominant_side_to_close = TradeType.SELL  # 平掉主导资产的空头
        hedge_side_to_close = TradeType.BUY      # 平掉对冲资产的多头
    elif signal == -1:  # 当前信号：做空主导/做多对冲
        dominant_side_to_close = TradeType.BUY   # 平掉主导资产的多头
        hedge_side_to_close = TradeType.SELL     # 平掉对冲资产的空头
    else:
        return []  # 无信号时不执行平仓

    # 找到需要停止的执行器
    dominant_executors_to_stop = filter_executors(
        executors_info, 
        filter_func=lambda e: (
            e.connector_name == connector_pair_dominant.connector_name and
            e.trading_pair == connector_pair_dominant.trading_pair and
            e.side == dominant_side_to_close
        )
    )
  
    hedge_executors_to_stop = filter_executors(
        executors_info,
        filter_func=lambda e: (
            e.connector_name == connector_pair_hedge.connector_name and
            e.trading_pair == connector_pair_hedge.trading_pair and
            e.side == hedge_side_to_close
        )
    )

    # 创建停止动作
    stop_actions = [
        StopExecutorAction(
            controller_id=config.id, 
            executor_id=executor.id, 
            keep_position=False
        ) for executor in dominant_executors_to_stop + hedge_executors_to_stop
    ]

    # 创建减仓动作
    reduce_actions = []
    for position in positions_held:
        if (position.connector_name == connector_pair_dominant.connector_name and 
            position.trading_pair == connector_pair_dominant.trading_pair and 
            position.side == dominant_side_to_close):
            reduce_actions.extend(get_executors_to_reduce_position(position))
        elif (position.connector_name == connector_pair_hedge.connector_name and 
              position.trading_pair == connector_pair_hedge.trading_pair and 
              position.side == hedge_side_to_close):
            reduce_actions.extend(get_executors_to_reduce_position(position))

    return stop_actions + reduce_actions
```

### 仓位平仓执行 (get_executors_to_reduce_position)

**文件位置**: `controllers/generic/stat_arb.py:331-359`

```python
def get_executors_to_reduce_position(position: PositionSummary):
    if position.amount > Decimal("0"):
        # 创建市价单平仓配置
        config = OrderExecutorConfig(
            timestamp=current_time,
            connector_name=position.connector_name,
            trading_pair=position.trading_pair,
            side=TradeType.BUY if position.side == TradeType.SELL else TradeType.SELL,
            amount=position.amount,
            position_action=PositionAction.CLOSE,
            execution_strategy=ExecutionStrategy.MARKET,  # 使用市价单快速平仓
            leverage=leverage,
        )
        return [CreateExecutorAction(controller_id=config.id, executor_config=config)]
    return []
```

## 订单生命周期管理 (Order Lifecycle Management)

### 冷却期管理 (get_executors_to_keep_position)

**文件位置**: `controllers/generic/stat_arb.py:230-247`

```python
def get_executors_to_keep_position():
    """
    已成交订单冷却期管理：
    - 订单成交后进入冷却期
    - 冷却期结束后停止执行器但保留仓位
    - 避免频繁交易
    """
    stop_actions = []
  
    # 遍历所有已成交的执行器
    for executor in executors_dominant_filled + executors_hedge_filled:
        if current_time - executor.timestamp >= quoter_cooldown:
            # 冷却期结束，停止执行器但保留仓位
            stop_actions.append(StopExecutorAction(
                controller_id=config.id, 
                executor_id=executor.id, 
                keep_position=True  # 关键：保留仓位
            ))
  
    return stop_actions
```

### 订单刷新机制 (get_executors_to_refresh)

**文件位置**: `controllers/generic/stat_arb.py:249-266`

```python
def get_executors_to_refresh():
    """
    未成交订单刷新机制：
    - 挂单时间超过刷新周期时取消重新下单
    - 确保订单价格跟上市场变化
    - 提高成交率
    """
    refresh_actions = []
  
    # 遍历所有已下单但未成交的执行器
    for executor in executors_dominant_placed + executors_hedge_placed:
        if current_time - executor.timestamp >= quoter_refresh:
            # 刷新期结束，取消订单并重新下单
            refresh_actions.append(StopExecutorAction(
                controller_id=config.id, 
                executor_id=executor.id, 
                keep_position=False  # 关键：不保留仓位，完全重新开始
            ))
  
    return refresh_actions
```

## 辅助功能模块 (Utility Modules)

### 价格获取 (get_pairs_prices)

**文件位置**: `controllers/generic/stat_arb.py:564-579`

```python
def get_pairs_prices():
    """获取两个交易对的当前中间价"""
    dominant_price = market_data_provider.get_price_by_type(
        connector_name=connector_pair_dominant.connector_name,
        trading_pair=connector_pair_dominant.trading_pair, 
        price_type=PriceType.MidPrice
    )

    hedge_price = market_data_provider.get_price_by_type(
        connector_name=connector_pair_hedge.connector_name,
        trading_pair=connector_pair_hedge.trading_pair, 
        price_type=PriceType.MidPrice
    )
  
    return dominant_price, hedge_price
```

### 执行器过滤 (get_executors_dominant/hedge)

**文件位置**: `controllers/generic/stat_arb.py:581-615`

```python
def get_executors_dominant():
    """获取主导资产的活跃执行器"""
    # 已下单但未成交的执行器
    placed = filter_executors(
        executors_info,
        filter_func=lambda e: (
            e.connector_name == connector_pair_dominant.connector_name and
            e.trading_pair == connector_pair_dominant.trading_pair and
            e.is_active and not e.is_trading and
            e.type == "position_executor"
        )
    )
  
    # 已成交的执行器
    filled = filter_executors(
        executors_info,
        filter_func=lambda e: (
            e.connector_name == connector_pair_dominant.connector_name and
            e.trading_pair == connector_pair_dominant.trading_pair and
            e.is_active and e.is_trading and
            e.type == "position_executor"
        )
    )
  
    return placed, filled
```

## 状态监控与展示 (Status Monitoring & Display)

### 状态格式化 (to_format_status)

**文件位置**: `controllers/generic/stat_arb.py:617-654`

```python
def to_format_status() -> List[str]:
    """
    格式化控制器状态展示
    包含：交易对信息、仓位状态、执行器状态、统计指标
    """
    status_lines = []
    status_lines.append(f"""
交易对信息:
主导交易对: {connector_pair_dominant} | 对冲交易对: {connector_pair_hedge}
时间框架: {interval} | 回看周期: {lookback_period} | 入场阈值: {entry_threshold}

仓位目标:
理论主导仓位: {theoretical_dominant_quote} | 理论对冲仓位: {theoretical_hedge_quote} | 对冲比例: {pos_hedge_ratio}
实际主导仓位: {processed_data['position_dominant_quote']:.2f} | 实际对冲仓位: {processed_data['position_hedge_quote']:.2f}
仓位不平衡: {processed_data['imbalance']:.2f} | 不平衡百分比: {processed_data['imbalance_scaled_pct']:.2f}%

当前执行器:
主导挂单数: {len(processed_data['executors_dominant_placed'])} | 对冲挂单数: {len(processed_data['executors_hedge_placed'])}
主导持仓数: {len(processed_data['executors_dominant_filled'])} | 对冲持仓数: {len(processed_data['executors_hedge_filled'])}

统计指标:
交易信号: {processed_data['signal']:.2f} | Z分数: {processed_data['z_score']:.2f} | 价差: {processed_data['spread']:.2f}
Alpha: {processed_data['alpha']:.2f} | Beta: {processed_data['beta']:.2f}
配对盈亏百分比: {processed_data['pair_pnl_pct'] * 100:.2f}%
""")
    return status_lines
```

## 风险管理机制 (Risk Management Mechanisms)

### 全局风险控制

1. **全局止盈止损**:

```python
# 在determine_executor_actions中首先检查
if pair_pnl_pct > tp_global or pair_pnl_pct < -sl_global:
    # 触发全局止盈止损，立即平掉所有仓位
    for position in positions_held:
        actions.extend(get_executors_to_reduce_position(position))
    return actions
```

2. **仓位偏差控制**:

```python
# 监控仓位不平衡，防止单边风险
imbalance_scaled_pct = imbalance_scaled / position_dominant

if imbalance_scaled_pct > max_position_deviation:
    filter_connector_pair = connector_pair_dominant  # 暂停主导资产交易
elif imbalance_scaled_pct < -max_position_deviation:
    filter_connector_pair = connector_pair_hedge     # 暂停对冲资产交易
```

3. **订单数量控制**:

```python
# 限制每边的最大订单数和持仓数
if (len(executors_dominant_placed) < max_orders_placed_per_side and
    len(executors_dominant_filled) < max_orders_filled_per_side):
    # 允许新开仓
```

### 个别仓位风险控制

通过 `triple_barrier_config` 实现个别仓位的止盈：

```python
@property
def triple_barrier_config(self) -> TripleBarrierConfig:
    return TripleBarrierConfig(
        take_profit=self.take_profit,           # 单笔止盈
        open_order_type=OrderType.LIMIT_MAKER,  # 开仓使用限价单
        take_profit_order_type=OrderType.LIMIT_MAKER,  # 止盈使用限价单
    )
```

## 参数调优与优化建议 (Parameter Tuning & Optimization)

### 关键参数分析

#### 1. lookback_period (回看周期)

- **当前默认值**: 300
- **建议调整范围**: 200-500
- **影响**: 统计关系的稳定性与响应速度
- **调优策略**:
  * 高波动市场：200-250（快速响应）
  * 稳定市场：400-500（稳定信号）
  * 与时间框架配合：1m用300，5m用100-150

#### 2. entry_threshold (入场阈值)

- **当前默认值**: 2.0
- **建议调整范围**: 1.5-3.0
- **影响**: 交易频率与信号质量
- **调优策略**:
  * 信号过少：降低到1.8-1.5
  * 信号过多：提高到2.5-3.0
  * 高相关性资产：可用较低值
  * 低相关性资产：用较高值

#### 3. 风险控制参数优化

```python
# 保守型配置
take_profit = Decimal("0.0006")     # 0.06%
tp_global = Decimal("0.005")        # 0.5%
sl_global = Decimal("0.02")         # 2%

# 激进型配置  
take_profit = Decimal("0.001")      # 0.1%
tp_global = Decimal("0.015")        # 1.5%
sl_global = Decimal("0.05")         # 5%
```

#### 4. 执行参数优化

```python
# 高频交易配置
quoter_cooldown = 15               # 15秒冷却
quoter_refresh = 5                 # 5秒刷新
quoter_spread = Decimal("0.0001")  # 0.01%价差

# 稳健交易配置
quoter_cooldown = 60               # 60秒冷却
quoter_refresh = 20                # 20秒刷新
quoter_spread = Decimal("0.0002")  # 0.02%价差
```

### 资产对选择标准

1. **协整性测试**: 使用ADF测试验证长期协整关系
2. **相关性分析**: 相关系数建议在0.3-0.8之间
3. **流动性要求**: 确保24小时交易量充足
4. **价格稳定性**: 避免极端波动的资产对
5. **基本面关联**: 选择有内在关联的资产

### 监控要点

1. **统计指标监控**:

   - Z分数分布是否正态
   - Beta值是否稳定
   - 信号频率是否合理
2. **交易表现监控**:

   - 日均交易次数：5-20次
   - 胜率目标：55%-70%
   - 夏普比率：>1.0
3. **风险指标监控**:

   - 最大回撤：<10%
   - 仓位不平衡频率
   - 单日最大损失

## 常见问题与解决方案 (Troubleshooting)

### 1. Z-Score计算异常

**问题症状**: z_score返回0或异常值

**可能原因**:

- 历史数据不足
- 价格序列标准差为0
- 数据获取失败

**解决方案**:

```python
# 在get_spread_and_z_score中添加检查
if min_length < lookback_period:
    logger.warning(f"数据不足: 需要{lookback_period}, 可用{min_length}")
    return None

if std_spread == 0:
    logger.warning("价差标准差为零，无法计算Z分数")
    return None
```

### 2. 仓位不平衡过度

**问题症状**: 一边仓位过大，频繁触发风控

**解决方案**:

- 调整 `max_position_deviation` 参数
- 检查两个资产的流动性匹配
- 优化 `pos_hedge_ratio` 设置
- 确认订单执行延迟问题

### 3. 交易频率异常

**信号过少**:

```python
# 降低入场阈值
entry_threshold = Decimal("1.8")  # 从2.0降低

# 缩短回看周期
lookback_period = 200  # 从300缩短
```

**信号过多**:

```python
# 提高入场阈值
entry_threshold = Decimal("2.5")  # 从2.0提高

# 延长冷却时间
quoter_cooldown = 60  # 从30秒延长
```

### 4. API调用失败

**问题**: 杠杆设置或仓位模式设置失败

**解决方案**: 在回测模式下禁用API调用

```python
# 检查是否为实盘模式
if "_perpetual" in connector_name and not is_backtesting:
    connector.set_leverage(trading_pair, leverage)
    connector.set_position_mode(position_mode)
```

## 扩展与改进方向 (Extensions & Improvements)

### 1. 动态参数调整

- 基于实时波动性调整 `entry_threshold`
- 根据相关性变化调整 `pos_hedge_ratio`
- 自适应 `lookback_period` 优化

### 2. 高级统计分析

- 加入卡尔曼滤波优化Beta估计
- 使用GARCH模型预测波动性
- 多因子模型增强信号质量

### 3. 风险管理增强

- 实时VaR计算
- 压力测试模拟
- 相关性破裂检测

### 4. 机器学习集成

- 使用ML预测价差回归概率
- 深度学习优化参数组合
- 强化学习动态调整策略

## 部署与使用指南 (Deployment & Usage Guide)

### 基础配置示例

```python
# 创建StatArb配置
config = StatArbConfig(
    connector_pair_dominant=ConnectorPair(
        connector_name="binance_perpetual",
        trading_pair="SOL-USDT"
    ),
    connector_pair_hedge=ConnectorPair(
        connector_name="binance_perpetual", 
        trading_pair="POPCAT-USDT"
    ),
    total_amount_quote=Decimal("1000"),    # 总资金1000 USDT
    interval="1m",                         # 1分钟K线
    lookback_period=300,                   # 300个周期回看
    entry_threshold=Decimal("2.0"),        # Z分数阈值2.0
    take_profit=Decimal("0.0008"),         # 0.08%止盈
    tp_global=Decimal("0.01"),             # 1%全局止盈
    sl_global=Decimal("0.05"),             # 5%全局止损
    leverage=10,                           # 10倍杠杆
)

# 创建控制器实例
controller = StatArb(config)
```

### Strategy V2框架集成

```python
from hummingbot.strategy_v2.strategy_v2_controller import StrategyV2Controller

class StatArbStrategy(StrategyV2Controller):
    def __init__(self):
        # 创建StatArb控制器
        stat_arb_config = StatArbConfig(
            # ... 配置参数
        )
      
        # 添加到控制器列表
        self.controllers = [StatArb(stat_arb_config)]
      
        super().__init__()
```

---

**注意**:

1. 本控制器基于Strategy V2框架构建，继承了现代化的执行器管理和风险控制机制
2. 统计套利依赖历史价格关系，不保证未来收益
3. 建议从小资金开始测试，逐步优化参数配置
4. 定期监控协整关系的稳定性，及时调整资产对选择
5. 严格遵守风险管理原则，不投入超过承受能力的资金

本文档基于 `controllers/generic/stat_arb.py` 的详细代码分析，提供了完整的实现原理和使用指导。在实际应用中，请根据具体市场环境和风险偏好调整相关参数。

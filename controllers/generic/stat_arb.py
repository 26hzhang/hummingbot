from decimal import Decimal
from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression

from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair, PositionSummary
from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy, OrderExecutorConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction


class StatArbConfig(ControllerConfigBase):
    """
    Configuration for a statistical arbitrage controller that trades two cointegrated assets.
    统计套利控制器的配置类，用于交易两个协整资产。

    This class defines all parameters needed for statistical arbitrage strategy that identifies
    and trades on price discrepancies between two correlated financial instruments.
    该类定义了统计套利策略所需的所有参数，用于识别和交易两个相关金融工具之间的价格差异。
    """
    controller_type: str = "generic"  # Type of controller / 控制器类型
    controller_name: str = "stat_arb"  # Name identifier for this controller / 此控制器的名称标识符
    candles_config: List[CandlesConfig] = []  # Configuration for candle data sources / K线数据源配置
    connector_pair_dominant: ConnectorPair = ConnectorPair(connector_name="binance_perpetual", trading_pair="SOL-USDT")  # Primary trading pair for the strategy / 策略的主要交易对
    connector_pair_hedge: ConnectorPair = ConnectorPair(connector_name="binance_perpetual", trading_pair="POPCAT-USDT")  # Hedge trading pair used to offset dominant pair risk / 用于对冲主要交易对风险的对冲交易对
    interval: str = "1m"  # Time interval for candle data (e.g., '1m', '5m', '1h') / K线数据的时间间隔（例如'1m'、'5m'、'1h'）
    lookback_period: int = 300  # Number of historical periods to analyze for statistical calculations / 用于统计计算的历史时期数量
    entry_threshold: Decimal = Decimal("2.0")  # Z-score threshold for triggering trades (higher = fewer trades) / 触发交易的Z分数阈值（越高=交易越少）
    take_profit: Decimal = Decimal("0.0008")  # Individual position take profit percentage / 单个仓位止盈百分比
    tp_global: Decimal = Decimal("0.01")  # Global take profit threshold for entire pair position / 整个配对仓位的全局止盈阈值
    sl_global: Decimal = Decimal("0.05")  # Global stop loss threshold for entire pair position / 整个配对仓位的全局止损阈值
    min_amount_quote: Decimal = Decimal("10")  # Minimum order amount in quote currency / 以报价货币计的最小订单金额
    quoter_spread: Decimal = Decimal("0.0001")  # Spread applied to quote prices for market making / 应用于报价的价差，用于做市
    quoter_cooldown: int = 30  # Cooldown period in seconds before closing filled positions / 平仓已成交仓位前的冷却期（秒）
    quoter_refresh: int = 10  # Time in seconds before refreshing unfilled orders / 刷新未成交订单前的时间（秒）
    max_orders_placed_per_side: int = 2  # Maximum number of pending orders per side / 每边的最大挂单数量
    max_orders_filled_per_side: int = 2  # Maximum number of filled orders per side before stopping / 每边停止前的最大成交订单数量
    max_position_deviation: Decimal = Decimal("0.1")  # Maximum allowed deviation between dominant and hedge positions / 主要仓位和对冲仓位之间允许的最大偏差
    pos_hedge_ratio: Decimal = Decimal("1.0")  # Ratio for hedging positions (hedge_amount = dominant_amount * ratio) / 对冲仓位比率（对冲金额=主要金额*比率）
    leverage: int = 20  # Leverage multiplier for perpetual contracts / 永续合约的杠杆倍数
    position_mode: PositionMode = PositionMode.HEDGE  # Position mode: HEDGE allows long/short simultaneously / 仓位模式：HEDGE允许同时做多/做空

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        """
        Create triple barrier configuration for position management.
        创建用于仓位管理的三重障碍配置。

        Returns:
            TripleBarrierConfig: Configuration with take profit and order types
            返回值：包含止盈和订单类型的配置
        """
        return TripleBarrierConfig(
            take_profit=self.take_profit,
            open_order_type=OrderType.LIMIT_MAKER,
            take_profit_order_type=OrderType.LIMIT_MAKER,
        )

    def update_markets(self, markets: dict) -> dict:
        """
        Update markets dictionary with both trading pairs.
        使用两个交易对更新市场字典。

        Args:
            markets (dict): Dictionary to store market configurations (存储市场配置的字典)

        Returns:
            dict: Updated markets dictionary with both pairs added (添加了两个交易对的更新市场字典)
        """
        # Add dominant pair / 添加主要交易对
        if self.connector_pair_dominant.connector_name not in markets:
            markets[self.connector_pair_dominant.connector_name] = set()
        markets[self.connector_pair_dominant.connector_name].add(self.connector_pair_dominant.trading_pair)

        # Add hedge pair / 添加对冲交易对
        if self.connector_pair_hedge.connector_name not in markets:
            markets[self.connector_pair_hedge.connector_name] = set()
        markets[self.connector_pair_hedge.connector_name].add(self.connector_pair_hedge.trading_pair)

        return markets


class StatArb(ControllerBase):
    """
    Statistical arbitrage controller that trades two cointegrated assets.
    统计套利控制器，交易两个协整资产。

    This controller implements a pairs trading strategy that identifies statistical
    relationships between two assets and trades on mean reversion opportunities.
    该控制器实现了一个配对交易策略，识别两个资产之间的统计关系，
    并在均值回归机会中进行交易。
    """

    def __init__(self, config: StatArbConfig, *args, **kwargs):
        """
        Initialize the statistical arbitrage controller.
        初始化统计套利控制器。

        Args:
            config (StatArbConfig): Configuration object containing all strategy parameters
                                   包含所有策略参数的配置对象
        """
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.theoretical_dominant_quote = self.config.total_amount_quote * (1 / (1 + self.config.pos_hedge_ratio))
        self.theoretical_hedge_quote = self.config.total_amount_quote * (self.config.pos_hedge_ratio / (1 + self.config.pos_hedge_ratio))

        # Initialize processed data dictionary / 初始化处理数据字典
        self.processed_data = {
            "dominant_price": None,
            "hedge_price": None,
            "spread": None,
            "z_score": None,
            "hedge_ratio": None,
            "position_dominant": Decimal("0"),
            "position_hedge": Decimal("0"),
            "active_orders_dominant": [],
            "active_orders_hedge": [],
            "pair_pnl": Decimal("0"),
            "signal": 0  # 0: no signal, 1: long dominant/short hedge, -1: short dominant/long hedge (交易信号：0无信号, 1做多主导/做空对冲, -1做空主导/做多对冲)
        }

        # Setup candles config if not already set / 如果尚未设置，则设置K线配置
        if len(self.config.candles_config) == 0:
            max_records = self.config.lookback_period + 20  # extra records for safety / 额外记录以确保安全
            self.max_records = max_records
            self.config.candles_config = [
                CandlesConfig(
                    connector=self.config.connector_pair_dominant.connector_name,
                    trading_pair=self.config.connector_pair_dominant.trading_pair,
                    interval=self.config.interval,
                    max_records=max_records
                ),
                CandlesConfig(
                    connector=self.config.connector_pair_hedge.connector_name,
                    trading_pair=self.config.connector_pair_hedge.trading_pair,
                    interval=self.config.interval,
                    max_records=max_records
                )
            ]
        if "_perpetual" in self.config.connector_pair_dominant.connector_name:
            connector = self.market_data_provider.get_connector(self.config.connector_pair_dominant.connector_name)
            connector.set_position_mode(self.config.position_mode)
            connector.set_leverage(self.config.connector_pair_dominant.trading_pair, self.config.leverage)
        if "_perpetual" in self.config.connector_pair_hedge.connector_name:
            connector = self.market_data_provider.get_connector(self.config.connector_pair_hedge.connector_name)
            connector.set_position_mode(self.config.position_mode)
            connector.set_leverage(self.config.connector_pair_hedge.trading_pair, self.config.leverage)

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        The execution logic for the statistical arbitrage strategy.
        统计套利策略的执行逻辑。

        Market Data Conditions: Signal is generated based on the z-score of the spread between the two assets.
        市场数据条件：信号基于两个资产之间价差的Z分数生成。
                                If signal == 1 --> long dominant/short hedge
                                如果信号==1 --> 做多主要资产/做空对冲资产
                                If signal == -1 --> short dominant/long hedge
                                如果信号==-1 --> 做空主要资产/做多对冲资产

        Execution Conditions: If the signal is generated add position executors to quote from the dominant and hedge markets.
        执行条件：如果生成信号，则添加仓位执行器从主要和对冲市场报价。
            We compare the current position with the theoretical position for the dominant and hedge assets.
            我们比较主要和对冲资产的当前仓位与理论仓位。
            If the current position + the active placed amount is greater than the theoretical position, can't place more orders.
            如果当前仓位+活跃下单金额大于理论仓位，则不能下更多订单。
            If the imbalance scaled pct is greater than the threshold, we avoid placing orders in the market passed on filtered_connector_pair.
            如果不平衡缩放百分比大于阈值，我们避免在过滤连接器对的市场中下单。
            If the pnl of total position is greater than the take profit or lower than the stop loss, we close the position.
            如果总仓位的盈亏大于止盈或低于止损，我们平仓。

        Returns:
            List[ExecutorAction]: List of actions to execute (create/stop executors) (要执行的动作列表（创建/停止执行器）)
        """
        actions: List[ExecutorAction] = []
        # Check global take profit and stop loss / 检查全局止盈和止损
        if self.processed_data["pair_pnl_pct"] > self.config.tp_global or self.processed_data["pair_pnl_pct"] < -self.config.sl_global:
            # Close all positions / 平掉所有仓位
            for position in self.positions_held:
                actions.extend(self.get_executors_to_reduce_position(position))
            return actions
        # Check the signal / 检查信号
        elif self.processed_data["signal"] != 0:
            actions.extend(self.get_executors_to_quote())
            actions.extend(self.get_executors_to_reduce_position_on_opposite_signal())

        # Get the executors to keep position after a cooldown is reached / 在达到冷却时间后获取保持仓位的执行器
        actions.extend(self.get_executors_to_keep_position())
        actions.extend(self.get_executors_to_refresh())

        return actions

    def get_executors_to_reduce_position_on_opposite_signal(self) -> List[ExecutorAction]:
        """
        Get executors to reduce positions that are opposite to the current signal.
        获取执行器来减少与当前信号相反的仓位。

        When signal changes direction, this method identifies and closes positions
        that are no longer aligned with the new trading signal.
        当信号改变方向时，此方法识别并平掉不再与新交易信号一致的仓位。

        Returns:
            List[ExecutorAction]: Actions to stop executors and reduce positions (停止执行器和减少仓位的动作)
        """
        if self.processed_data["signal"] == 1:
            dominant_side, hedge_side = TradeType.SELL, TradeType.BUY
        elif self.processed_data["signal"] == -1:
            dominant_side, hedge_side = TradeType.BUY, TradeType.SELL
        else:
            return []
        # Get executors to stop / 获取要停止的执行器
        dominant_active_executors_to_stop = self.filter_executors(self.executors_info, filter_func=lambda e: e.connector_name == self.config.connector_pair_dominant.connector_name and e.trading_pair == self.config.connector_pair_dominant.trading_pair and e.side == dominant_side)
        hedge_active_executors_to_stop = self.filter_executors(self.executors_info, filter_func=lambda e: e.connector_name == self.config.connector_pair_hedge.connector_name and e.trading_pair == self.config.connector_pair_hedge.trading_pair and e.side == hedge_side)
        stop_actions = [StopExecutorAction(controller_id=self.config.id, executor_id=executor.id, keep_position=False) for executor in dominant_active_executors_to_stop + hedge_active_executors_to_stop]

        # Get order executors to reduce positions / 获取订单执行器来减少仓位
        reduce_actions: List[ExecutorAction] = []
        for position in self.positions_held:
            if position.connector_name == self.config.connector_pair_dominant.connector_name and position.trading_pair == self.config.connector_pair_dominant.trading_pair and position.side == dominant_side:
                reduce_actions.extend(self.get_executors_to_reduce_position(position))
            elif position.connector_name == self.config.connector_pair_hedge.connector_name and position.trading_pair == self.config.connector_pair_hedge.trading_pair and position.side == hedge_side:
                reduce_actions.extend(self.get_executors_to_reduce_position(position))
        return stop_actions + reduce_actions

    def get_executors_to_keep_position(self) -> List[ExecutorAction]:
        """
        Get executors to stop after cooldown period while keeping the position.
        在冷却期后获取要停止的执行器，同时保持仓位。

        After a position is filled and cooldown time has passed, this method
        stops the executor but keeps the position for further management.
        在仓位成交并超过冷却时间后，此方法停止执行器但保持仓位以便进一步管理。

        Returns:
            List[ExecutorAction]: Stop actions for filled executors past cooldown (超过冷却期的已成交执行器的停止动作)
        """
        stop_actions: List[ExecutorAction] = []
        for executor in self.processed_data["executors_dominant_filled"] + self.processed_data["executors_hedge_filled"]:
            if self.market_data_provider.time() - executor.timestamp >= self.config.quoter_cooldown:
                # Create a new executor to keep the position / 创建新执行器来保持仓位
                stop_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=executor.id, keep_position=True))
        return stop_actions

    def get_executors_to_refresh(self) -> List[ExecutorAction]:
        """
        Get executors to refresh (stop and recreate) after refresh period.
        在刷新期后获取要刷新（停止并重新创建）的执行器。

        For unfilled orders that have been placed for longer than the refresh period,
        this method stops them so new orders can be placed at updated prices.
        对于下单时间超过刷新期的未成交订单，此方法停止它们以便在更新价格下下新订单。

        Returns:
            List[ExecutorAction]: Stop actions for executors needing refresh (需要刷新的执行器的停止动作)
        """
        refresh_actions: List[ExecutorAction] = []
        for executor in self.processed_data["executors_dominant_placed"] + self.processed_data["executors_hedge_placed"]:
            if self.market_data_provider.time() - executor.timestamp >= self.config.quoter_refresh:
                # Create a new executor to refresh the position / 创建新执行器来刷新仓位
                refresh_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=executor.id, keep_position=False))
        return refresh_actions

    def get_executors_to_quote(self) -> List[ExecutorAction]:
        """
        Get Order Executor to quote from the dominant and hedge markets.
        获取订单执行器从主要和对冲市场报价。

        This method creates position executors for both dominant and hedge assets
        based on the current signal and market conditions.
        此方法根据当前信号和市场条件为主要和对冲资产创建仓位执行器。

        Returns:
            List[ExecutorAction]: Actions to create new position executors (创建新仓位执行器的动作)
        """
        actions: List[ExecutorAction] = []
        trade_type_dominant = TradeType.BUY if self.processed_data["signal"] == 1 else TradeType.SELL
        trade_type_hedge = TradeType.SELL if self.processed_data["signal"] == 1 else TradeType.BUY

        # Analyze dominant active orders, max deviation and imbalance to create a new executor
        # 分析主要活跃订单、最大偏差和不平衡来创建新执行器
        if self.processed_data["dominant_gap"] > Decimal("0") and \
                self.processed_data["filter_connector_pair"] != self.config.connector_pair_dominant and \
                len(self.processed_data["executors_dominant_placed"]) < self.config.max_orders_placed_per_side and \
                len(self.processed_data["executors_dominant_filled"]) < self.config.max_orders_filled_per_side:
            # Create Position Executor for dominant asset / 为主要资产创建仓位执行器
            if trade_type_dominant == TradeType.BUY:
                price = self.processed_data["min_price_dominant"] * (1 - self.config.quoter_spread)
            else:
                price = self.processed_data["max_price_dominant"] * (1 + self.config.quoter_spread)
            dominant_executor_config = PositionExecutorConfig(
                timestamp=self.market_data_provider.time(),
                connector_name=self.config.connector_pair_dominant.connector_name,
                trading_pair=self.config.connector_pair_dominant.trading_pair,
                side=trade_type_dominant,
                entry_price=price,
                amount=self.config.min_amount_quote / self.processed_data["dominant_price"],
                triple_barrier_config=self.config.triple_barrier_config,
                leverage=self.config.leverage,
            )
            actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=dominant_executor_config))

        # Analyze hedge active orders, max deviation and imbalance to create a new executor
        # 分析对冲活跃订单、最大偏差和不平衡来创建新执行器
        if self.processed_data["hedge_gap"] > Decimal("0") and \
                self.processed_data["filter_connector_pair"] != self.config.connector_pair_hedge and \
                len(self.processed_data["executors_hedge_placed"]) < self.config.max_orders_placed_per_side and \
                len(self.processed_data["executors_hedge_filled"]) < self.config.max_orders_filled_per_side:
            # Create Position Executor for hedge asset / 为对冲资产创建仓位执行器
            if trade_type_hedge == TradeType.BUY:
                price = self.processed_data["min_price_hedge"] * (1 - self.config.quoter_spread)
            else:
                price = self.processed_data["max_price_hedge"] * (1 + self.config.quoter_spread)
            hedge_executor_config = PositionExecutorConfig(
                timestamp=self.market_data_provider.time(),
                connector_name=self.config.connector_pair_hedge.connector_name,
                trading_pair=self.config.connector_pair_hedge.trading_pair,
                side=trade_type_hedge,
                entry_price=price,
                amount=self.config.min_amount_quote / self.processed_data["hedge_price"],
                triple_barrier_config=self.config.triple_barrier_config,
                leverage=self.config.leverage,
            )
            actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=hedge_executor_config))
        return actions

    def get_executors_to_reduce_position(self, position: PositionSummary) -> List[ExecutorAction]:
        """
        Get Order Executor to reduce position.
        获取订单执行器来减少仓位。

        Creates market orders to close existing positions when needed for risk management
        or when positions need to be unwound due to signal changes.
        在需要进行风险管理或由于信号变化需要平仓时，创建市价订单来平掉现有仓位。

        Args:
            position (PositionSummary): Position to be reduced/closed (要减少/平掉的仓位)

        Returns:
            List[ExecutorAction]: Actions to create market order executors for closing (创建市价订单执行器用于平仓的动作)
        """
        if position.amount > Decimal("0"):
            # Close position / 平掉仓位
            config = OrderExecutorConfig(
                timestamp=self.market_data_provider.time(),
                connector_name=position.connector_name,
                trading_pair=position.trading_pair,
                side=TradeType.BUY if position.side == TradeType.SELL else TradeType.SELL,
                amount=position.amount,
                position_action=PositionAction.CLOSE,
                execution_strategy=ExecutionStrategy.MARKET,
                leverage=self.config.leverage,
            )
            return [CreateExecutorAction(controller_id=self.config.id, executor_config=config)]
        return []

    async def update_processed_data(self):
        """
        Update processed data with the latest market information and statistical calculations
        needed for the statistical arbitrage strategy.
        使用统计套利策略所需的最新市场信息和统计计算更新处理数据。

        This method performs the core statistical analysis including:
        - Calculating spread and z-score between asset pairs
        - Generating trading signals based on z-score thresholds
        - Updating position and order tracking data
        - Computing risk metrics and imbalances

        此方法执行核心统计分析，包括：
        - 计算资产对之间的价差和Z分数
        - 根据Z分数阈值生成交易信号
        - 更新仓位和订单追踪数据
        - 计算风险指标和不平衡
        """
        # Stat arb analysis / 统计套利分析
        spread, z_score = self.get_spread_and_z_score()

        # Generate trading signal based on z-score / 根据Z分数生成交易信号
        entry_threshold = float(self.config.entry_threshold)
        if z_score > entry_threshold:
            # Spread is too high, expect it to revert: long dominant, short hedge
            # 价差过高，预期回归：做多主要资产，做空对冲资产
            signal = 1
            dominant_side, hedge_side = TradeType.BUY, TradeType.SELL
        elif z_score < -entry_threshold:
            # Spread is too low, expect it to revert: short dominant, long hedge
            # 价差过低，预期回归：做空主要资产，做多对冲资产
            signal = -1
            dominant_side, hedge_side = TradeType.SELL, TradeType.BUY
        else:
            # No signal / 无信号
            signal = 0
            dominant_side, hedge_side = None, None

        # Current prices / 当前价格
        dominant_price, hedge_price = self.get_pairs_prices()

        # Get current positions stats by signal / 根据信号获取当前仓位统计
        positions_dominant = next((position for position in self.positions_held if position.connector_name == self.config.connector_pair_dominant.connector_name and position.trading_pair == self.config.connector_pair_dominant.trading_pair and (position.side == dominant_side or dominant_side is None)), None)
        positions_hedge = next((position for position in self.positions_held if position.connector_name == self.config.connector_pair_hedge.connector_name and position.trading_pair == self.config.connector_pair_hedge.trading_pair and (position.side == hedge_side or hedge_side is None)), None)
        # Get position stats / 获取仓位统计
        position_dominant_quote = positions_dominant.amount_quote if positions_dominant else Decimal("0")
        position_hedge_quote = positions_hedge.amount_quote if positions_hedge else Decimal("0")
        position_dominant_pnl_quote = positions_dominant.global_pnl_quote if positions_dominant else Decimal("0")
        position_hedge_pnl_quote = positions_hedge.global_pnl_quote if positions_hedge else Decimal("0")
        pair_pnl_pct = (position_dominant_pnl_quote + position_hedge_pnl_quote) / (position_dominant_quote + position_hedge_quote) if (position_dominant_quote + position_hedge_quote) != 0 else Decimal("0")
        # Get active executors / 获取活跃执行器
        executors_dominant_placed, executors_dominant_filled = self.get_executors_dominant()
        executors_hedge_placed, executors_hedge_filled = self.get_executors_hedge()
        min_price_dominant = Decimal(str(min([executor.config.entry_price for executor in executors_dominant_placed]))) if executors_dominant_placed else None
        max_price_dominant = Decimal(str(max([executor.config.entry_price for executor in executors_dominant_placed]))) if executors_dominant_placed else None
        min_price_hedge = Decimal(str(min([executor.config.entry_price for executor in executors_hedge_placed]))) if executors_hedge_placed else None
        max_price_hedge = Decimal(str(max([executor.config.entry_price for executor in executors_hedge_placed]))) if executors_hedge_placed else None

        active_amount_dominant = Decimal(str(sum([executor.filled_amount_quote for executor in executors_dominant_filled])))
        active_amount_hedge = Decimal(str(sum([executor.filled_amount_quote for executor in executors_hedge_filled])))

        # Compute imbalance based on the hedge ratio / 根据对冲比率计算不平衡
        dominant_gap = self.theoretical_dominant_quote - position_dominant_quote - active_amount_dominant
        hedge_gap = self.theoretical_hedge_quote - position_hedge_quote - active_amount_hedge
        imbalance = position_dominant_quote - position_hedge_quote
        imbalance_scaled = position_dominant_quote - position_hedge_quote * self.config.pos_hedge_ratio
        imbalance_scaled_pct = imbalance_scaled / position_dominant_quote if position_dominant_quote != Decimal("0") else Decimal("0")
        filter_connector_pair = None
        if imbalance_scaled_pct > self.config.max_position_deviation:
            # Avoid placing orders in the dominant market / 避免在主要市场下单
            filter_connector_pair = self.config.connector_pair_dominant
        elif imbalance_scaled_pct < -self.config.max_position_deviation:
            # Avoid placing orders in the hedge market / 避免在对冲市场下单
            filter_connector_pair = self.config.connector_pair_hedge

        # Update processed data / 更新处理数据
        self.processed_data.update({
            "dominant_price": Decimal(str(dominant_price)),
            "hedge_price": Decimal(str(hedge_price)),
            "spread": Decimal(str(spread)),
            "z_score": Decimal(str(z_score)),
            "dominant_gap": Decimal(str(dominant_gap)),
            "hedge_gap": Decimal(str(hedge_gap)),
            "position_dominant_quote": position_dominant_quote,
            "position_hedge_quote": position_hedge_quote,
            "active_amount_dominant": active_amount_dominant,
            "active_amount_hedge": active_amount_hedge,
            "signal": signal,
            # Store full dataframes for reference / 存储完整数据框以供参考
            "imbalance": Decimal(str(imbalance)),
            "imbalance_scaled_pct": Decimal(str(imbalance_scaled_pct)),
            "filter_connector_pair": filter_connector_pair,
            "min_price_dominant": min_price_dominant if min_price_dominant is not None else Decimal(str(dominant_price)),
            "max_price_dominant": max_price_dominant if max_price_dominant is not None else Decimal(str(dominant_price)),
            "min_price_hedge": min_price_hedge if min_price_hedge is not None else Decimal(str(hedge_price)),
            "max_price_hedge": max_price_hedge if max_price_hedge is not None else Decimal(str(hedge_price)),
            "executors_dominant_filled": executors_dominant_filled,
            "executors_hedge_filled": executors_hedge_filled,
            "executors_dominant_placed": executors_dominant_placed,
            "executors_hedge_placed": executors_hedge_placed,
            "pair_pnl_pct": pair_pnl_pct,
        })

    def get_spread_and_z_score(self):
        """
        Calculate the spread and z-score between the two assets for statistical arbitrage.
        计算两个资产之间的价差和Z分数用于统计套利。

        This method performs the core statistical analysis by:
        1. Fetching historical candle data for both assets
        2. Calculating percentage returns and cumulative returns
        3. Performing linear regression to find the relationship
        4. Computing the spread as deviation from predicted values
        5. Calculating z-score to standardize the spread

        此方法执行核心统计分析：
        1. 获取两个资产的历史K线数据
        2. 计算百分比收益和累积收益
        3. 执行线性回归以找到关系
        4. 计算价差作为与预测值的偏差
        5. 计算Z分数以标准化价差

        Returns:
            tuple: (current_spread, current_z_score) or None if insufficient data （当前价差，当前Z分数）或如果数据不足则返回None
        """
        # Fetch candle data for both assets / 获取两个资产的K线数据
        dominant_df = self.market_data_provider.get_candles_df(
            connector_name=self.config.connector_pair_dominant.connector_name,
            trading_pair=self.config.connector_pair_dominant.trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        hedge_df = self.market_data_provider.get_candles_df(
            connector_name=self.config.connector_pair_hedge.connector_name,
            trading_pair=self.config.connector_pair_hedge.trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        if dominant_df.empty or hedge_df.empty:
            self.logger().warning("Not enough candle data available for statistical analysis")
            return

        # Extract close prices / 提取收盘价
        dominant_prices = dominant_df['close'].values
        hedge_prices = hedge_df['close'].values

        # Ensure we have enough data and both series have the same length
        # 确保我们有足够的数据且两个序列长度相同
        min_length = min(len(dominant_prices), len(hedge_prices))
        if min_length < self.config.lookback_period:
            self.logger().warning(
                f"Not enough data points for analysis. Required: {self.config.lookback_period}, Available: {min_length}")
            return

        # Use the most recent data points / 使用最近的数据点
        dominant_prices = dominant_prices[-self.config.lookback_period:]
        hedge_prices = hedge_prices[-self.config.lookback_period:]

        # Convert to numpy arrays / 转换为numpy数组
        dominant_prices_np = np.array(dominant_prices, dtype=float)
        hedge_prices_np = np.array(hedge_prices, dtype=float)

        # Calculate percentage returns / 计算百分比收益
        dominant_pct_change = np.diff(dominant_prices_np) / dominant_prices_np[:-1]
        hedge_pct_change = np.diff(hedge_prices_np) / hedge_prices_np[:-1]

        # Convert to cumulative returns / 转换为累积收益
        dominant_cum_returns = np.cumprod(dominant_pct_change + 1)
        hedge_cum_returns = np.cumprod(hedge_pct_change + 1)

        # Normalize to start at 1 / 标准化以从1开始
        dominant_cum_returns = dominant_cum_returns / dominant_cum_returns[0] if len(dominant_cum_returns) > 0 else np.array([1.0])
        hedge_cum_returns = hedge_cum_returns / hedge_cum_returns[0] if len(hedge_cum_returns) > 0 else np.array([1.0])

        # Perform linear regression / 执行线性回归
        dominant_cum_returns_reshaped = dominant_cum_returns.reshape(-1, 1)
        reg = LinearRegression().fit(dominant_cum_returns_reshaped, hedge_cum_returns)
        alpha = reg.intercept_
        beta = reg.coef_[0]
        self.processed_data.update({
            "alpha": alpha,
            "beta": beta,
        })

        # Calculate spread as percentage difference from predicted value
        # 计算价差作为与预测值的百分比差异
        y_pred = alpha + beta * dominant_cum_returns
        spread_pct = (hedge_cum_returns - y_pred) / y_pred * 100

        # Calculate z-score / 计算Z分数
        mean_spread = np.mean(spread_pct)
        std_spread = np.std(spread_pct)
        if std_spread == 0:
            self.logger().warning("Standard deviation of spread is zero, cannot calculate z-score")
            return

        current_spread = spread_pct[-1]
        current_z_score = (current_spread - mean_spread) / std_spread

        return current_spread, current_z_score

    def get_pairs_prices(self):
        """
        Get current mid prices for both trading pairs.
        获取两个交易对的当前中间价。

        Returns:
            tuple: (dominant_price, hedge_price) - Current mid prices（主要价格，对冲价格）- 当前中间价
        """
        current_dominant_price = self.market_data_provider.get_price_by_type(
            connector_name=self.config.connector_pair_dominant.connector_name,
            trading_pair=self.config.connector_pair_dominant.trading_pair, price_type=PriceType.MidPrice)

        current_hedge_price = self.market_data_provider.get_price_by_type(
            connector_name=self.config.connector_pair_hedge.connector_name,
            trading_pair=self.config.connector_pair_hedge.trading_pair, price_type=PriceType.MidPrice)
        return current_dominant_price, current_hedge_price

    def get_executors_dominant(self):
        """
        Get active executors for the dominant trading pair.
        获取主要交易对的活跃执行器。

        Returns:
            tuple: (placed_executors, filled_executors) - Active and filled executors （已下单执行器，已成交执行器）- 活跃和已成交的执行器
        """
        active_executors_dominant_placed = self.filter_executors(
            self.executors_info,
            filter_func=lambda e: e.connector_name == self.config.connector_pair_dominant.connector_name and e.trading_pair == self.config.connector_pair_dominant.trading_pair and e.is_active and not e.is_trading and e.type == "position_executor"
        )
        active_executors_dominant_filled = self.filter_executors(
            self.executors_info,
            filter_func=lambda e: e.connector_name == self.config.connector_pair_dominant.connector_name and e.trading_pair == self.config.connector_pair_dominant.trading_pair and e.is_active and e.is_trading and e.type == "position_executor"
        )
        return active_executors_dominant_placed, active_executors_dominant_filled

    def get_executors_hedge(self):
        """
        Get active executors for the hedge trading pair.
        获取对冲交易对的活跃执行器。

        Returns:
            tuple: (placed_executors, filled_executors) - Active and filled executors（已下单执行器，已成交执行器）- 活跃和已成交的执行器
        """
        active_executors_hedge_placed = self.filter_executors(
            self.executors_info,
            filter_func=lambda e: e.connector_name == self.config.connector_pair_hedge.connector_name and e.trading_pair == self.config.connector_pair_hedge.trading_pair and e.is_active and not e.is_trading and e.type == "position_executor"
        )
        active_executors_hedge_filled = self.filter_executors(
            self.executors_info,
            filter_func=lambda e: e.connector_name == self.config.connector_pair_hedge.connector_name and e.trading_pair == self.config.connector_pair_hedge.trading_pair and e.is_active and e.is_trading and e.type == "position_executor"
        )
        return active_executors_hedge_placed, active_executors_hedge_filled

    def to_format_status(self) -> List[str]:
        """
        Format the status of the controller for display.
        格式化控制器状态用于显示。

        Creates a formatted string showing current strategy status including:
        - Trading pair information and parameters
        - Position targets and current positions
        - Active executors count
        - Signal, z-score, and PnL information

        创建格式化字符串显示当前策略状态，包括：
        - 交易对信息和参数
        - 仓位目标和当前仓位
        - 活跃执行器数量
        - 信号、Z分数和盈亏信息

        Returns:
            List[str]: Formatted status lines for display (用于显示的格式化状态行)
        """
        status_lines = []
        status_lines.append(f"""
Dominant Pair: {self.config.connector_pair_dominant} | Hedge Pair: {self.config.connector_pair_hedge} |
Timeframe: {self.config.interval} | Lookback Period: {self.config.lookback_period} | Entry Threshold: {self.config.entry_threshold}

Positions targets:
Theoretical Dominant         : {self.theoretical_dominant_quote} | Theoretical Hedge: {self.theoretical_hedge_quote} | Position Hedge Ratio: {self.config.pos_hedge_ratio}
Position Dominant            : {self.processed_data['position_dominant_quote']:.2f} | Position Hedge: {self.processed_data['position_hedge_quote']:.2f} | Imbalance: {self.processed_data['imbalance']:.2f} | Imbalance Scaled: {self.processed_data['imbalance_scaled_pct']:.2f} %

Current Executors:
Active Orders Dominant       : {len(self.processed_data['executors_dominant_placed'])} | Active Orders Hedge       : {len(self.processed_data['executors_hedge_placed'])} |
Active Orders Dominant Filled: {len(self.processed_data['executors_dominant_filled'])} | Active Orders Hedge Filled: {len(self.processed_data['executors_hedge_filled'])}

Signal: {self.processed_data['signal']:.2f} | Z-Score: {self.processed_data['z_score']:.2f} | Spread: {self.processed_data['spread']:.2f}
Alpha : {self.processed_data['alpha']:.2f} | Beta: {self.processed_data['beta']:.2f}
Pair PnL PCT: {self.processed_data['pair_pnl_pct'] * 100:.2f} %
""")
        return status_lines


# 统计套利策略参数优化建议和操作指导
# Statistical Arbitrage Strategy Parameter Optimization Guide

"""
## 核心参数分析与建议

### 1. 统计分析参数 (Statistical Analysis Parameters)

#### lookback_period = 300（回望期）
- 当前值：300个周期
- 建议调整范围：200-500
- 影响：
  * 过小(100-200)：对短期波动敏感，信号频繁但可能不稳定
  * 过大(500+)：信号稳定但反应滞后，可能错过机会
- 优化建议：
  * 高波动市场：使用较小值(200-250)
  * 稳定市场：使用较大值(400-500)
  * 与interval配合：1m周期用300，5m周期可用100-150

#### entry_threshold = 2.0（进入阈值）
- 当前值：2.0（Z分数）
- 建议调整范围：1.5-3.0
- 影响：
  * 过低(1.0-1.5)：交易频繁，手续费高，假信号多
  * 过高(3.0+)：交易机会少，但胜率高
- 优化建议：
  * 回测确定最优值，通常1.8-2.5之间
  * 高相关性资产可用较低值(1.5-2.0)
  * 低相关性资产用较高值(2.5-3.0)

### 2. 风险管理参数 (Risk Management Parameters)

#### take_profit = 0.0008（单次止盈）
- 当前值：0.08%
- 建议调整范围：0.0005-0.002
- 影响：与entry_threshold需要匹配
- 优化建议：
  * entry_threshold=2.0时，take_profit建议0.0008-0.0012
  * 需要覆盖交易成本（手续费约0.0002-0.0004）

#### tp_global = 0.01, sl_global = 0.05（全局止盈止损）
- 当前值：1%止盈，5%止损
- 建议调整：
  * 风险偏好低：0.5%止盈，2%止损
  * 风险偏好高：2%止盈，10%止损
- 重要性：这是最重要的风险控制，建议风险收益比1:2-1:4

### 3. 交易执行参数 (Execution Parameters)

#### min_amount_quote = 10（最小交易金额）
- 建议根据资金规模调整：
  * 小资金(1000-5000U)：5-10U
  * 中等资金(10000-50000U)：20-50U
  * 大资金(100000U+)：100U+

#### quoter_spread = 0.0001（报价价差）
- 当前值：0.01%
- 建议调整：
  * 高流动性币对：0.0001-0.0002
  * 低流动性币对：0.0003-0.0005
  * 需要平衡成交率和利润

#### quoter_cooldown = 30, quoter_refresh = 10（冷却和刷新时间）
- cooldown建议：
  * 高频交易：15-30秒
  * 中频交易：60-120秒
- refresh建议：
  * 活跃市场：5-10秒
  * 稳定市场：15-30秒

### 4. 仓位管理参数 (Position Management)

#### pos_hedge_ratio = 1.0（对冲比率）
- 建议：通常保持1.0，让算法自动计算Beta
- 特殊情况：如果发现某个资产系统性强于另一个，可调整为0.8-1.2

#### max_position_deviation = 0.1（最大仓位偏差）
- 当前值：10%
- 建议：5%-15%之间
- 影响：控制仓位不平衡，防止单边风险

#### leverage = 20（杠杆倍数）
- 高风险参数，建议根据经验调整：
  * 新手：3-5倍
  * 有经验：10-15倍
  * 专业：20倍+
- 注意：杠杆越高，爆仓风险越大

### 5. 订单控制参数 (Order Control)

#### max_orders_placed_per_side = 2, max_orders_filled_per_side = 2
- 建议保持较小值(1-3)避免过度交易
- 资金大的情况下可以适当增加到3-5

## 参数相互影响关系

### 主要关联关系：

1. **entry_threshold ↔ take_profit**
    - entry_threshold高 → take_profit可以相应提高
    - 比例关系大约：take_profit = entry_threshold * 0.0004

2. **lookback_period ↔ interval ↔ entry_threshold**
    - interval短 → lookback_period需要更大
    - lookback_period大 → entry_threshold可以适当降低

3. **min_amount_quote ↔ leverage ↔ 总资金**
    - leverage高 → min_amount_quote可以相对较小
    - 总资金 = min_amount_quote * leverage * 预期最大订单数

4. **quoter_spread ↔ take_profit ↔ 交易成本**
    - take_profit必须覆盖 quoter_spread + 手续费
    - quoter_spread太小可能成交困难

## 实际操作建议

### 新手推荐配置：
- entry_threshold = 2.0
- take_profit = 0.001
- tp_global = 0.005, sl_global = 0.02
- leverage = 5
- min_amount_quote = 20
- quoter_cooldown = 60

### 进阶配置：
- entry_threshold = 1.8
- take_profit = 0.0008
- tp_global = 0.01, sl_global = 0.03
- leverage = 10-15
- min_amount_quote = 50
- quoter_cooldown = 30

### 专业配置：
- entry_threshold = 根据回测优化(1.5-2.5)
- take_profit = 根据历史价差优化
- tp_global = 0.015, sl_global = 0.05
- leverage = 15-20
- 动态调整所有参数

## 监控要点

1. **Z分数分布**：观察历史Z分数是否正态分布
2. **信号频率**：每天信号数量是否合理(建议5-20次)
3. **胜率**：建议保持55%-70%
4. **最大回撤**：不应超过总资金的10%
5. **夏普比率**：目标>1.0

## 风险提醒

1. **协整性检验**：定期检查两个资产是否仍保持协整关系
2. **市场制度变化**：重大事件可能破坏历史关系
3. **流动性风险**：确保两个资产都有足够流动性
4. **技术风险**：网络延迟、API限制等技术问题
5. **资金管理**：不要投入超过能承受损失的资金

## 持续优化策略

1. **定期回测**：每月用最新数据回测参数
2. **参数优化**：使用网格搜索或遗传算法优化参数组合
3. **多时间框架**：尝试不同的interval设置
4. **资产对选择**：定期评估不同资产对的表现
5. **资金曲线分析**：监控策略的资金曲线和最大回撤

记住：统计套利的核心是概率优势，不是每笔交易都会盈利，
重要的是长期的期望收益为正。建议从小资金开始，
逐步增加投入，持续优化参数配置。
"""

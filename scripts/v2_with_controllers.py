import os
from decimal import Decimal
from typing import Dict, List, Optional, Set

from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction


class V2WithControllersConfig(StrategyV2ConfigBase):
    """
    控制器策略配置类 / Controller strategy configuration class
    定义了策略运行所需的所有配置参数 / Defines all configuration parameters required for strategy execution
    """
    
    # 脚本文件名，用于识别当前策略脚本 / Script file name for identifying current strategy script
    # 自动获取当前文件名，用于策略识别和日志记录
    # Automatically gets current file name for strategy identification and logging
    script_file_name: str = os.path.basename(__file__)
    
    # K线数据配置列表，用于获取市场数据 / Candles configuration list for market data retrieval
    # 定义需要订阅的K线数据源，包括交易对、时间周期等信息
    # 如果策略需要技术指标分析，需要配置相应的K线数据
    # Defines candlestick data sources to subscribe to, including trading pairs and time intervals
    # Required if strategy needs technical indicator analysis
    candles_config: List[CandlesConfig] = []
    
    # 市场交易对字典，键为交易所名称，值为交易对集合 / Markets dictionary with exchange names as keys and trading pairs as values
    # 格式示例: {"binance": {"BTC-USDT", "ETH-USDT"}, "kucoin": {"ADA-USDT"}}
    # 定义策略将在哪些交易所的哪些交易对上运行
    # Example format: {"binance": {"BTC-USDT", "ETH-USDT"}, "kucoin": {"ADA-USDT"}}
    # Defines which trading pairs on which exchanges the strategy will operate
    markets: Dict[str, Set[str]] = {}
    
    # 全局最大回撤限制（计价货币），用于风险控制 / Global maximum drawdown limit in quote currency for risk control
    # 当所有控制器的总回撤超过此值时，整个策略将停止运行
    # 设置为None表示不启用全局回撤保护，建议根据风险承受能力设置合理值
    # When total drawdown of all controllers exceeds this value, the entire strategy will stop
    # Set to None to disable global drawdown protection; recommended to set based on risk tolerance
    max_global_drawdown_quote: Optional[float] = None
    
    # 单个控制器最大回撤限制（计价货币），用于单控制器风险管理 / Individual controller maximum drawdown limit in quote currency for per-controller risk management
    # 当单个控制器的回撤超过此值时，该控制器将被停止，但不影响其他控制器
    # 设置为None表示不启用单控制器回撤保护，适用于精细化风险管理
    # When individual controller drawdown exceeds this value, that controller will be stopped without affecting others
    # Set to None to disable per-controller drawdown protection; useful for granular risk management
    max_controller_drawdown_quote: Optional[float] = None


class V2WithControllers(StrategyV2Base):
    """
    带控制器的V2策略主类 / V2 Strategy with Controllers Main Class
    
    == 核心功能 Core Functionality ==
    这是一个通用的多控制器策略管理框架，支持同时运行多个交易控制器，每个控制器可以独立执行不同的交易策略。
    主要功能包括：统一的风险管理、性能监控、资源协调和生命周期管理。
    This is a generic multi-controller strategy management framework that supports running multiple trading controllers 
    simultaneously, each capable of executing different trading strategies independently. Main features include unified 
    risk management, performance monitoring, resource coordination, and lifecycle management.
    
    == 工作原理 Working Principles ==
    1. 控制器管理：统一管理多个交易控制器的启动、停止和重启
    2. 执行器协调：协调各控制器下的订单执行器，确保资源合理分配
    3. 实时监控：持续监控各控制器的交易状态、盈亏和风险指标
    4. 动态配置：支持运行时配置更新和控制器参数调整
    5. 优雅退出：支持手动和自动现金出场机制，确保订单安全关闭
    
    1. Controller Management: Unified management of multiple trading controllers' start, stop, and restart operations
    2. Executor Coordination: Coordinates order executors under each controller for optimal resource allocation
    3. Real-time Monitoring: Continuously monitors trading status, PnL, and risk metrics of all controllers
    4. Dynamic Configuration: Supports runtime configuration updates and controller parameter adjustments
    5. Graceful Exit: Supports manual and automatic cash-out mechanisms with safe order closure
    
    == 风险管理 Risk Management ==
    多层次风险控制体系：
    - 全局回撤保护：监控所有控制器的总体回撤，超限时停止整个策略
    - 单控制器回撤保护：独立监控每个控制器的回撤，单独停止风险控制器
    - 手动紧急停止：支持手动触发特定控制器或整体策略的紧急停止
    - 执行器状态管理：自动清理非交易状态的执行器，防止资源泄漏
    - 性能报告机制：定期发送性能数据，支持外部监控和告警
    
    Multi-layered risk control system:
    - Global Drawdown Protection: Monitors overall drawdown of all controllers, stops entire strategy when exceeded
    - Per-Controller Drawdown Protection: Independently monitors each controller's drawdown, stops risky controllers individually
    - Manual Emergency Stop: Supports manual triggering of specific controller or overall strategy emergency stop
    - Executor Status Management: Automatically cleans up non-trading executors to prevent resource leaks
    - Performance Reporting: Regularly sends performance data supporting external monitoring and alerts
    """
    # 性能报告间隔时间（秒），定义多久发送一次性能报告 / Performance report interval in seconds, defines how often to send performance reports
    # 该参数控制策略向外部系统（如监控面板、日志系统等）发送性能数据的频率
    # 值为1表示每秒发送一次报告，包含所有控制器的PnL、交易状态等信息
    # Performance report interval controls how frequently the strategy sends performance data to external systems
    # A value of 1 means reports are sent every second, containing PnL, trading status, and other metrics for all controllers
    performance_report_interval: int = 1

    def __init__(self, connectors: Dict[str, ConnectorBase], config: V2WithControllersConfig):
        """
        初始化V2控制器策略 / Initialize V2 Controllers Strategy
        
        Args:
            connectors (Dict[str, ConnectorBase]): 交易所连接器字典 / Dictionary of exchange connectors
                - 键为交易所名称（如 "binance", "kucoin"）/ Keys are exchange names (e.g., "binance", "kucoin")
                - 值为对应的连接器实例，负责与交易所API通信 / Values are connector instances responsible for exchange API communication
                - 提供订单管理、市场数据获取、账户信息查询等功能 / Provides order management, market data retrieval, account information queries
                
            config (V2WithControllersConfig): 策略配置对象 / Strategy configuration object
                - 包含策略运行所需的所有配置参数 / Contains all configuration parameters required for strategy execution
                - 定义回撤限制、市场配置、K线数据源等关键设置 / Defines drawdown limits, market configuration, candlestick data sources
                - 控制策略的风险管理和运行行为 / Controls strategy's risk management and operational behavior
        """
        super().__init__(connectors, config)
        
        # 策略配置对象引用，用于访问用户定义的策略参数 / Strategy configuration object reference for accessing user-defined strategy parameters
        self.config = config
        
        # 每个控制器的最大盈亏记录字典，用于回撤计算 / Maximum PnL record dictionary for each controller, used for drawdown calculation
        # 格式: {controller_id: max_pnl_value} / Format: {controller_id: max_pnl_value}
        # 跟踪每个控制器的历史最高盈利点，作为回撤计算的基准 / Tracks historical highest profit point for each controller as drawdown calculation baseline
        self.max_pnl_by_controller = {}
        
        # 全局最大盈亏记录，用于全局回撤计算 / Global maximum PnL record, used for global drawdown calculation
        # 记录所有控制器总盈亏的历史最高点 / Records historical highest point of total PnL across all controllers
        # 用于触发全局回撤保护机制 / Used to trigger global drawdown protection mechanism
        self.max_global_pnl = Decimal("0")
        
        # 因回撤而退出的控制器ID列表 / List of controller IDs that exited due to drawdown
        # 防止已因风险管理而停止的控制器被意外重启 / Prevents controllers stopped due to risk management from being accidentally restarted
        # 确保风险控制决策的持久性 / Ensures persistence of risk control decisions
        self.drawdown_exited_controllers = []
        
        # 已关闭执行器的缓冲时间（秒），避免频繁检查 / Buffer time for closed executors in seconds, to avoid frequent checks
        # 用于优化性能，避免对已完成执行器的重复状态检查 / Used for performance optimization to avoid repeated status checks on completed executors
        self.closed_executors_buffer: int = 30
        
        # 上次性能报告的时间戳，控制报告发送频率 / Timestamp of last performance report, controls report sending frequency
        # 配合performance_report_interval使用，实现定时性能数据发布 / Used with performance_report_interval for timed performance data publishing
        self._last_performance_report_timestamp = 0

    def on_tick(self):
        """
        每个tick周期的主要逻辑处理 / Main logic processing for each tick cycle
        
        功能描述 / Function Description:
        - 这是策略的核心执行循环，每个时间周期都会被调用
        - 负责协调所有控制器的运行状态和风险管理
        - 确保系统的实时监控和响应能力
        
        执行流程 / Execution Flow:
        1. 调用父类的tick处理逻辑
        2. 检查策略是否被停止
        3. 依次执行手动控制、风险管理和性能报告
        """
        # 调用父类的tick处理方法，执行基础的策略框架逻辑 / Call parent class tick processing for basic strategy framework logic
        super().on_tick()
        
        # 检查策略停止标志，只有在策略未停止时才执行后续逻辑 / Check strategy stop flag, only execute subsequent logic when strategy is not stopped
        if not self._is_stop_triggered:
            # 检查手动停止开关，处理用户的手动控制指令 / Check manual kill switch, handle user manual control commands
            self.check_manual_kill_switch()
            
            # 执行回撤控制逻辑，保护账户安全 / Execute drawdown control logic to protect account safety
            self.control_max_drawdown()
            
            # 发送性能报告，更新外部监控系统 / Send performance report, update external monitoring systems
            self.send_performance_report()

    def control_max_drawdown(self):
        """
        控制最大回撤的主函数 / Main function to control maximum drawdown
        
        功能描述 / Function Description:
        - 统一管理回撤风险控制，是风险管理系统的入口点
        - 根据用户配置决定启用哪种回撤保护机制
        - 支持控制器级别和全局级别的双重保护
        
        风险控制层次 / Risk Control Levels:
        1. 单控制器回撤保护 - 防止单个策略过度亏损
        2. 全局回撤保护 - 防止整体投资组合过度亏损
        """
        # 检查是否启用了单控制器回撤限制配置 / Check if per-controller drawdown limit is configured
        if self.config.max_controller_drawdown_quote:
            # 执行单个控制器的回撤检查和保护 / Execute individual controller drawdown check and protection
            self.check_max_controller_drawdown()
            
        # 检查是否启用了全局回撤限制配置 / Check if global drawdown limit is configured  
        if self.config.max_global_drawdown_quote:
            # 执行全局级别的回撤检查和保护 / Execute global-level drawdown check and protection
            self.check_max_global_drawdown()

    def check_max_controller_drawdown(self):
        """
        检查每个控制器的最大回撤 / Check maximum drawdown for each controller
        
        功能描述 / Function Description:
        - 监控每个控制器的盈亏变化，实时计算回撤幅度
        - 当单个控制器回撤超过预设限制时，自动停止该控制器
        - 保护其他控制器不受影响，实现风险隔离
        
        处理流程 / Processing Flow:
        1. 遍历所有控制器
        2. 获取当前盈亏并与历史最高盈亏比较
        3. 计算回撤幅度并判断是否超限
        4. 超限时停止控制器并清理相关执行器
        """
        # 遍历所有注册的控制器 / Iterate through all registered controllers
        for controller_id, controller in self.controllers.items():
            # 跳过非运行状态的控制器，避免无效检查 / Skip non-running controllers to avoid invalid checks
            if controller.status != RunnableStatus.RUNNING:
                continue
            
            # 从性能报告中获取控制器当前总盈亏 / Get current total PnL from performance report
            controller_pnl = self.get_performance_report(controller_id).global_pnl_quote
            # 获取该控制器记录的历史最大盈亏值 / Get recorded historical maximum PnL for this controller
            last_max_pnl = self.max_pnl_by_controller[controller_id]
            
            # 检查是否创造新的盈亏高点 / Check if new PnL high is achieved
            if controller_pnl > last_max_pnl:
                # 更新最大盈亏记录，为后续回撤计算提供基准 / Update max PnL record as baseline for future drawdown calculations
                self.max_pnl_by_controller[controller_id] = controller_pnl
            else:
                # 计算从峰值到当前的回撤幅度 / Calculate drawdown from peak to current value
                current_drawdown = last_max_pnl - controller_pnl
                # 判断回撤是否超过用户设定的安全阈值 / Check if drawdown exceeds user-defined safety threshold
                if current_drawdown > self.config.max_controller_drawdown_quote:
                    # 记录回撤超限事件到日志 / Log drawdown exceeded event
                    self.logger().info(f"Controller {controller_id} reached max drawdown. Stopping the controller.")
                    # 立即停止控制器，防止进一步亏损 / Immediately stop controller to prevent further losses
                    controller.stop()
                    
                    # 筛选出该控制器下活跃但非交易状态的执行器 / Filter active but non-trading executors under this controller
                    executors_order_placed = self.filter_executors(
                        executors=self.get_executors_by_controller(controller_id),
                        filter_func=lambda x: x.is_active and not x.is_trading,
                    )
                    # 批量停止相关执行器，确保资源释放 / Batch stop related executors to ensure resource release
                    self.executor_orchestrator.execute_actions(
                        actions=[StopExecutorAction(controller_id=controller_id, executor_id=executor.id) for executor in executors_order_placed]
                    )
                    # 记录控制器因回撤而退出，防止意外重启 / Record controller exit due to drawdown, prevent accidental restart
                    self.drawdown_exited_controllers.append(controller_id)

    def check_max_global_drawdown(self):
        """
        检查全局最大回撤 / Check maximum global drawdown
        
        功能描述 / Function Description:
        - 监控所有控制器的总体盈亏表现
        - 当整体投资组合回撤过大时，紧急停止所有策略
        - 提供最后一道风险防线，保护整体资金安全
        
        处理流程 / Processing Flow:
        1. 汇总所有控制器的盈亏数据
        2. 与历史最高总盈亏比较
        3. 计算全局回撤幅度
        4. 超限时触发紧急停止机制
        """
        # 汇总计算所有活跃控制器的盈亏总和 / Aggregate total PnL from all active controllers
        current_global_pnl = sum([self.get_performance_report(controller_id).global_pnl_quote for controller_id in self.controllers.keys()])
        
        # 检查全局盈亏是否达到新的历史高点 / Check if global PnL reaches new historical high
        if current_global_pnl > self.max_global_pnl:
            # 更新全局最大盈亏记录，作为回撤计算基准 / Update global max PnL record as baseline for drawdown calculation
            self.max_global_pnl = current_global_pnl
        else:
            # 计算从全局盈亏峰值的回撤幅度 / Calculate drawdown from global PnL peak
            current_global_drawdown = self.max_global_pnl - current_global_pnl
            # 检查全局回撤是否触及危险阈值 / Check if global drawdown reaches dangerous threshold
            if current_global_drawdown > self.config.max_global_drawdown_quote:
                # 将所有控制器标记为因回撤退出，防止重启 / Mark all controllers as drawdown-exited to prevent restart
                self.drawdown_exited_controllers.extend(list(self.controllers.keys()))
                # 记录全局回撤事件到系统日志 / Log global drawdown event to system log
                self.logger().info("Global drawdown reached. Stopping the strategy.")
                # 设置策略停止标志，阻止后续tick执行 / Set strategy stop flag to prevent subsequent tick execution
                self._is_stop_triggered = True
                # 调用应用程序停止方法，安全退出整个系统 / Call application stop method for safe system exit
                HummingbotApplication.main_application().stop()

    def send_performance_report(self):
        """
        发送性能报告 / Send performance report
        
        功能描述 / Function Description:
        - 定期向外部系统发送策略性能数据
        - 支持实时监控和数据分析
        - 为外部告警和决策系统提供数据支持
        
        数据内容 / Data Content:
        - 各控制器的盈亏状况
        - 交易统计和执行器状态
        - 风险指标和性能指标
        """
        # 检查时间间隔条件：是否达到发送间隔且发布器可用 / Check timing conditions: interval reached and publisher available
        if self.current_timestamp - self._last_performance_report_timestamp >= self.performance_report_interval and self._pub:
            # 遍历收集所有控制器的完整性能数据 / Iterate and collect complete performance data for all controllers
            performance_reports = {controller_id: self.get_performance_report(controller_id).dict() for controller_id in self.controllers.keys()}
            # 通过发布器发送性能报告到外部订阅者 / Send performance reports to external subscribers via publisher
            self._pub(performance_reports)
            # 记录本次发送时间，用于下次间隔计算 / Record current send time for next interval calculation
            self._last_performance_report_timestamp = self.current_timestamp

    def check_manual_kill_switch(self):
        for controller_id, controller in self.controllers.items():
            if controller.config.manual_kill_switch and controller.status == RunnableStatus.RUNNING:
                self.logger().info(f"Manual cash out for controller {controller_id}.")
                controller.stop()
                executors_to_stop = self.get_executors_by_controller(controller_id)
                self.executor_orchestrator.execute_actions(
                    [StopExecutorAction(executor_id=executor.id,
                                        controller_id=executor.controller_id) for executor in executors_to_stop])
            if not controller.config.manual_kill_switch and controller.status == RunnableStatus.TERMINATED:
                if controller_id in self.drawdown_exited_controllers:
                    continue
                self.logger().info(f"Restarting controller {controller_id}.")
                controller.start()

    def check_executors_status(self):
        active_executors = self.filter_executors(
            executors=self.get_all_executors(),
            filter_func=lambda executor: executor.status == RunnableStatus.RUNNING
        )
        if not active_executors:
            self.logger().info("All executors have finalized their execution. Stopping the strategy.")
            HummingbotApplication.main_application().stop()
        else:
            non_trading_executors = self.filter_executors(
                executors=active_executors,
                filter_func=lambda executor: not executor.is_trading
            )
            self.executor_orchestrator.execute_actions(
                [StopExecutorAction(executor_id=executor.id,
                                    controller_id=executor.controller_id) for executor in non_trading_executors])

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        return []

    def apply_initial_setting(self):
        """
        应用初始设置 / Apply initial settings
        为每个控制器设置初始PnL记录，并配置永续合约的持仓模式和杠杆 / Set initial PnL records for each controller and configure position mode and leverage for perpetual contracts
        """
        # 存储连接器的持仓模式配置 / Store position mode configuration for connectors
        connectors_position_mode = {}
        
        # 遍历所有控制器进行初始化设置 / Iterate through all controllers for initialization
        for controller_id, controller in self.controllers.items():
            # 初始化每个控制器的最大PnL为0 / Initialize maximum PnL for each controller to 0
            self.max_pnl_by_controller[controller_id] = Decimal("0")
            
            # 获取控制器配置字典 / Get controller configuration dictionary
            config_dict = controller.config.model_dump()
            
            # 如果配置中包含连接器名称 / If configuration contains connector name
            if "connector_name" in config_dict:
                # 检查是否为永续合约连接器 / Check if it's a perpetual contract connector
                if self.is_perpetual(config_dict["connector_name"]):
                    # 设置持仓模式 / Set position mode
                    if "position_mode" in config_dict:
                        connectors_position_mode[config_dict["connector_name"]] = config_dict["position_mode"]
                    # 设置杠杆倍数 / Set leverage
                    if "leverage" in config_dict:
                        self.connectors[config_dict["connector_name"]].set_leverage(leverage=config_dict["leverage"],
                                                                                    trading_pair=config_dict["trading_pair"])
        
        # 为每个连接器应用持仓模式设置 / Apply position mode settings for each connector
        for connector_name, position_mode in connectors_position_mode.items():
            self.connectors[connector_name].set_position_mode(position_mode)


# 📚 V2多控制器策略完整使用指南 📚
# Complete Usage Guide for V2 Multi-Controller Strategy

"""
🎯【超详细教程】手把手教你玩转Hummingbot V2多控制器策略！
=================================================================

🔥 什么是V2多控制器策略？
-----------------------------
想象一下你是个交易高手，可以同时在多个交易所用不同策略赚钱💰
这就是V2多控制器策略！它让你：
✅ 同时运行多个独立的交易控制器
✅ 智能风险管理，自动止损保护
✅ 实时性能监控，数据一目了然
✅ 支持手动和自动退出机制

📝 快速上手配置
-------------------------------

第一步：配置基础参数 🛠️

```python
# 创建配置对象
config = V2WithControllersConfig(
    # 脚本文件名（自动设置，不用管）
    script_file_name="v2_with_controllers.py",
    
    # 📈 K线数据配置（如果需要技术指标分析）
    candles_config=[
        CandlesConfig(
            connector="binance",
            trading_pair="BTC-USDT", 
            interval="1m"
        )
    ],
    
    # 🏪 交易市场配置
    markets={
        "binance": {"BTC-USDT", "ETH-USDT"},
        "kucoin": {"ADA-USDT", "DOT-USDT"}
    },
    
    # 🛡️ 全局最大回撤保护（建议设置）
    max_global_drawdown_quote=1000.0,  # 全局最大亏损1000 USDT就停止
    
    # 🎯 单控制器最大回撤保护
    max_controller_drawdown_quote=200.0  # 单个策略最大亏损200 USDT就停止
)
```

第二步：启动策略 🚀

```python
# 连接交易所
connectors = {
    "binance": your_binance_connector,
    "kucoin": your_kucoin_connector
}

# 创建策略实例
strategy = V2WithControllers(connectors=connectors, config=config)

# 开始运行！
strategy.start()
```

🎛️ 核心功能详解
-----------------

💡 智能风险管理系统
~~~~~~~~~~~~~~~~~~~
这个功能真的是救命神器！自动帮你：

🔴 **单控制器保护**
- 监控每个策略的盈亏
- 回撤超限自动停止该策略
- 其他策略继续运行，风险隔离

🔴 **全局回撤保护** 
- 监控总体投资组合表现
- 整体亏损过大紧急停止所有策略
- 最后一道安全防线

🔴 **手动紧急停止**
- 随时可以手动停止特定策略
- 支持一键停止所有策略
- 安全退出，订单妥善处理

📊 实时性能监控
~~~~~~~~~~~~~~~
```python
# 性能报告每秒更新一次
performance_report_interval = 1

# 自动发送数据到外部监控系统
# 包含：盈亏状况、交易统计、风险指标
```

🎯 使用场景推荐
---------------

🌟 **场景一：多策略组合**
适合：想要分散风险的用户
配置：不同交易对使用不同策略
好处：降低单一策略风险，提高收益稳定性

🌟 **场景二：多交易所套利**  
适合：追求无风险套利的用户
配置：同一交易对在不同交易所运行
好处：捕捉交易所间价差机会

🌟 **场景三：大资金分仓管理**
适合：资金量大的用户
配置：同一策略分多个控制器运行
好处：降低单笔订单冲击，优化执行效果

⚠️ 重要注意事项
---------------

🚨 **风险管理设置**
```python
# 建议设置：
max_global_drawdown_quote = 账户余额 * 0.1  # 最大承受10%亏损
max_controller_drawdown_quote = 单策略资金 * 0.2  # 单策略最大承受20%亏损
```

🚨 **资源管理**
- 控制器数量不要过多（建议<10个）
- 监控系统资源使用情况
- 定期检查日志文件

🚨 **连接稳定性**
- 确保网络连接稳定
- 配置好API密钥权限
- 监控交易所连接状态

💰 实战技巧分享
---------------

✨ **新手推荐配置**
```python
# 保守型配置
max_global_drawdown_quote = 100.0      # 低风险
max_controller_drawdown_quote = 50.0   # 单策略低风险
performance_report_interval = 5        # 5秒报告一次
```

✨ **进阶用户配置**
```python
# 激进型配置  
max_global_drawdown_quote = 500.0      # 中等风险
max_controller_drawdown_quote = 150.0  # 单策略中等风险
performance_report_interval = 1        # 1秒报告一次
```

✨ **专业交易员配置**
```python
# 高频交易配置
max_global_drawdown_quote = 1000.0     # 高风险高收益
max_controller_drawdown_quote = 200.0  # 精细化风险控制
performance_report_interval = 1        # 实时监控
```

🔧 常见问题解决
---------------

❓ **Q: 策略突然停止了怎么办？**
A: 检查是否触发了回撤保护，查看日志文件了解具体原因

❓ **Q: 可以动态添加新的控制器吗？**  
A: 当前版本不支持运行时动态添加，需要重启策略

❓ **Q: 性能报告发送失败怎么办？**
A: 检查网络连接和发布器配置，确保外部系统正常

❓ **Q: 如何优化策略性能？**
A: 合理设置控制器数量，监控系统资源，调优报告间隔

🎉 结语
-------
V2多控制器策略真的是交易自动化的神器！
记住：先小金额测试，熟悉后再加大投入💪

希望这个教程对大家有帮助！
有问题欢迎在评论区讨论～

#Hummingbot #量化交易 #自动化交易 #策略框架 #风险管理




v2_with_controllers.py 使用说明】

🎯 **脚本用途**
这是一个通用的 Strategy V2 主控制器脚本，用于管理和监控多个子控制器的运行状态。
特别适合运行统计套利（stat_arb_fish）等复杂交易策略。

🔧 **核心组件关系**
1. **v2_with_controllers.py（主策略）**：负责全局协调和风险控制
2. **stat_arb_fish.py（子控制器）**：实现具体的配对交易逻辑
3. **执行器编排器**：统一管理所有订单的执行

📊 **风险管理特性**
- **单个控制器回撤保护**：当某个策略亏损过大时自动停止
- **全局回撤保护**：当总体亏损超限时紧急停止所有交易
- **手动开关**：支持手动暂停/恢复特定策略
- **智能重启**：区分手动停止和风险停止，防止问题策略重启

🚀 **使用场景**
1. **多策略组合**：同时运行多个不同的交易策略
2. **配对交易**：使用 stat_arb_fish 进行统计套利
3. **风险管理**：对高风险策略进行严格的资金管理
4. **策略研究**：快速测试和调整策略参数

⚙️ **配置要点**
- max_global_drawdown_quote: 建议设置为总资金的10-20%
- max_controller_drawdown_quote: 建议设置为单个策略资金的15-30%
- 永续合约自动配置杠杆和仓位模式
- 支持纸上交易模式进行无风险测试

📈 **监控功能**
- 实时性能报告：每秒发送各策略的PnL数据
- 状态监控：跟踪控制器和执行器状态
- 风险警告：回撤接近阈值时主动提醒
- 交易记录：完整记录所有交易操作
"""

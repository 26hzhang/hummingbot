import sys
import warnings
from pathlib import Path

import backtrader as bt
import backtrader.analyzers as btanalyzers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arbitragelab.other_approaches.kalman_filter import KalmanFilterStrategy
from sklearn.linear_model import LinearRegression

# Add current directory to path for local imports
sys.path.append(str(Path(__file__).parent))
warnings.filterwarnings('ignore')


class DataLoader:

    def __init__(self, data_dir="/Users/zhanghao/GitHub/hummingbot/data/", market="futures", interval="1m"):
        self.interval = interval
        _data_dir = data_dir + f"{market}_{interval}"
        self.data_dir = Path(_data_dir)
        self.pairs = [f.stem.replace(f"_{interval}", "") for f in self.data_dir.glob(f"*_{interval}.csv")]

    def load_pair_data(self, coin1, coin2, start_date=None, end_date=None):
        pair1 = f"{coin1}USDT"
        pair2 = f"{coin2}USDT"

        # 检查文件存在性
        file1 = self.data_dir / f"{pair1}_{self.interval}.csv"
        file2 = self.data_dir / f"{pair2}_{self.interval}.csv"

        missing_files = []
        if not file1.exists():
            missing_files.append(f"{pair1}_{self.interval}.csv")
        if not file2.exists():
            missing_files.append(f"{pair2}_{self.interval}.csv")

        if missing_files:
            return None

        # 加载数据
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # 转换时间戳
        df1['timestamp'] = pd.to_datetime(df1['open_time'], utc=True)
        df2['timestamp'] = pd.to_datetime(df2['open_time'], utc=True)

        # 计算共同时间范围
        common_start = max(df1['timestamp'].min(), df2['timestamp'].min())
        common_end = min(df1['timestamp'].max(), df2['timestamp'].max())

        # 用户指定日期范围
        if start_date:
            start_date_tz = pd.to_datetime(start_date, utc=True)
            common_start = max(common_start, start_date_tz)
        if end_date:
            end_date_tz = pd.to_datetime(end_date, utc=True)
            common_end = min(common_end, end_date_tz)

        # 过滤到共同时间范围并排序
        df1_filtered = df1[(df1['timestamp'] >= common_start) & (df1['timestamp'] <= common_end)].sort_values('timestamp').reset_index(drop=True)
        df2_filtered = df2[(df2['timestamp'] >= common_start) & (df2['timestamp'] <= common_end)].sort_values('timestamp').reset_index(drop=True)

        # 构建对齐数据
        df1_clean = df1_filtered[['timestamp', 'open', 'high', 'low', 'close', 'volume']].rename(columns={
            'open': f'{coin1}_open', 'high': f'{coin1}_high', 'low': f'{coin1}_low',
            'close': f'{coin1}_close', 'volume': f'{coin1}_volume'
        })
        df2_clean = df2_filtered[['timestamp', 'open', 'high', 'low', 'close', 'volume']].rename(columns={
            'open': f'{coin2}_open', 'high': f'{coin2}_high', 'low': f'{coin2}_low',
            'close': f'{coin2}_close', 'volume': f'{coin2}_volume'
        })

        # 内连接得到对齐的数据
        data = pd.merge(df1_clean, df2_clean, on='timestamp', how='inner').sort_values('timestamp').reset_index(drop=True)

        return data


class AssetDataFeed(bt.feeds.PandasData):
    """
    单资产数据源 - 用于双数据源架构
    """
    params = (
        ('datetime', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', -1),
    )


class KalmanFilterWrapper:
    """
    Kalman滤波器包装器，适配backtrader实时交易环境
    """

    def __init__(self, obs_cov=None, trans_cov=None, delta=1e-2):
        self.obs_cov = obs_cov
        self.trans_cov = trans_cov
        self.delta = delta
        self.kalman_filter = None
        self.initialized = False
        if self.obs_cov is not None and self.trans_cov is not None:
            self.initialize()

        # 存储历史价格用于参数校准
        self.price1_history = []
        self.price2_history = []

    def _calibrate_parameters(self, price1_history, price2_history):
        """使用OLS残差校准Kalman参数"""
        n_train = min(len(price1_history), 200)

        X = np.array(price1_history[:n_train]).reshape(-1, 1)
        y = np.array(price2_history[:n_train])

        ols = LinearRegression(fit_intercept=True).fit(X, y)
        ols_residuals = y - ols.predict(X)
        ols_variance = np.var(ols_residuals, ddof=1)

        obs_cov = ols_variance
        trans_cov = self.delta * obs_cov / (1 - self.delta)

        return obs_cov, trans_cov

    def initialize(self):
        if self.obs_cov is None or self.trans_cov is None:
            obs_cov, trans_cov = self._calibrate_parameters(
                self.price1_history, self.price2_history
            )
            self.obs_cov = obs_cov
            self.trans_cov = trans_cov
        # 初始化Kalman滤波器
        self.kalman_filter = KalmanFilterStrategy(
            observation_covariance=self.obs_cov,
            transition_covariance=self.trans_cov
        )
        # 预热Kalman滤波器
        for i in range(len(self.price1_history)):
            self.kalman_filter.update(self.price1_history[i], self.price2_history[i])
        self.initialized = True

    def update(self, price1, price2):
        """更新Kalman滤波器"""
        self.kalman_filter.update(price1, price2)

    def get_current_estimates(self):
        """获取当前参数估计"""
        if not self.initialized or not self.kalman_filter:
            return None

        if len(self.kalman_filter.hedge_ratios) == 0:
            return None

        current_alpha = self.kalman_filter.intercepts[-1]
        current_beta = self.kalman_filter.hedge_ratios[-1]

        # 获取预测误差和标准差用于z-score计算
        if len(self.kalman_filter.spread_series) > 0:
            prediction_error = self.kalman_filter.spread_series[-1]
            prediction_std = self.kalman_filter.spread_std_series[-1]
        else:
            prediction_error = 0.0
            prediction_std = 1.0

        return {
            'alpha': current_alpha,
            'beta': current_beta,
            'prediction_error': prediction_error,
            'prediction_std': prediction_std
        }


class StatArbKalmanStrategy(bt.Strategy):
    """
    基于Kalman滤波的统计套利策略
    """

    params = (
        # 资产标识参数
        ('coin1', 'ASSET1'),     # 第一个资产名称
        ('coin2', 'ASSET2'),     # 第二个资产名称

        # Kalman滤波参数
        ('kalman_obs_cov', None),     # Kalman观测协方差
        ('kalman_trans_cov', None),   # Kalman转换协方差
        ('kalman_delta', 1e-2),       # 控制β漂移速度的参数

        # 交易信号参数
        ('entry_sigma', 2.0),         # 入场标准差倍数
        ('exit_sigma', 0.0),          # 出场标准差倍数
        ('stop_loss_sigma', 3.0),     # 止损标准差倍数

        # 风险管理参数
        ('position_size', 0.1),  # 每次交易使用10%资金
        ('max_positions', 1),    # 最大同时持仓数

        # Beta波动率过滤参数
        ('beta_volatility_threshold', 0.05),  # beta变化波动率阈值 (归一化，相对于beta均值)
        ('beta_lookback_period', 20),          # beta波动率计算回望期

        # 调试参数
        ('debug', False),
    )

    def __init__(self):
        """初始化策略"""

        # 双数据源架构 - 获取两个资产的数据引用
        self.asset1_data = self.datas[0]  # 第一个资产 (如MANA)
        self.asset2_data = self.datas[1]  # 第二个资产 (如SAND)

        # 便捷访问当前价格数据
        self.asset1_close = self.asset1_data.close
        self.asset1_high = self.asset1_data.high
        self.asset1_low = self.asset1_data.low
        self.asset2_close = self.asset2_data.close
        self.asset2_high = self.asset2_data.high
        self.asset2_low = self.asset2_data.low

        # 策略状态变量
        self.current_spread_position = None  # 'long_spread', 'short_spread', None
        self.entry_zscore = 0.0
        self.current_alpha = 0.0
        self.current_beta = 0.0

        self.warmup_period = 2000  # Kalman滤波预热期

        # 持仓跟踪
        self.asset1_position_size = 0.0  # 实际asset1持仓大小
        self.asset2_position_size = 0.0  # 实际asset2持仓大小

        # Kalman滤波器初始化
        self.kalman_wrapper = KalmanFilterWrapper(
            obs_cov=self.params.kalman_obs_cov,
            trans_cov=self.params.kalman_trans_cov,
            delta=self.params.kalman_delta
        )

        # 历史数据存储
        self.zscore_history = []
        self.alpha_history = []
        self.beta_history = []
        self.portfolio_value_history = []

        # 交易统计
        self.trade_count = 0
        self.signal_count = 0
        self.beta_filtered_count = 0  # Beta波动率过大过滤的交易次数

        # Beta波动率状态跟踪
        self.last_beta_volatility = 0.0  # 上次计算的beta波动率

        # Hedge交易跟踪
        self.pair_trades = []
        self.current_pair_trade = None

        # 从参数中获取币种信息
        self.coin1 = self.params.coin1
        self.coin2 = self.params.coin2

        if self.params.debug:
            print(f"Kalman策略初始化完成: {self.coin1}-{self.coin2}")
            print(f"交易参数: entry=±{self.params.entry_sigma}σ, exit={self.params.exit_sigma}σ")

    def next(self):
        """主要的策略逻辑 - 每个数据点调用一次"""

        # 获取当前数据索引
        current_idx = len(self.asset1_data) - 1

        # 获取当前价格（使用对数变换）
        current_p1 = np.log(self.asset1_close[0])
        current_p2 = np.log(self.asset2_close[0])

        # 预热期数据收集 - 仅收集数据，不交易
        if current_idx < self.warmup_period and not self.kalman_wrapper.initialized:
            self.kalman_wrapper.price1_history.append(current_p1)
            self.kalman_wrapper.price2_history.append(current_p2)
            return

        # 收集到足够的预热数据，初始化Kalman滤波器
        if self.kalman_wrapper.kalman_filter is None or not self.kalman_wrapper.initialized:
            print(f"Kalman滤波器预热完成，初始化中... 共收集 {len(self.kalman_wrapper.price1_history)} 条数据")
            self.kalman_wrapper.initialize()
            return  # 初始化后等待下一周期，即可开始正式交易

        # 更新Kalman滤波器
        self.kalman_wrapper.update(current_p1, current_p2)

        # 获取当前参数估计
        estimates = self.kalman_wrapper.get_current_estimates()
        if estimates is None:
            return

        self.current_alpha = estimates['alpha']
        self.current_beta = estimates['beta']
        prediction_error = estimates['prediction_error']
        prediction_std = estimates['prediction_std']

        # 计算z-score（使用Kalman滤波器的官方方法）
        if prediction_std > 0:
            current_zscore = prediction_error / prediction_std
        else:
            current_zscore = 0.0

        # 存储历史数据
        self.zscore_history.append(current_zscore)
        self.alpha_history.append(self.current_alpha)
        self.beta_history.append(self.current_beta)
        self.portfolio_value_history.append(self.broker.getvalue())

        # 价差交易逻辑
        self._execute_spread_trading_logic(current_zscore)

    def _is_beta_stable(self):
        """
        检查Beta稳定性 - 归一化波动率检测
        返回: True 如果beta稳定, False 如果beta波动过大
        """
        if len(self.beta_history) < self.params.beta_lookback_period:
            # Beta历史数据不足，认为不稳定
            return False

        # 获取最近N个beta值
        recent_betas = self.beta_history[-self.params.beta_lookback_period:]

        # 计算beta的绝对值均值和标准差
        beta_abs_mean = np.mean(np.abs(recent_betas))
        beta_std = np.std(recent_betas)

        # 避免除零错误
        if beta_abs_mean == 0:
            self.last_beta_volatility = 0.0
            return True

        # 计算归一化波动率
        normalized_volatility = beta_std / beta_abs_mean
        self.last_beta_volatility = normalized_volatility

        # 判断是否超过阈值
        is_stable = normalized_volatility < self.params.beta_volatility_threshold

        return is_stable

    def _execute_spread_trading_logic(self, zscore):
        """
        执行价差交易逻辑
        """
        self.signal_count += 1

        # 检查现有价差头寸的退出条件
        if self.current_spread_position is not None:
            should_exit, exit_reason = self._check_spread_exit_conditions(zscore)

            if should_exit:
                self._exit_spread_position(exit_reason)
                return

        # 检查新入场条件（仅在空仓时）
        if self.current_spread_position is None:
            entry_signal = self._check_spread_entry_conditions(zscore)

            if entry_signal is not None:
                # Beta稳定性检查 - 实时判断
                if not self._is_beta_stable():
                    self.beta_filtered_count += 1
                    if self.params.debug:
                        print(f"Beta波动率过大({self.last_beta_volatility:.4f}>{self.params.beta_volatility_threshold:.4f})，跳过本次交易")
                    return
                self._enter_spread_position(entry_signal, zscore)

    def _check_spread_exit_conditions(self, zscore):
        """
        检查价差头寸退出条件
        """
        if self.current_spread_position == 'long_spread':
            # 做多价差头寸的退出条件
            if zscore >= self.params.exit_sigma:
                return True, 'take_profit'
            elif zscore <= -self.params.stop_loss_sigma:
                return True, 'stop_loss'

        elif self.current_spread_position == 'short_spread':
            # 做空价差头寸的退出条件
            if zscore <= self.params.exit_sigma:
                return True, 'take_profit'
            elif zscore >= self.params.stop_loss_sigma:
                return True, 'stop_loss'

        return False, None

    def _check_spread_entry_conditions(self, zscore):
        """
        检查价差入场条件
        增强逻辑：要求当前突破阈值且上一根K线未突破阈值（避免持续超阈值区域加仓）
        """
        # 检查是否有足够的历史数据
        if len(self.zscore_history) < 2:
            return None

        # 获取上一根K线的z-score
        prev_zscore = self.zscore_history[-2]  # -1是当前，-2是上一根

        # 做多价差入场条件：当前z-score <= -entry_sigma 且 上一根z-score > -entry_sigma
        if zscore <= -self.params.entry_sigma and prev_zscore > -self.params.entry_sigma:
            return 'long_spread'  # Z-score刚突破负阈值，做多价差

        # 做空价差入场条件：当前z-score >= entry_sigma 且 上一根z-score < entry_sigma
        elif zscore >= self.params.entry_sigma and prev_zscore < self.params.entry_sigma:
            return 'short_spread'  # Z-score刚突破正阈值，做空价差

        return None

    def _enter_spread_position(self, signal, zscore):
        """
        建立价差头寸 - 双资产hedge交易

        价差定义: Spread = ln(Asset2) - α - β × ln(Asset1)
        做多价差: 买Asset2, 卖β×Asset1
        做空价差: 卖Asset2, 买β×Asset1
        """
        # 获取当前资金状态
        cash = self.broker.get_cash()
        portfolio_value = self.broker.getvalue()

        # 获取当前价格
        asset1_price = self.asset1_close[0]
        asset2_price = self.asset2_close[0]

        # 计算总投资金额
        V = portfolio_value * self.params.position_size

        # 弹性中性资金分配：按 1:|β| 比例分配
        abs_beta = abs(self.current_beta)
        P1 = float(asset1_price)
        P2 = float(asset2_price)

        # 资金分配
        value_y = V / (1.0 + abs_beta)           # Asset2分配资金
        value_x = abs_beta * V / (1.0 + abs_beta)    # Asset1分配资金

        # 计算交易数量
        asset2_size = value_y / P2
        asset1_size = value_x / P1

        # 验证资金充足性
        required_asset2_value = asset2_size * P2
        required_asset1_value = asset1_size * P1
        total_required = required_asset2_value + required_asset1_value

        if total_required > cash * 0.99:  # 留1%缓冲
            scale_factor = (cash * 0.99) / total_required
            asset2_size *= scale_factor
            asset1_size *= scale_factor

        # 创建新的hedge交易记录
        self.current_pair_trade = {
            'entry_time': self.data.datetime.datetime(0),
            'entry_zscore': zscore,
            'entry_alpha': self.current_alpha,
            'entry_beta': self.current_beta,
            'signal': signal,
            'entry_prices': {
                'asset1': asset1_price,
                'asset2': asset2_price
            },
            'sizes': {
                'asset1': asset1_size if signal == 'short_spread' else -asset1_size,
                'asset2': asset2_size if signal == 'long_spread' else -asset2_size
            },
            'entry_portfolio_value': self.broker.getvalue(),
            'theoretical_hedge_ratio': self.current_beta
        }

        if signal == 'long_spread':
            # 做多价差: 买Asset2, 卖β×Asset1
            self.buy(data=self.asset2_data, size=asset2_size)
            self.sell(data=self.asset1_data, size=asset1_size)
            self.current_spread_position = 'long_spread'
            self.asset2_position_size = asset2_size
            self.asset1_position_size = -asset1_size

        elif signal == 'short_spread':
            # 做空价差: 卖Asset2, 买β×Asset1
            self.sell(data=self.asset2_data, size=asset2_size)
            self.buy(data=self.asset1_data, size=asset1_size)
            self.current_spread_position = 'short_spread'
            self.asset2_position_size = -asset2_size
            self.asset1_position_size = asset1_size

        self.entry_zscore = zscore
        self.trade_count += 1

        if self.params.debug:
            print(f"建立价差头寸 {signal}: zscore={zscore:.3f}, α={self.current_alpha:.3f}, β={self.current_beta:.3f}")

    def _exit_spread_position(self, reason):
        """
        平掉价差头寸
        """
        if self.current_spread_position is not None and self.current_pair_trade is not None:
            # 获取退出时的信息
            exit_asset1_price = self.asset1_close[0]
            exit_asset2_price = self.asset2_close[0]
            current_zscore = self.zscore_history[-1] if self.zscore_history else 0.0

            # 完成hedge交易记录
            self.current_pair_trade.update({
                'exit_time': self.data.datetime.datetime(0),
                'exit_zscore': current_zscore,
                'exit_alpha': self.current_alpha,
                'exit_beta': self.current_beta,
                'exit_reason': reason,
                'exit_prices': {
                    'asset1': exit_asset1_price,
                    'asset2': exit_asset2_price
                },
                'exit_portfolio_value': self.broker.getvalue()
            })

            # 计算交易分析
            self._calculate_hedge_trade_analysis()

            # 将完成的交易添加到记录中
            self.pair_trades.append(self.current_pair_trade)

            # 平掉所有相关头寸
            self.close(data=self.asset1_data)
            self.close(data=self.asset2_data)

            if self.params.debug:
                print(f"平掉价差头寸: {self.current_spread_position}, 原因: {reason}")

            # 重置状态
            self.current_spread_position = None
            self.entry_zscore = 0.0
            self.asset1_position_size = 0.0
            self.asset2_position_size = 0.0
            self.current_pair_trade = None

    def _calculate_hedge_trade_analysis(self):
        """
        计算单次hedge交易的分析
        """
        if self.current_pair_trade is None:
            return

        trade = self.current_pair_trade

        # 信号分析
        entry_zscore = trade['entry_zscore']
        exit_zscore = trade['exit_zscore']
        signal = trade['signal']
        reason = trade['exit_reason']

        # 判断信号是否正确
        zscore_delta = exit_zscore - entry_zscore
        signal_success = False

        if signal == 'long_spread':
            signal_success = (zscore_delta > 0 and reason == 'take_profit')
        elif signal == 'short_spread':
            signal_success = (zscore_delta < 0 and reason == 'take_profit')

        # 实际盈亏分析
        entry_value = trade['entry_portfolio_value']
        exit_value = trade['exit_portfolio_value']
        total_hedge_pnl = exit_value - entry_value

        # 计算各资产的价格变化
        asset1_price_change = (trade['exit_prices']['asset1'] / trade['entry_prices']['asset1']) - 1
        asset2_price_change = (trade['exit_prices']['asset2'] / trade['entry_prices']['asset2']) - 1

        # 计算理论hedge盈亏（不考虑手续费等成本）
        asset1_size = trade['sizes']['asset1']
        asset2_size = trade['sizes']['asset2']

        theoretical_asset1_pnl = asset1_size * trade['entry_prices']['asset1'] * asset1_price_change
        theoretical_asset2_pnl = asset2_size * trade['entry_prices']['asset2'] * asset2_price_change
        theoretical_hedge_pnl = theoretical_asset1_pnl + theoretical_asset2_pnl

        # 计算hedge效率
        hedge_effectiveness = total_hedge_pnl / theoretical_hedge_pnl if theoretical_hedge_pnl != 0 else 1.0
        execution_cost = theoretical_hedge_pnl - total_hedge_pnl

        # 计算投入资金总额（基于持仓价值）
        position_investment = abs(asset1_size * trade['entry_prices']['asset1']) + abs(asset2_size * trade['entry_prices']['asset2'])

        # 计算盈亏比 (Profit %)
        profit_percentage = (total_hedge_pnl / position_investment) * 100 if position_investment > 0 else 0

        # 将分析结果添加到交易记录
        trade['analysis'] = {
            # Z-Score信号维度
            'zscore_signal_analysis': {
                'signal_correctness': reason,
                'entry_zscore': entry_zscore,
                'exit_zscore': exit_zscore,
                'zscore_delta': zscore_delta,
                'signal_success': signal_success
            },

            # 实际仓位盈亏维度
            'position_pnl_analysis': {
                'total_hedge_pnl': total_hedge_pnl,
                'theoretical_hedge_pnl': theoretical_hedge_pnl,
                'asset1_pnl': theoretical_asset1_pnl,
                'asset2_pnl': theoretical_asset2_pnl,
                'hedge_effectiveness': hedge_effectiveness,
                'execution_cost': execution_cost,
                'asset1_price_change': asset1_price_change,
                'asset2_price_change': asset2_price_change,
                'position_investment': position_investment,
                'profit_percentage': profit_percentage,
                'entry_alpha': trade['entry_alpha'],
                'entry_beta': trade['entry_beta'],
                'exit_alpha': trade['exit_alpha'],
                'exit_beta': trade['exit_beta']
            },

            # 四象限分类
            'trade_category': self._categorize_trade(signal_success, total_hedge_pnl > 0)
        }

    def _categorize_trade(self, signal_success, position_profit):
        """
        将交易分类到四象限
        """
        if signal_success and position_profit:
            return 'perfect_trade'
        elif signal_success and not position_profit:
            return 'signal_good_execution_poor'
        elif not signal_success and position_profit:
            return 'signal_poor_execution_good'
        else:
            return 'double_failure'


class StatArbBacktraderKalman:
    """
    基于Kalman滤波的统计套利Backtrader集成类
    """

    def __init__(self, coin1='MANA', coin2='SAND', data_dir="/Users/zhanghao/GitHub/hummingbot/data/", interval="1m"):
        self.coin1 = coin1
        self.coin2 = coin2
        self.data_dir = data_dir
        self.loader = DataLoader(data_dir, market="futures", interval=interval)
        self.cerebro = None
        self.results = None

    def prepare_data(self, start_date='2025-08-12', end_date=None):
        """
        准备双数据源回测数据
        """
        print(f"加载数据: {self.coin1}-{self.coin2}")

        # 使用现有DataLoader加载数据
        data = self.loader.load_pair_data(self.coin1, self.coin2, start_date, end_date)
        if data is None:
            raise ValueError("数据加载失败")

        # 准备两个单独的资产数据
        asset1_data = data[['timestamp', f'{self.coin1}_open', f'{self.coin1}_high',
                           f'{self.coin1}_low', f'{self.coin1}_close', f'{self.coin1}_volume']].copy()
        asset1_data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        asset1_data['datetime'] = pd.to_datetime(asset1_data['datetime'])
        asset1_data = asset1_data.set_index('datetime')

        asset2_data = data[['timestamp', f'{self.coin2}_open', f'{self.coin2}_high',
                           f'{self.coin2}_low', f'{self.coin2}_close', f'{self.coin2}_volume']].copy()
        asset2_data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        asset2_data['datetime'] = pd.to_datetime(asset2_data['datetime'])
        asset2_data = asset2_data.set_index('datetime')

        print(f"Asset1({self.coin1})数据: {len(asset1_data)} 条记录")
        print(f"Asset2({self.coin2})数据: {len(asset2_data)} 条记录")

        return asset1_data, asset2_data

    def run_backtest(self,
                     start_date='2025-08-12',
                     end_date=None,
                     initial_cash=100000,
                     commission=0.001,
                     strategy_params=None):
        """
        运行Kalman滤波双数据源backtrader回测
        """

        # 准备双数据源
        asset1_data, asset2_data = self.prepare_data(start_date, end_date)

        # 创建Cerebro引擎
        self.cerebro = bt.Cerebro()

        # 设置初始资金
        self.initial_cash = initial_cash
        self.cerebro.broker.set_cash(initial_cash)

        # 设置手续费
        self.cerebro.broker.setcommission(commission=commission)

        # 添加双数据源
        asset1_feed = AssetDataFeed(dataname=asset1_data, name=f'{self.coin1}')
        asset2_feed = AssetDataFeed(dataname=asset2_data, name=f'{self.coin2}')

        self.cerebro.adddata(asset1_feed)  # datas[0] = Asset1
        self.cerebro.adddata(asset2_feed)  # datas[1] = Asset2

        # 添加策略
        if strategy_params is None:
            strategy_params = {}

        # 将coin1和coin2传入策略参数
        strategy_params.update({
            'coin1': self.coin1,
            'coin2': self.coin2
        })

        self.cerebro.addstrategy(StatArbKalmanStrategy, **strategy_params)

        # 添加分析器
        self.cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(btanalyzers.TimeReturn, _name='time_return')

        # 添加观察器
        self.cerebro.addobserver(bt.observers.Broker)
        self.cerebro.addobserver(bt.observers.Trades)
        self.cerebro.addobserver(bt.observers.BuySell)

        print(f"开始Kalman滤波价差回测...")
        print(f"Asset1: {self.coin1}, Asset2: {self.coin2}")
        print(f"初始资金: ${initial_cash:,.2f}")
        print(f"手续费: {commission:.3%}")

        # 运行回测
        self.results = self.cerebro.run()

        # 获取最终价值
        final_value = self.cerebro.broker.getvalue()

        print(f"回测完成!")
        print(f"最终价值: ${final_value:,.2f}")
        print(f"总收益: {(final_value - initial_cash) / initial_cash:.2%}")

        return self.results

    def print_analysis(self):
        """
        打印详细的backtrader分析结果
        """
        if self.results is None:
            print("请先运行回测")
            return

        result = self.results[0]

        print("\n" + "=" * 80)
        print("📊 BACKTRADER 双资产HEDGE统计套利回测分析报告")
        print("=" * 80)

        # 获取初始资金和最终价值
        initial_cash = self.initial_cash
        final_value = self.cerebro.broker.getvalue()

        # 基本信息
        print(f"🎯 交易策略: Statistical Arbitrage")
        print(f"📈 交易对: {self.coin1} (Asset1) - {self.coin2} (Asset2)")
        print(f"💰 初始资金: ${initial_cash:,.2f}")
        print(f"💵 最终价值: ${final_value:,.2f}")

        # 收益分析
        total_return = (final_value - initial_cash) / initial_cash
        time_return_analyzer = result.analyzers.time_return

        print(f"\n📊 收益分析:")
        print(f"总收益: ${final_value - initial_cash:,.2f}")
        print(f"总收益率: {total_return:.2%}")

        # 年化收益率
        if hasattr(time_return_analyzer, 'get_analysis'):
            time_returns = time_return_analyzer.get_analysis()
            if time_returns:
                total_days = len(time_returns)
                if total_days > 0:
                    daily_return = total_return / total_days
                    annualized_return = (1 + daily_return) ** 252 - 1
                    print(f"年化收益率: {annualized_return:.2%}")

        # 风险分析
        print(f"\n⚠️  风险分析:")

        # 夏普比率
        sharpe_ratio = result.analyzers.sharpe.get_analysis().get('sharperatio', None)
        if sharpe_ratio is not None:
            print(f"夏普比率: {sharpe_ratio:.3f}")
        else:
            print("夏普比率: N/A (无足够数据)")

        # 回撤分析
        drawdown = result.analyzers.drawdown.get_analysis()
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0)
        max_drawdown_len = drawdown.get('max', {}).get('len', 0)

        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"最大回撤持续期: {max_drawdown_len} 周期")

        # 交易分析
        trades = result.analyzers.trades.get_analysis()
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        lost_trades = trades.get('lost', {}).get('total', 0)

        print(f"\n🔄 交易统计:")
        print(f"总交易次数: {total_trades}")
        print(f"盈利交易: {won_trades}")
        print(f"亏损交易: {lost_trades}")

        if total_trades > 0:
            win_rate = won_trades / total_trades
            print(f"胜率: {win_rate:.2%}")

            # 详细盈亏分析
            total_pnl = trades.get('pnl', {}).get('net', {}).get('total', 0)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

            won_pnl = trades.get('won', {}).get('pnl', {})
            lost_pnl = trades.get('lost', {}).get('pnl', {})

            avg_win = won_pnl.get('average', 0)
            avg_loss = lost_pnl.get('average', 0)
            max_win = won_pnl.get('max', 0)
            max_loss = lost_pnl.get('max', 0)

            print(f"平均每笔交易: ${avg_pnl:.2f}")
            print(f"平均盈利: ${avg_win:.2f}")
            print(f"平均亏损: ${avg_loss:.2f}")
            print(f"最大单笔盈利: ${max_win:.2f}")
            print(f"最大单笔亏损: ${max_loss:.2f}")

            if avg_loss != 0:
                profit_factor = abs(avg_win / avg_loss)
                print(f"盈亏比: {profit_factor:.2f}")

            # 交易持续时间
            len_stats = trades.get('len', {})
            if len_stats:
                avg_len = len_stats.get('average', 0)
                max_len = len_stats.get('max', 0)
                min_len = len_stats.get('min', 0)

                print(f"\n⏱️  持仓时间统计:")
                print(f"平均持仓时间: {avg_len:.1f} 个周期")
                print(f"最长持仓时间: {max_len} 个周期")
                print(f"最短持仓时间: {min_len} 个周期")

        # 策略特定统计
        print(f"\n🎛️  策略参数:")
        # 注意：result本身就是策略实例
        strategy_instance = result
        print(f"Entry Sigma: ±{strategy_instance.params.entry_sigma}")
        print(f"Exit Sigma: {strategy_instance.params.exit_sigma}")
        print(f"Stop Loss Sigma: ±{strategy_instance.params.stop_loss_sigma}")
        print(f"Position Size: {strategy_instance.params.position_size:.1%}")
        print(f"执行模式: Kalman滤波动态参数 (包含alpha截距项)")
        print(f"对冲模式: 弹性中性 (基于Kalman滤波alpha+beta)")

        # 市场表现对比
        print(f"\n📈 市场对比:")
        if hasattr(strategy_instance, 'zscore_history') and len(strategy_instance.zscore_history) > 0:
            z_scores = strategy_instance.zscore_history
            print(f"Z-Score 信号数: {len(z_scores)}")
            print(f"Z-Score 范围: {min(z_scores):.2f} 至 {max(z_scores):.2f}")

        print("=" * 80)

        # 双维度Hedge交易分析
        self._print_hedge_analysis(strategy_instance)

        # 保存z-score图表
        self._save_zscore_chart(strategy_instance)

    def _print_hedge_analysis(self, strategy_instance):
        """
        打印双维度Hedge交易分析报告
        """
        if not hasattr(strategy_instance, 'pair_trades') or len(strategy_instance.pair_trades) == 0:
            print("⚠️ 没有完整的hedge交易数据可供分析")
            return

        pair_trades = strategy_instance.pair_trades
        total_pairs = len(pair_trades)

        print(f"\n" + "=" * 80)
        print("🔄 双维度HEDGE交易分析报告")
        print("=" * 80)

        # 基本统计
        print(f"📊 基本统计:")
        print(f"总Hedge交易次数: {total_pairs}")

        if total_pairs == 0:
            return

        # 收集统计数据
        signal_success_count = 0
        position_profit_count = 0
        total_hedge_pnl = 0
        total_theoretical_pnl = 0
        total_execution_cost = 0

        # 四象限统计
        perfect_trades = 0
        signal_good_execution_poor = 0
        signal_poor_execution_good = 0
        double_failure = 0

        # 信号维度统计
        take_profit_count = 0
        stop_loss_count = 0

        # 持续时间统计
        durations = []
        hedge_pnls = []
        signal_success_pnls = []
        signal_failure_pnls = []

        for trade in pair_trades:
            analysis = trade.get('analysis', {})
            zscore_analysis = analysis.get('zscore_signal_analysis', {})
            pnl_analysis = analysis.get('position_pnl_analysis', {})

            # Z-Score信号统计
            signal_success = zscore_analysis.get('signal_success', False)
            if signal_success:
                signal_success_count += 1

            reason = zscore_analysis.get('signal_correctness', '')
            if 'take_profit' in reason:
                take_profit_count += 1
            elif 'stop_loss' in reason:
                stop_loss_count += 1

            # 仓位盈亏统计
            hedge_pnl = pnl_analysis.get('total_hedge_pnl', 0)
            theoretical_pnl = pnl_analysis.get('theoretical_hedge_pnl', 0)
            execution_cost = pnl_analysis.get('execution_cost', 0)

            hedge_pnls.append(hedge_pnl)
            total_hedge_pnl += hedge_pnl
            total_theoretical_pnl += theoretical_pnl
            total_execution_cost += execution_cost

            if hedge_pnl > 0:
                position_profit_count += 1

            # 按信号成功与否分类盈亏
            if signal_success:
                signal_success_pnls.append(hedge_pnl)
            else:
                signal_failure_pnls.append(hedge_pnl)

            # 四象限分类
            category = analysis.get('trade_category', '')
            if category == 'perfect_trade':
                perfect_trades += 1
            elif category == 'signal_good_execution_poor':
                signal_good_execution_poor += 1
            elif category == 'signal_poor_execution_good':
                signal_poor_execution_good += 1
            else:
                double_failure += 1

            # 持续时间
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60  # 分钟
                durations.append(duration)

        # 打印Z-Score信号维度分析
        print(f"\n📊 Z-Score信号维度分析:")
        signal_accuracy = signal_success_count / total_pairs if total_pairs > 0 else 0
        print(f"信号准确率: {signal_accuracy:.1%} ({signal_success_count}/{total_pairs})")
        print(f"- 止盈退出: {take_profit_count} trades ({take_profit_count / total_pairs:.1%})")
        print(f"- 止损退出: {stop_loss_count} trades ({stop_loss_count / total_pairs:.1%})")

        # 验证统计完整性
        accounted_trades = take_profit_count + stop_loss_count
        if accounted_trades != total_pairs:
            other_exits = total_pairs - accounted_trades
            print(f"- 其他退出: {other_exits} trades ({other_exits / total_pairs:.1%})")

        # 打印实际仓位盈亏维度分析
        print(f"\n💰 实际仓位盈亏维度分析:")
        position_win_rate = position_profit_count / total_pairs if total_pairs > 0 else 0
        print(f"Hedge盈利率: {position_win_rate:.1%} ({position_profit_count}/{total_pairs})")
        print(f"总Hedge盈亏: ${total_hedge_pnl:.2f}")
        print(f"平均单次Hedge盈亏: ${total_hedge_pnl / total_pairs:.2f}")
        print(f"最佳/最差Hedge: ${max(hedge_pnls):.2f} / ${min(hedge_pnls):.2f}")

        # 执行效率分析
        execution_efficiency = total_hedge_pnl / total_theoretical_pnl if total_theoretical_pnl != 0 else 1.0
        print(f"\n🎯 执行效率分析:")
        print(f"理论最大盈亏: ${total_theoretical_pnl:.2f}")
        print(f"实际盈亏: ${total_hedge_pnl:.2f}")
        print(f"执行效率: {execution_efficiency:.1%}")
        print(f"总执行成本: ${total_execution_cost:.2f}")

        # 四象限分析
        print(f"\n🎯 四象限分析:")
        print(f"✅信号对+✅盈利: {perfect_trades} trades ({perfect_trades / total_pairs:.1%}) - 完美交易")
        print(f"✅信号对+❌亏损: {signal_good_execution_poor} trades ({signal_good_execution_poor / total_pairs:.1%}) - 执行问题")
        print(f"❌信号错+✅盈利: {signal_poor_execution_good} trades ({signal_poor_execution_good / total_pairs:.1%}) - 意外收获")
        print(f"❌信号错+❌亏损: {double_failure} trades ({double_failure / total_pairs:.1%}) - 双重失败")

        # 信号成功vs失败的盈亏对比
        if signal_success_pnls and signal_failure_pnls:
            avg_signal_success_pnl = sum(signal_success_pnls) / len(signal_success_pnls)
            avg_signal_failure_pnl = sum(signal_failure_pnls) / len(signal_failure_pnls)
            print(f"\n📈 信号质量对盈亏的影响:")
            print(f"信号成功时平均盈亏: ${avg_signal_success_pnl:.2f}")
            print(f"信号失败时平均盈亏: ${avg_signal_failure_pnl:.2f}")
            print(f"信号质量收益差: ${avg_signal_success_pnl - avg_signal_failure_pnl:.2f}")

        # 盈亏比分布详细统计
        profit_percentages = []
        for trade in pair_trades:
            analysis = trade.get('analysis', {})
            pnl_analysis = analysis.get('position_pnl_analysis', {})
            profit_pct = pnl_analysis.get('profit_percentage', 0)
            profit_percentages.append(profit_pct)

        if profit_percentages:
            print(f"\n📊 盈亏比分布详细统计:")

            # 基本统计
            avg_profit_pct = sum(profit_percentages) / len(profit_percentages)
            max_profit_pct = max(profit_percentages)
            min_profit_pct = min(profit_percentages)
            win_trades = [p for p in profit_percentages if p > 0]
            loss_trades = [p for p in profit_percentages if p <= 0]

            print(f"平均盈亏比: {avg_profit_pct:.2f}%")
            print(f"盈利交易胜率: {len(win_trades) / len(profit_percentages):.1%} ({len(win_trades)}/{len(profit_percentages)})")
            print(f"最佳/最差盈亏比: {max_profit_pct:.2f}% / {min_profit_pct:.2f}%")

            if win_trades:
                avg_win = sum(win_trades) / len(win_trades)
                print(f"平均盈利幅度: {avg_win:.2f}%")

            if loss_trades:
                avg_loss = sum(loss_trades) / len(loss_trades)
                print(f"平均亏损幅度: {avg_loss:.2f}%")

            # 分区间统计占比
            intervals = [
                (-float('inf'), 0, "亏损 (<0%)"),
                (0, 0.02, "盈利 (0~0.02%)"),
                (0.02, 0.04, "盈利 (0.02~0.04%)"),
                (0.04, 0.06, "盈利 (0.04~0.06%)"),
                (0.06, 0.08, "盈利 (0.06~0.08%)"),
                (0.08, 0.1, "盈利 (0.08~0.1%)"),
                (0.1, 0.3, "盈利 (0.1~0.3%)"),
                (0.3, float("inf"), "盈利 (>=0.3%)"),
            ]

            print(f"\n📈 盈亏比区间分布:")
            for lower, upper, desc in intervals:
                if lower == -float('inf'):
                    count = len([p for p in profit_percentages if p < upper])
                elif upper == float('inf'):
                    count = len([p for p in profit_percentages if p >= lower])
                else:
                    count = len([p for p in profit_percentages if lower <= p < upper])

                percentage = count / len(profit_percentages) * 100
                print(f"{desc}: {count} trades ({percentage:.1f}%)")

        # 持续时间统计
        if durations:
            print(f"\n⏱️ Hedge持续时间统计:")
            print(f"平均持续时间: {sum(durations) / len(durations):.1f} 分钟")
            print(f"最长/最短持续时间: {max(durations):.1f} / {min(durations):.1f} 分钟")

        # Beta波动率过滤统计
        if hasattr(strategy_instance, 'beta_filtered_count'):
            trade_count = getattr(strategy_instance, 'trade_count', 0)
            filtered_count = strategy_instance.beta_filtered_count
            if trade_count > 0:
                filter_rate = filtered_count / trade_count
                print(f"Beta波动率过滤: {filtered_count}/{trade_count} 交易 ({filter_rate:.1%})")
            else:
                print(f"Beta波动率过滤: {filtered_count} 交易")

        print("=" * 80)

    def _save_zscore_chart(self, strategy_instance):
        """
        保存z-score和beta图表到当前目录
        """
        if not hasattr(strategy_instance, 'zscore_history') or len(strategy_instance.zscore_history) == 0:
            print("⚠️ 没有z-score数据可供绘图")
            return

        try:
            # 创建图表，包含4个子图：Z-Score、Beta、PnL曲线、Z-Score分布
            plt.figure(figsize=(15, 16))

            # 绘制z-score时间序列 - 跳过初始warmup期的异常值
            z_scores = strategy_instance.zscore_history
            beta_values = strategy_instance.beta_history if hasattr(strategy_instance, 'beta_history') else []

            # 跳过初始warmup期 (warmup_period + 50个点)
            warmup_skip = getattr(strategy_instance, 'warmup_period', 200) + 50
            skip_points = min(warmup_skip, len(z_scores) // 4)  # 最多跳过1/4的数据

            z_scores_filtered = z_scores[skip_points:] if len(z_scores) > skip_points else z_scores
            beta_values_filtered = beta_values[skip_points:] if len(beta_values) > skip_points else beta_values
            time_index_filtered = range(skip_points, skip_points + len(z_scores_filtered))

            plt.subplot(4, 1, 1)
            plt.plot(time_index_filtered, z_scores_filtered, 'b-', linewidth=1, alpha=0.7, label='Z-Score')

            if skip_points > 0:
                plt.axvline(x=skip_points, color='gray', linestyle=':', alpha=0.5,
                            label=f'Warmup Skip ({skip_points} points)')

            # 添加entry和exit水平线
            plt.axhline(y=strategy_instance.params.entry_sigma, color='red', linestyle='--',
                        linewidth=2, label=f'Entry Sigma (+{strategy_instance.params.entry_sigma})')
            plt.axhline(y=-strategy_instance.params.entry_sigma, color='red', linestyle='--',
                        linewidth=2, label=f'Entry Sigma (-{strategy_instance.params.entry_sigma})')
            plt.axhline(y=strategy_instance.params.stop_loss_sigma, color='orange', linestyle=':',
                        linewidth=2, label=f'Stop Loss Sigma (+{strategy_instance.params.stop_loss_sigma})')
            plt.axhline(y=-strategy_instance.params.stop_loss_sigma, color='orange', linestyle=':',
                        linewidth=2, label=f'Stop Loss Sigma (-{strategy_instance.params.stop_loss_sigma})')
            plt.axhline(y=strategy_instance.params.exit_sigma, color='green', linestyle='-',
                        linewidth=2, label=f'Exit ({strategy_instance.params.exit_sigma})')

            plt.title(f'{self.coin1}-{self.coin2} Statistical Arbitrage Z-Score Analysis', fontsize=14, fontweight='bold')
            plt.ylabel('Z-Score')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')

            # 添加颜色填充区域 - 使用过滤后的时间索引
            plt.fill_between(time_index_filtered, strategy_instance.params.entry_sigma, strategy_instance.params.stop_loss_sigma,
                             alpha=0.1, color='red')
            plt.fill_between(time_index_filtered, -strategy_instance.params.stop_loss_sigma, -strategy_instance.params.entry_sigma,
                             alpha=0.1, color='blue')

            # 第二个子图：显示beta变化趋势 - 使用过滤后的数据
            plt.subplot(4, 1, 2)
            if beta_values_filtered:
                beta_time_index = range(skip_points, skip_points + len(beta_values_filtered))
                plt.plot(beta_time_index, beta_values_filtered, 'r-', linewidth=1.5, alpha=0.8, label='Beta (Hedge Ratio)')
                plt.title('Beta (Hedge Ratio) Evolution', fontsize=12, fontweight='bold')
                plt.ylabel('Beta Value')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='upper right')

                # 显示beta统计信息 - 使用过滤后的数据
                if len(beta_values_filtered) > 0:
                    mean_beta = np.mean(beta_values_filtered)
                    std_beta = np.std(beta_values_filtered)
                    plt.axhline(y=mean_beta, color='orange', linestyle=':', alpha=0.7,
                                label=f'Mean Beta: {mean_beta:.3f}')
                    plt.axhline(y=mean_beta + std_beta, color='gray', linestyle=':', alpha=0.5)
                    plt.axhline(y=mean_beta - std_beta, color='gray', linestyle=':', alpha=0.5)
            else:
                plt.text(0.5, 0.5, 'No Beta Data Available',
                         transform=plt.gca().transAxes, ha='center', va='center', fontsize=14)

            # 第三个子图：显示Portfolio PnL曲线和交易标记点
            plt.subplot(4, 1, 3)
            portfolio_values = strategy_instance.portfolio_value_history if hasattr(strategy_instance, 'portfolio_value_history') else []

            if portfolio_values:
                portfolio_values_filtered = portfolio_values[skip_points:] if len(portfolio_values) > skip_points else portfolio_values

                # 计算PnL百分比
                if len(portfolio_values_filtered) > 0:
                    initial_value = portfolio_values_filtered[0]
                    pnl_percentage = [(value / initial_value - 1) * 100 for value in portfolio_values_filtered]

                    # 绘制PnL曲线 - 简洁的黑色曲线
                    plt.plot(time_index_filtered[:len(pnl_percentage)], pnl_percentage, 'black', linewidth=1.5, label='Portfolio PnL (%)')

                    # 添加零线
                    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)

                    plt.title('Portfolio PnL Curve', fontsize=12, fontweight='bold')
                    plt.ylabel('PnL (%)')
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='upper left')

                    # 显示最终收益
                    final_pnl = pnl_percentage[-1] if pnl_percentage else 0
                    plt.text(0.02, 0.95, f'Final PnL: {final_pnl:.2f}%',
                             transform=plt.gca().transAxes, verticalalignment='top',
                             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                plt.text(0.5, 0.5, 'No Portfolio Value Data Available',
                         transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)

            # 第四个子图：显示z-score分布直方图 - 使用过滤后的数据
            plt.subplot(4, 1, 4)
            if z_scores_filtered:
                # 进一步过滤极端异常值用于直方图显示
                z_scores_hist = [z for z in z_scores_filtered if abs(z) < 20]  # 移除|z|>20的极端值

                if z_scores_hist:
                    plt.hist(z_scores_hist, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.axvline(x=strategy_instance.params.entry_sigma, color='red', linestyle='--', linewidth=2)
                    plt.axvline(x=-strategy_instance.params.entry_sigma, color='red', linestyle='--', linewidth=2)
                    plt.axvline(x=strategy_instance.params.stop_loss_sigma, color='orange', linestyle=':', linewidth=2)
                    plt.axvline(x=-strategy_instance.params.stop_loss_sigma, color='orange', linestyle=':', linewidth=2)
                    plt.axvline(x=strategy_instance.params.exit_sigma, color='green', linestyle='-', linewidth=2)

                    # 添加统计信息
                    mean_z = np.mean(z_scores_hist)
                    plt.axvline(x=mean_z, color='black', linestyle='-', alpha=0.5, label=f'Mean: {mean_z:.2f}')
                    plt.text(0.02, 0.95, f'Filtered {len(z_scores) - len(z_scores_hist)} extreme values (|z|>20)',
                             transform=plt.gca().transAxes, verticalalignment='top', fontsize=10, alpha=0.7)
                else:
                    plt.text(0.5, 0.5, 'No valid Z-Score data for histogram',
                             transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
            else:
                plt.text(0.5, 0.5, 'No Z-Score data available',
                         transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)

            plt.title('Z-Score Distribution', fontsize=12, fontweight='bold')
            plt.xlabel('Z-Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图表
            filename = f"/Users/zhanghao/GitHub/hummingbot/statarb_project/{self.coin1}_{self.coin2}_zscore_beta_kalman_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Z-Score和Beta分析图表已保存: {filename}")

            # 可选：也显示图表
            # plt.show()
            plt.close()

        except Exception as e:
            print(f"❌ 保存z-score图表失败: {e}")

    def plot_results(self):
        """
        绘制回测结果
        """
        if self.cerebro is None:
            print("请先运行回测")
            return

        print("显示Kalman backtrader回测图表...")
        self.cerebro.plot(style='candlestick', volume=False, figsize=(20, 12))
        plt.show()


def main():
    """
    主函数 - 演示完整的backtrader集成
    """
    print("=" * 80)
    print("🚀 Statistical Arbitrage Backtrader 集成测试")
    print("=" * 80)

    # 配置参数
    coin1, coin2 = 'MANA', 'SAND'
    # coin1, coin2 = 'BTC', 'ETH'
    start_date = '2025-02-01'
    # start_date = '2023-01-01'

    # 策略参数
    strategy_params = {
        'kalman_obs_cov': None,      # None表示自动校准
        'kalman_trans_cov': None,    # None表示自动校准
        'kalman_delta': 1e-2,        # β漂移速度控制
        'entry_sigma': 2.0,          # 入场标准差倍数
        'exit_sigma': 0.0,           # 出场标准差倍数
        'stop_loss_sigma': 4.0,      # 止损标准差倍数
        'position_size': 0.1,        # 10%资金
        'beta_volatility_threshold': 0.1,  # beta变化波动率阈值 (归一化，相对于beta均值)
        'beta_lookback_period': 20,          # beta波动率计算回望期
        'debug': False,
    }

    try:
        # 创建回测实例
        stat_arb_kalman_bt = StatArbBacktraderKalman(coin1=coin1, coin2=coin2, interval="5m")

        # 运行回测
        stat_arb_kalman_bt.run_backtest(
            start_date=start_date,
            initial_cash=10000,
            # commission=0,  # 暂时设为0以观察纯策略效果
            commission=0.0003,
            strategy_params=strategy_params
        )

        # 分析结果
        stat_arb_kalman_bt.print_analysis()

        print("\n✅ 集成测试完成!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

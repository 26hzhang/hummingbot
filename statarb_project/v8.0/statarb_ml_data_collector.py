"""
Statistical Arbitrage ML Data Collector

严格遵循延时原则：使用t-1时刻及之前的数据预测t时刻入场后的盈利率
"""
import sys
import json
from pathlib import Path
from tqdm import tqdm

from ..statarb_backtrader_kalman import (
    DataLoader, AssetDataFeed, KalmanFilterWrapper,
    StatArbKalmanStrategy as BaseStrategy,
    StatArbBacktraderKalman as BaseBacktrader
)


class StatArbKalmanMLStrategy(BaseStrategy):
    
    # 继承父类参数并添加新参数
    params = (
        # 继承父类所有参数
        ('coin1', 'ASSET1'),     
        ('coin2', 'ASSET2'),     
        ('kalman_obs_cov', None),     
        ('kalman_trans_cov', None),   
        ('kalman_delta', 1e-2),       
        ('entry_sigma', 2.0),         
        ('exit_sigma', 0.0),          
        ('stop_loss_sigma', 3.0),     
        ('position_size', 0.1),  
        ('max_positions', 1),    
        ('beta_volatility_threshold', 0.05),  
        ('beta_lookback_period', 20),          
        ('debug', False),
        
        ('enable_ml_data_collection', True),   # 是否启用ML数据收集
        ('ml_lookback_periods', 50),           # ML特征历史回望期
    )

    def _enter_spread_position(self, signal, zscore):
        """
        扩展父类方法，增加ML特征收集
        """
        # 调用父类原有逻辑
        super()._enter_spread_position(signal, zscore)
        
        if self.params.enable_ml_data_collection and self.current_pair_trade is not None:
            entry_features = self._extract_entry_features_delayed()
            if entry_features is not None:
                entry_features['signal_type'] = signal
                entry_features['signal_strength'] = abs(zscore)
                self.current_pair_trade['entry_features_delayed'] = entry_features
                self.current_pair_trade['feature_extraction_valid'] = True
                self.current_pair_trade['feature_delay_confirmed'] = True

    def _extract_entry_features_delayed(self):
        """
        延时特征提取：只保存基础的历史价格、成交量、alpha、beta信息
        严格使用t-1时刻及之前的数据，确保不包含当前入场K线信息
        """
        # 确保有足够的历史数据（至少需要2个时刻：t-1和更早）
        if len(self.zscore_history) < 2:
            return None
        
        # 定义历史回望窗口
        max_lookback = self.params.ml_lookback_periods
        current_len = len(self.zscore_history)
        # 关键：actual_lookback基于t-1时刻，不包含当前t时刻
        actual_lookback = min(max_lookback, current_len - 1)
        
        # 构建基础历史信息字典
        basic_features = {
            # 1. 基础价格数据历史 (OHLCV) - 使用t-1及之前的数据
            'asset1_open_history': self._extract_price_history(self.asset1_data.open, actual_lookback),
            'asset1_high_history': self._extract_price_history(self.asset1_data.high, actual_lookback),
            'asset1_low_history': self._extract_price_history(self.asset1_data.low, actual_lookback),
            'asset1_close_history': self._extract_price_history(self.asset1_close, actual_lookback),
            'asset1_volume_history': self._extract_price_history(self.asset1_data.volume, actual_lookback),
            
            'asset2_open_history': self._extract_price_history(self.asset2_data.open, actual_lookback),
            'asset2_high_history': self._extract_price_history(self.asset2_data.high, actual_lookback),
            'asset2_low_history': self._extract_price_history(self.asset2_data.low, actual_lookback),
            'asset2_close_history': self._extract_price_history(self.asset2_close, actual_lookback),
            'asset2_volume_history': self._extract_price_history(self.asset2_data.volume, actual_lookback),
            
            # 2. Kalman参数历史 (alpha, beta, zscore) - 使用t-1及之前的数据
            # 这里有一个小的问题，alpha和beta其实是不需要使用t-1及之前的数据的，因为这里的alpha和beta不会造成数据泄露
            # 但是为了方便逻辑暂时先这样
            'zscore_history': self._extract_list_history(self.zscore_history, actual_lookback),
            'alpha_history': self._extract_list_history(self.alpha_history, actual_lookback),
            'beta_history': self._extract_list_history(self.beta_history, actual_lookback),

            # 3. 时间信息 - 使用t-1及之前的时间点
            'datetime_history': [dt.isoformat() for dt in self._extract_list_history(self.datetime_history, actual_lookback)],
            
            # 4. 当前入场信号信息 (t时刻) - 这些是决策时刻的信息，不是特征
            'current_zscore': self.zscore_history[-1],  # t时刻的z-score（用于确认信号）
            'current_alpha': self.current_alpha,         # t时刻的alpha
            'current_beta': self.current_beta,           # t时刻的beta
            'current_asset1_price': float(self.asset1_close[0]),
            'current_asset2_price': float(self.asset2_close[0]),
            'signal_type': None,  # 将在调用处填入
            'signal_strength': None,  # 将在调用处填入
            
            # 5. 元数据信息
            'metadata': {
                'lookback_periods': actual_lookback,
                'total_history_length': current_len,
                'extraction_timestamp': self.data.datetime.datetime(0).isoformat(),
                'coin1': self.coin1,
                'coin2': self.coin2,
                'warmup_completed': (current_len > self.warmup_period),
                'feature_delay_confirmed': True,  # 确认使用了延时特征
                'strategy_params': {
                    'entry_sigma': self.params.entry_sigma,
                    'exit_sigma': self.params.exit_sigma,
                    'stop_loss_sigma': self.params.stop_loss_sigma,
                    'position_size': self.params.position_size,
                    'beta_volatility_threshold': self.params.beta_volatility_threshold,
                    'beta_lookback_period': self.params.beta_lookback_period,
                }
            },
            
            # 6. 预留目标变量字段
            'target_variables': {
                'profit_percentage': None,  # 实际盈利率（交易完成后填入）
                'trade_duration_minutes': None,
                'exit_reason': None,
            }
        }
        
        return basic_features

    def _extract_price_history(self, price_series, lookback):
        """提取价格历史数据，确保不包含当前时刻数据
        backtrader索引逻辑: [0]=t0(当前), [-1]=t-1, [-2]=t-2, [-3]=t-3, ...
        """
        series_len = len(price_series)
        if series_len < 2:  # 至少需要当前和上一个时刻
            return []
        
        if series_len - 1 < lookback:
            # 如果历史数据不足，返回所有历史数据（除了当前时刻[0]）
            # 从[-1]=t-1, [-2]=t-2, ..., [-(series_len-1)]=t-(series_len-1)
            return [float(price_series[-i]) for i in range(1, series_len)][::-1]
        else:
            # 返回指定长度的历史数据：[-1]=t-1, [-2]=t-2, ..., [-lookback]=t-lookback
            return [float(price_series[-i]) for i in range(1, lookback + 1)][::-1]

    def _extract_list_history(self, data_list, lookback):
        """提取列表类型历史数据，确保不包含当前时刻数据"""
        list_len = len(data_list)
        if list_len < 2:  # 至少需要当前和上一个时刻
            return []
        
        if list_len - 1 < lookback:
            # 如果历史数据不足，返回所有可用数据（除了当前时刻）
            return data_list[:-1]
        else:
            # 返回指定长度的历史数据，排除当前时刻
            return data_list[-(lookback + 1):-1]


class StatArbBacktraderMLCollector(BaseBacktrader):

    def run_backtest(self, start_date='2025-08-12', end_date=None, initial_cash=100000, commission=0.001, strategy_params=None):
        """
        运行回测，使用扩展的ML策略
        """
        # 复用父类数据准备逻辑
        asset1_data, asset2_data = self.prepare_data(start_date, end_date)

        # 创建Cerebro引擎（复用父类逻辑）
        import backtrader as bt
        import backtrader.analyzers as btanalyzers
        
        self.cerebro = bt.Cerebro()
        self.initial_cash = initial_cash
        self.cerebro.broker.set_cash(initial_cash)
        self.cerebro.broker.setcommission(commission=commission)

        # 添加双数据源
        asset1_feed = AssetDataFeed(dataname=asset1_data, name=f'{self.coin1}')
        asset2_feed = AssetDataFeed(dataname=asset2_data, name=f'{self.coin2}')
        self.cerebro.adddata(asset1_feed)
        self.cerebro.adddata(asset2_feed)

        # 使用扩展的ML策略替代原策略
        if strategy_params is None:
            strategy_params = {}
        strategy_params.update({
            'coin1': self.coin1,
            'coin2': self.coin2
        })

        self.cerebro.addstrategy(StatArbKalmanMLStrategy, **strategy_params)

        # 添加分析器（复用父类配置）
        self.cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(btanalyzers.TimeReturn, _name='time_return')

        print(f"开始ML数据收集回测...")
        print(f"Asset1: {self.coin1}, Asset2: {self.coin2}")
        print(f"初始资金: ${initial_cash:,.2f}")

        # 运行回测
        self.results = self.cerebro.run()
        final_value = self.cerebro.broker.getvalue()
        
        print(f"回测完成! 最终价值: ${final_value:,.2f}")
        print(f"总收益: {(final_value - initial_cash) / initial_cash:.2%}")

        return self.results

    def save_ml_training_data(self, filename):
        """
        保存ML训练数据，基于回测结果
        """
        if self.results is None:
            print("请先运行回测")
            return None
        
        strategy_instance = self.results[0]
        ml_samples = []
        
        for trade in strategy_instance.pair_trades:
            if ('entry_features_delayed' in trade and 
                trade.get('feature_extraction_valid', False) and
                trade.get('feature_delay_confirmed', False)):
                
                # 获取基础特征
                features = trade['entry_features_delayed'].copy()
                
                # 填入目标变量
                if 'analysis' in trade:
                    pnl_analysis = trade['analysis']['position_pnl_analysis']
                    features['target_variables']['profit_percentage'] = pnl_analysis['profit_percentage']
                    
                    # 计算交易持续时间
                    if 'entry_time' in trade and 'exit_time' in trade:
                        duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
                        features['target_variables']['trade_duration_minutes'] = duration
                    
                    features['target_variables']['exit_reason'] = trade['exit_reason']
                
                # 添加交易基本信息
                features['trade_info'] = {
                    'pair': f"{self.coin1}-{self.coin2}",
                    'entry_time': trade['entry_time'].isoformat(),
                    'exit_time': trade['exit_time'].isoformat() if 'exit_time' in trade else None,
                    'signal_type': trade['signal'],
                    'entry_zscore': trade['entry_zscore'],
                }
                
                ml_samples.append(features)
        
        # 保存为JSON格式
        with open(filename, 'w') as f:
            json.dump(ml_samples, f, indent=2, default=str)
        
        print(f"✅ ML训练数据已保存: {filename}")
        print(f"📊 {self.coin1}-{self.coin2} 样本数: {len(ml_samples)}")
        
        return ml_samples


class MultiPairMLDataCollector:
    """
    多交易对ML数据收集器
    """
    
    def __init__(self, data_dir="/storage/tianzichen/sicheng/hummingbot/data/", interval="5m"):
        self.data_dir = data_dir
        self.interval = interval
        self.all_samples = []
    
    def collect_multi_pair_data(self, pair_list, start_date='2023-01-01', end_date='2024-12-31', 
                               strategy_params=None):
        """
        批量收集多交易对ML训练数据
        """
        if strategy_params is None:
            strategy_params = {
                'entry_sigma': 1.5,
                'exit_sigma': 0.0,
                'stop_loss_sigma': 4.0,
                'position_size': 0.1,
                'enable_ml_data_collection': True,
                'ml_lookback_periods': 50,
                'debug': False,
            }
        
        for i, (coin1, coin2) in enumerate(tqdm(pair_list, desc="处理交易对", unit="pair")):
            print(f"\n🔄 处理交易对 [{i+1}/{len(pair_list)}]: {coin1}-{coin2}")
            
            try:
                # 使用扩展的ML收集器
                stat_arb_bt = StatArbBacktraderMLCollector(coin1=coin1, coin2=coin2, data_dir=self.data_dir, interval=self.interval)
                
                # 运行回测
                results = stat_arb_bt.run_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=10000,
                    commission=0,
                    strategy_params=strategy_params
                )
                
                # 保存该交易对的ML数据
                pair_filename = f"/Users/zhanghao/GitHub/hummingbot/data/futures_5m_ml_data/ml_data_{coin1}_{coin2}.json"
                pair_samples = stat_arb_bt.save_ml_training_data(pair_filename)
                
                if pair_samples:
                    self.all_samples.extend(pair_samples)
                    print(f"✅ {coin1}-{coin2} 完成，收集到 {len(pair_samples)} 个样本")
                else:
                    print(f"⚠️ {coin1}-{coin2} 未收集到有效样本")
                    
            except Exception as e:
                print(f"❌ {coin1}-{coin2} 处理失败: {e}")
                continue
        
        # 合并保存所有数据
        if self.all_samples:
            combined_filename = f"/Users/zhanghao/GitHub/hummingbot/data/futures_5m_ml_data/all_pairs_ml_training_data.json"
            with open(combined_filename, 'w') as f:
                json.dump(self.all_samples, f, indent=2, default=str)
            
            print(f"\n✅ 所有交易对ML数据已合并保存: {combined_filename}")
            print(f"📊 总样本数: {len(self.all_samples)}")
        
        return self.all_samples


def collect_ml_training_data():
    print("=" * 80)
    print("🚀 Statistical Arbitrage ML数据收集")
    print("=" * 80)
    
    pair_list_dir = '/Users/zhanghao/GitHub/hummingbot/data/futures_5m_ml_data/pair_list.txt'
    with open(pair_list_dir, 'r') as f:
        raw_pair_list = f.readlines()
    
    # 解析并过滤交易对，排除包含MANA和SAND的pair
    pair_list = []
    for line in raw_pair_list:
        pair_str = line.strip()
        if pair_str and '-' in pair_str:
            asset1, asset2 = pair_str.split('-')
            # 排除包含MANA或SAND的交易对
            if 'MANA' not in [asset1, asset2] and 'SAND' not in [asset1, asset2]:
                pair_list.append((asset1, asset2))
    
    print(f"从{len(raw_pair_list)}个交易对中过滤出{len(pair_list)}个有效交易对（已排除MANA/SAND相关对）")
    
    # 策略参数
    strategy_params = {
        'entry_sigma': 1.5,
        'exit_sigma': 0.0,
        'stop_loss_sigma': 4.0,
        'position_size': 0.1,
        'enable_ml_data_collection': True,
        'ml_lookback_periods': 64,
        'debug': False,
    }
    
    try:
        # 创建多交易对收集器
        collector = MultiPairMLDataCollector(interval="5m")
        
        # 批量收集数据
        all_samples = collector.collect_multi_pair_data(
            pair_list=pair_list,
            start_date='2023-01-01',
            end_date='2024-12-31',
            strategy_params=strategy_params
        )
        
        print(f"\n🎉 ML数据收集完成!")
        print(f"📊 总样本数: {len(all_samples)}")
        
    except Exception as e:
        print(f"❌ ML数据收集失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    collect_ml_training_data()

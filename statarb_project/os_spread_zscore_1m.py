"""
# OS Spread Z-Score 可视化分析 - 纯1分钟数据版本

基于现有 pair_selection_analysis.py 逻辑，使用**纯1分钟数据**可视化给定价格对的 OS spread z-score 变化。

**与双时间窗口版本的区别**:
- **单一数据源**: 仅使用1分钟K线数据
- **统一时间粒度**: IS和OS阶段都使用相同的1分钟间隔
- **架构**: 无需处理时间对齐和数据转换问题
- **直接对应**: 更接近stat_arb控制器实际使用的数据结构
"""
# 禁用所有matplotlib字体警告
import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from statsmodels.tsa.stattools import adfuller

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 简单设置字体，避免警告
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class DataLoader:

    def __init__(self, data_dir="/Users/zhanghao/GitHub/hummingbot/data/futures_1m"):
        self.data_dir = Path(data_dir)
        self.pairs = [f.stem.replace("_1m", "") for f in self.data_dir.glob("*_1m.csv")]
        # print(f"Found {len(self.pairs)} pairs (1m data)")
        # print(f"Available pairs: {', '.join(self.pairs[:10])}{'...' if len(self.pairs) > 10 else ''}")

    def load_pair_data(self, coin1, coin2, start_date=None, end_date=None, silent=False):
        pair1 = f"{coin1}USDT"
        pair2 = f"{coin2}USDT"

        # 检查文件存在性
        file1 = self.data_dir / f"{pair1}_1m.csv"
        file2 = self.data_dir / f"{pair2}_1m.csv"

        missing_files = []
        if not file1.exists():
            missing_files.append(f"{pair1}_1m.csv")
        if not file2.exists():
            missing_files.append(f"{pair2}_1m.csv")

        if missing_files:
            if not silent:
                print(f"Files not found: {missing_files}")
            return None

        try:
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

            if not silent:
                print(f"1m数据加载完成 {coin1}-{coin2}")
                print(f"数据量: {len(data)} 条")
                print(f"时间范围: {common_start} 至 {common_end}")
                print(f"时间跨度: {(common_end - common_start).days} 天")

            return data

        except Exception as e:
            if not silent:
                print(f"数据加载错误: {str(e)}")
            return None


class RollingWindowAnalyzer:
    """滑动窗口分析器 - 直接的统计套利分析"""

    def __init__(
        self, window_size=2000, step_size=500, lookback_zscore=1000, ma_window=None, ma_type='sma', reg_method=None):
        """
        参数:
        window_size: OLS拟合的滑动窗口大小（数据点数）
        step_size: 每次前进的步长（数据点数）
        lookback_zscore: 计算z-score基准时使用的历史spread数量
        ma_window: int, MA平滑窗口大小，None表示不启用OLS平滑
        ma_type: str, MA类型 ('sma'=简单移动平均, 'ema'=指数移动平均)
        reg_method: str, 回归方法 (None='普通OLS', 'huber'='Huber回归')
        """
        self.window_size = window_size
        self.step_size = step_size
        self.lookback_zscore = lookback_zscore
        self.ma_window = ma_window
        self.ma_type = ma_type
        self.reg_method = reg_method if reg_method is not None else 'ols'

        # 存储分析结果
        self.analysis_results = None
        self.data = None
        self.coin1 = None
        self.coin2 = None

    def analyze_pair(self, data, coin1, coin2, price_transform='log'):
        """滑动窗口分析 - 直接计算当前z-score"""

        # 存储数据用于后续分析
        self.data = data
        self.coin1 = coin1
        self.coin2 = coin2

        # 验证数据长度
        min_required = self.window_size + self.lookback_zscore + self.step_size
        if len(data) < min_required:
            print(f"数据不足: 需要至少{min_required}条数据")
            print(f"实际: {len(data)}条")
            return None

        # 提取价格数据
        if price_transform == 'log':
            price1 = np.log(data[f'{coin1}_close'])
            price2 = np.log(data[f'{coin2}_close'])
        elif price_transform in ['raw', 'ratio']:
            price1 = data[f'{coin1}_close']
            price2 = data[f'{coin2}_close']
        else:
            raise ValueError(f"Unsupported price_transform: {price_transform}")

        # 存储结果
        results = []
        historical_spreads = []  # 严格的历史spread记录

        print(f"滑动窗口分析 {coin1}-{coin2}")
        print(f"窗口大小: {self.window_size}, 步长: {self.step_size}, Z-Score回看: {self.lookback_zscore}")
        if price_transform == 'ratio':
            print(f"💹 价格变换: {price_transform.upper()} 方式 (基于窗口第一个数据点)")
        else:
            print(f"💹 价格变换: {price_transform.upper()} 方式")
        if self.ma_window is not None:
            print(f"🔧 OLS参数平滑: {self.ma_type.upper()} 窗口={self.ma_window}")
        if self.reg_method is not None:
            print(f"🛡️ 鲁棒回归: {self.reg_method.upper()} 方法")
        print("✅ 逻辑: 基于历史数据直接计算当前z-score")

        # 滑动窗口分析
        for i in range(self.window_size, len(data), self.step_size):
            # 检查是否还有足够的数据
            if i + self.step_size > len(data):
                break

            # 1. 提取窗口数据进行OLS拟合（基于历史数据）
            window_start = i - self.window_size
            window_end = i
            window_p1 = price1.iloc[window_start:window_end]
            window_p2 = price2.iloc[window_start:window_end]

            # 如果是ratio模式，基于窗口第一个数据点计算比率
            if price_transform == 'ratio':
                base_price1 = window_p1.iloc[0]
                base_price2 = window_p2.iloc[0]
                window_p1 = window_p1 / base_price1
                window_p2 = window_p2 / base_price2

            # OLS拟合
            cointegration = self._fit_algo(window_p1, window_p2)
            if cointegration is None:
                continue

            # 2. 基于历史spread建立z-score基准
            if len(historical_spreads) >= self.lookback_zscore:
                zscore_baseline_spreads = historical_spreads[-self.lookback_zscore:-1]
            else:
                zscore_baseline_spreads = historical_spreads.copy()

            # 计算历史基准统计量
            if len(zscore_baseline_spreads) > 0:
                baseline_mean = np.mean(zscore_baseline_spreads)
                baseline_std = np.std(zscore_baseline_spreads)
                if baseline_std == 0:
                    baseline_std = 1.0  # 避免除零
            else:
                baseline_mean = 0.0
                baseline_std = 1.0

            # 3. 逐点处理下一个step_size区间
            for j in range(self.step_size):
                current_position = i + j
                if current_position >= len(data):
                    break

                # 获取当前时点的价格
                current_p1 = price1.iloc[current_position]
                current_p2 = price2.iloc[current_position]
                current_timestamp = data['timestamp'].iloc[current_position]

                # 4. 计算当前spread和z-score
                # 基于OLS参数计算当前spread
                current_spread = current_p2 - cointegration['alpha'] - cointegration['beta'] * current_p1

                # 基于历史基准计算当前z-score
                current_zscore = (current_spread - baseline_mean) / baseline_std

                # 5. 存储当前点的结果
                results.append({
                    'timestamp': current_timestamp,
                    'window_start_idx': window_start,
                    'window_end_idx': window_end,
                    'position': current_position,
                    'alpha': cointegration['alpha'],
                    'beta': cointegration['beta'],
                    'r_squared': cointegration['r_squared'],
                    'adf_pvalue': cointegration['pvalue'],
                    'current_spread': current_spread,
                    'current_zscore': current_zscore,
                    'baseline_mean': baseline_mean,
                    'baseline_std': baseline_std,
                    'lookback_count': len(zscore_baseline_spreads)
                })

                # 6. **关键**: 将当前spread加入历史记录（防数据泄露）
                historical_spreads.append(current_spread)

        if len(results) == 0:
            print("没有找到有效的分析结果")
            return None

        # 整理分析结果
        self.analysis_results = {
            'results': results,
            'timestamps': [r['timestamp'] for r in results],
            'current_spreads': [r['current_spread'] for r in results],
            'current_zscores': [r['current_zscore'] for r in results],
            'alphas': [r['alpha'] for r in results],
            'betas': [r['beta'] for r in results],
            'r_squared_values': [r['r_squared'] for r in results],
            'adf_pvalues': [r['adf_pvalue'] for r in results]
        }

        print(f"成功分析{len(results)}个数据点")
        print(f"当前z-score: ✅ 基于历史统计计算当前spread的异常程度")

        # 生成可视化
        # self._plot_analysis()

        # 打印统计摘要
        self._print_summary()

        return self.analysis_results

    def _fit_algo(self, p1, p2):
        """拟合协整关系 - 内部进行价格平滑"""
        try:
            # 仅在OLS计算时进行价格平滑
            if self.ma_window is not None and self.ma_window > 1:
                if self.ma_type == 'sma':
                    p1_for_fit = p1.rolling(window=self.ma_window, min_periods=1).mean()
                    p2_for_fit = p2.rolling(window=self.ma_window, min_periods=1).mean()
                elif self.ma_type == 'ema':
                    p1_for_fit = p1.ewm(span=self.ma_window).mean()
                    p2_for_fit = p2.ewm(span=self.ma_window).mean()
                else:
                    p1_for_fit = p1
                    p2_for_fit = p2
            else:
                p1_for_fit = p1
                p2_for_fit = p2

            # 使用平滑后的价格进行线性回归
            X = p1_for_fit.values.reshape(-1, 1)
            y = p2_for_fit.values

            # 根据robust_method选择回归器
            if self.reg_method == 'huber':
                reg = HuberRegressor(epsilon=1.5).fit(X, y)
            else:
                # 默认使用普通OLS
                reg = LinearRegression().fit(X, y)

            alpha = reg.intercept_
            beta = reg.coef_[0]

            # 计算R-squared（使用平滑价格）
            y_pred = reg.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # ADF检验（使用平滑价格计算的spread）
            spread = y - alpha - beta * p1_for_fit.values
            _, pvalue, _, _, _, _ = adfuller(spread, regression="n", autolag='AIC')

            return {
                'alpha': alpha,
                'beta': beta,
                'r_squared': r_squared,
                'pvalue': pvalue
            }
        except:
            return None

    def _plot_analysis(self):
        """绘制分析结果"""
        if self.analysis_results is None:
            return

        timestamps = self.analysis_results['timestamps']
        current_zscores = self.analysis_results['current_zscores']
        current_spreads = self.analysis_results['current_spreads']
        alphas = self.analysis_results['alphas']
        betas = self.analysis_results['betas']

        # 第一步：使用mplfinance绘制蜡烛图
        # 从原始数据中提取对应时间段的价格
        analysis_start_time = timestamps[0]
        analysis_end_time = timestamps[-1]
        # 过滤原始数据到分析时间范围
        price_data = self.data[
            (self.data['timestamp'] >= analysis_start_time) &
            (self.data['timestamp'] <= analysis_end_time)
        ].copy()

        # 准备蜡烛图数据格式
        ohlc_data1 = price_data[['timestamp', f'{self.coin1}_open', f'{self.coin1}_high',
                                f'{self.coin1}_low', f'{self.coin1}_close', f'{self.coin1}_volume']].copy()
        ohlc_data1.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        ohlc_data1.set_index('Date', inplace=True)

        ohlc_data2 = price_data[['timestamp', f'{self.coin2}_open', f'{self.coin2}_high',
                                f'{self.coin2}_low', f'{self.coin2}_close', f'{self.coin2}_volume']].copy()
        ohlc_data2.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        ohlc_data2.set_index('Date', inplace=True)

        # 绘制蜡烛图 - 上下两个子图
        print("📊 显示蜡烛图...")
        # 使用mplfinance创建包含两个币种的组合蜡烛图
        # 由于mplfinance外部axes的复杂性，我们使用单独的图但紧密排列

        # 第一个蜡烛图
        mpf.plot(ohlc_data1, type='candle', volume=True, title=f'{self.coin1} Candlestick', style='charles', figsize=(15, 6))

        # 第二个蜡烛图
        mpf.plot(ohlc_data2, type='candle', volume=True, title=f'{self.coin2} Candlestick', style='charles', figsize=(15, 6))

        # 第二步：创建分析图表
        print("📈 显示分析图表...")
        _, axes = plt.subplots(4, 1, figsize=(20, 24))

        # 第一图：Z-Score时间序列（交易决策信号）
        axes[0].plot(timestamps, current_zscores, 'b-', alpha=0.8, linewidth=1.0)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].axhline(y=1, color='orange', linestyle='--', alpha=0.7, label=r'±1$\sigma$')
        axes[0].axhline(y=-1, color='orange', linestyle='--', alpha=0.7)
        axes[0].axhline(y=2, color='red', linestyle='--', alpha=0.7, label=r'±2$\sigma$')
        axes[0].axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        axes[0].axhline(y=3, color='darkred', linestyle=':', alpha=0.7, label=r'±3$\sigma$')
        axes[0].axhline(y=-3, color='darkred', linestyle=':', alpha=0.7)
        axes[0].set_ylabel('Current Z-Score')
        axes[0].set_ylim(-5, 5)
        axes[0].set_title(f'{self.coin1}-{self.coin2} Current Z-Score (Trading Decision Signal)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 第二图：Spread时间序列
        axes[1].plot(timestamps, current_spreads, 'g-', alpha=0.7, linewidth=1.0)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('Current Spread')
        axes[1].set_title('Current Spread Time Series')
        axes[1].grid(True, alpha=0.3)

        # 第三图：动态Alpha和Beta参数
        ax3_alpha = axes[2]
        ax3_beta = ax3_alpha.twinx()
        line1 = ax3_alpha.plot(timestamps, alphas, 'g-', alpha=0.7, linewidth=1.0, label='Alpha')
        line2 = ax3_beta.plot(timestamps, betas, 'r-', alpha=0.7, linewidth=1.0, label='Beta')
        ax3_alpha.set_ylabel('Alpha', color='g')
        ax3_beta.set_ylabel('Beta', color='r')
        ax3_alpha.set_title('Rolling OLS Parameters')
        ax3_alpha.grid(True, alpha=0.3)

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3_alpha.legend(lines, labels, loc='upper left')

        # 第四图：Z-Score分布
        axes[3].hist(current_zscores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[3].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[3].axvline(x=1, color='orange', linestyle='--', alpha=0.7)
        axes[3].axvline(x=-1, color='orange', linestyle='--', alpha=0.7)
        axes[3].axvline(x=2, color='red', linestyle='--', alpha=0.7)
        axes[3].axvline(x=-2, color='red', linestyle='--', alpha=0.7)

        axes[3].set_xlabel('Current Z-Score')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Current Z-Score Distribution (Trading Decision Basis)')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _print_summary(self):
        """打印分析结果摘要"""
        if self.analysis_results is None:
            return

        results = self.analysis_results['results']
        current_zscores = self.analysis_results['current_zscores']
        alphas = self.analysis_results['alphas']
        betas = self.analysis_results['betas']

        print(f"\n=== {self.coin1}-{self.coin2} 滑动窗口分析摘要 ===")
        print(f"分析参数: 窗口大小={self.window_size}, 步长={self.step_size}, Z-Score回看={self.lookback_zscore}")
        if self.ma_window is not None:
            print(f"窗口数据平滑参数: {self.ma_type.upper()} 窗口={self.ma_window}")
        print(f"回归模型: {self.reg_method.upper()}")
        print(f"总数据点: {len(results)}")

        # 数据泄露检查
        lookback_counts = [r['lookback_count'] for r in results]
        max_lookback = max(lookback_counts) if lookback_counts else 0
        leakage_check = max_lookback <= self.lookback_zscore
        print(f"\n🔒 数据泄露检查:")
        print(f"最大历史窗口: {max_lookback}, 设定上限: {self.lookback_zscore}")
        print(f"泄露检查结果: {'✅ 通过' if leakage_check else '❌ 失败'}")

        # Z-Score统计
        print(f"\n📊 Z-Score统计 (交易决策信号):")
        print(f"范围: {np.min(current_zscores):.2f} 至 {np.max(current_zscores):.2f}")
        print(f"均值: {np.mean(current_zscores):.3f}, 标准差: {np.std(current_zscores):.3f}")
        print(f"|Z|>1 信号比例: {np.mean(np.abs(current_zscores) > 1):.2%}")
        print(f"|Z|>2 强信号比例: {np.mean(np.abs(current_zscores) > 2):.2%}")
        print(f"|Z|>3 极端信号比例: {np.mean(np.abs(current_zscores) > 3):.2%}")

        # 参数稳定性
        alpha_volatility = np.std(alphas)
        beta_volatility = np.std(betas)
        alpha_change = (max(alphas) - min(alphas)) / abs(np.mean(alphas)) * 100
        beta_change = (max(betas) - min(betas)) / abs(np.mean(betas)) * 100

        print(f"\n🎛️ 参数稳定性:")
        print(f"Alpha 波动率: {alpha_volatility:.6f}, 变化幅度: {alpha_change:.2f}%")
        print(f"Beta 波动率: {beta_volatility:.6f}, 变化幅度: {beta_change:.2f}%")

        # 协整质量
        r_squared_values = self.analysis_results['r_squared_values']
        adf_pvalues = self.analysis_results['adf_pvalues']
        adf_pass_rate = np.mean([p < 0.05 for p in adf_pvalues])

        print(f"\n📈 协整质量:")
        print(f"平均R²: {np.mean(r_squared_values):.4f}")
        print(f"ADF检验通过率: {adf_pass_rate:.2%}")


def main():
    loader = DataLoader()

    # 滑动窗口分析 - 支持MA平滑功能
    coin1, coin2 = 'MANA', 'SAND'  # 可以修改这里选择不同的交易对

    print(f"=== 滑动窗口分析 {coin1}-{coin2} ===")

    # MA平滑参数配置
    ma_window = 3       # None=不启用平滑, 整数=平滑窗口大小
    ma_type = 'sma'        # 'sma'=简单移动平均, 'ema'=指数移动平均

    # 鰁棒回归参数配置
    reg_method = 'ols'  # None or 'ols'=普通OLS, 'huber'=Huber回归

    # 价格变换参数配置
    price_transform = 'log'  # 'log'=对数变换, 'raw'=原始价格, 'ratio'=比率变换(基于窗口第一个数据点)

    # 加载1分钟数据
    start_date = '2025-08-12'
    data = loader.load_pair_data(coin1, coin2, start_date=start_date)
    print(f'分析时间跨度，起始时间{start_date}')
    print(f"\n数据验证:")
    print(f"1m数据长度: {len(data)}")

    # 创建分析器
    rw_analyzer = RollingWindowAnalyzer(
        window_size=100,           # OLS拟合窗口
        step_size=10,              # 参数更新频率
        lookback_zscore=100,       # z-score历史基准
        ma_window=ma_window,       # MA平滑窗口
        ma_type=ma_type,           # MA类型
        reg_method=reg_method  # 鲁棒回归方法
    )

    min_required = rw_analyzer.window_size + rw_analyzer.lookback_zscore + rw_analyzer.step_size
    print(f"最小数据需求: {min_required}条 ({min_required / 1440:.1f}天)")

    print("\n✅ 开始分析...")
    print("🎯 直接计算当前z-score，无预测复杂性")
    result = rw_analyzer.analyze_pair(data, coin1, coin2, price_transform=price_transform)
    print("\n🎉 分析完成！")

    # 分析结果
    current_zscores = result['current_zscores']
    alphas = result['alphas']
    betas = result['betas']

    print(f"\n📊 当前z-score信号分析:")
    print(f"总信号数: {len(current_zscores)}")
    print(f"信号范围: {np.min(current_zscores):.2f} 至 {np.max(current_zscores):.2f}")

    # 交易信号统计
    weak_signals = np.mean(np.abs(current_zscores) > 1)
    medium_signals = np.mean(np.abs(current_zscores) > 2)
    strong_signals = np.mean(np.abs(current_zscores) > 3)

    print(f"\n🎯 交易信号分布:")
    print(f"  弱信号 (|Z|>1): {weak_signals:.1%} - 考虑建仓")
    print(f"  中等信号 (|Z|>2): {medium_signals:.1%} - 建议建仓")
    print(f"  强信号 (|Z|>3): {strong_signals:.1%} - 强烈建仓")

    # 参数稳定性
    alpha_stability = np.std(alphas) / abs(np.mean(alphas)) * 100
    beta_stability = np.std(betas) / abs(np.mean(betas)) * 100

    print(f"\n🎛️ 参数稳定性:")
    print(f"Alpha相对波动: {alpha_stability:.2f}%")
    print(f"Beta相对波动: {beta_stability:.2f}%")

    # 实际交易建议
    print(f"\n💡 实际交易建议:")
    if medium_signals > 0.1:
        print("✅ 该配对有足够的中等强度信号，适合交易")
        print(f"预期每{len(current_zscores) / medium_signals / len(current_zscores) * 500:.0f}分钟有一次中等信号")
    else:
        print("⚠️  该配对信号较弱，可能不适合当前参数设置")

    if alpha_stability < 5 and beta_stability < 5:
        print("✅ 协整参数稳定，关系可靠")
    else:
        print("⚠️  协整参数波动较大，需要谨慎交易")

    print(f"\n🔧 Hummingbot配置建议:")
    avg_zscore_std = np.std(current_zscores)
    print(f"entry_threshold: {avg_zscore_std * 1.5:.2f}")
    print(f"lookback_period: {rw_analyzer.window_size}")


if __name__ == "__main__":
    main()

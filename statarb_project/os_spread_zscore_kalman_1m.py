import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arbitragelab.other_approaches.kalman_filter import KalmanFilterStrategy
from sklearn.linear_model import LinearRegression

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 简单设置字体，避免警告
matplotlib.rcParams['axes.unicode_minus'] = False
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


class KalmanFilterAnalyzer:

    def __init__(
        self, lookback_zscore=1000, kalman_obs_cov=None, kalman_trans_cov=None, delta=1e-2):
        """
        参数:
        lookback_zscore: 计算z-score基准时使用的历史spread数量
        kalman_obs_cov: float or None, Kalman滤波观测协方差（None表示自动校准）
        kalman_trans_cov: float or None, Kalman滤波转换协方差（None表示自动校准）
        delta: float, 控制β漂移速度的参数，用于trans_cov校准
        """
        self.lookback_zscore = lookback_zscore
        self.kalman_obs_cov = kalman_obs_cov
        self.kalman_trans_cov = kalman_trans_cov
        self.delta = delta

        # 存储分析结果
        self.analysis_results = None
        self.data = None
        self.coin1 = None
        self.coin2 = None

        # 全局Kalman滤波器
        self.global_kalman_filter = None

    def _calibrate_parameters(self, price1_series, price2_series, n_train):
        """使用OLS残差校准Kalman参数"""
        print(f"🔧 使用{n_train}个点进行OLS参数估计")

        X = price1_series.iloc[:n_train].values.reshape(-1, 1)
        y = price2_series.iloc[:n_train].values

        ols = LinearRegression().fit(X, y)
        ols_residuals = y - ols.predict(X)
        ols_variance = np.var(ols_residuals, ddof=1)

        obs_cov = ols_variance
        trans_cov = self.delta * obs_cov / (1 - self.delta)

        return obs_cov, trans_cov

    def analyze_pair(self, data, coin1, coin2, price_transform='log'):
        # 存储数据用于后续分析
        self.data = data
        self.coin1 = coin1
        self.coin2 = coin2

        # 提取价格数据（仅使用close）
        if price_transform == 'log':
            price1_close = np.log(data[f'{coin1}_close'])
            price2_close = np.log(data[f'{coin2}_close'])
        else:
            price1_close = data[f'{coin1}_close']
            price2_close = data[f'{coin2}_close']

        # 参数校准（如果需要）并确定训练集大小
        if self.kalman_obs_cov is None or self.kalman_trans_cov is None:
            print("🔧 校准Kalman参数...")
            n_train = min(self.lookback_zscore, len(price1_close) // 3)
            obs_cov, trans_cov = self._calibrate_parameters(price1_close, price2_close, n_train)
            print(f"✅ 校准后参数: obs_cov={obs_cov:.6f}, trans_cov={trans_cov:.6f}")
        else:
            obs_cov = self.kalman_obs_cov
            trans_cov = self.kalman_trans_cov
            n_train = 0  # 预设参数时无训练集
            print(f"🔄 使用预设参数: obs_cov={obs_cov}, trans_cov={trans_cov}")

        # 初始化全局Kalman滤波器（使用校准或预设参数）
        self.global_kalman_filter = KalmanFilterStrategy(
            observation_covariance=obs_cov,
            transition_covariance=trans_cov
        )
        # 静默预热，更新Kalman滤波器状态，但不产生信号
        for t in range(1, n_train + 1):
            self.global_kalman_filter.update(price1_close.iloc[t - 1], price2_close.iloc[t - 1])

        results = []

        # 确定分析起始点，避免重复使用训练数据
        start_idx = max(1, n_train + 1)

        print(f"Kalman滤波逐点分析 {coin1}-{coin2}")
        print(f"Z-Score回看: {self.lookback_zscore}")
        print("✅ 逻辑: 每个数据点增量更新，时点同步信号")

        # 逐点Kalman滤波分析
        for i in range(start_idx, len(data)):

            # 获取当前时点信息
            current_p1 = price1_close.iloc[i]
            current_p2 = price2_close.iloc[i]
            current_timestamp = data['timestamp'].iloc[i]

            # 使用当前观测更新Kalman滤波器并获取同时点信号
            self.global_kalman_filter.update(current_p1, current_p2)

            # 获取当前参数估计
            if len(self.global_kalman_filter.hedge_ratios) > 0:
                current_alpha = self.global_kalman_filter.intercepts[-1]
                current_beta = self.global_kalman_filter.hedge_ratios[-1]

                # 使用官方库的预测误差和标准差（统一信号源）
                if len(self.global_kalman_filter.spread_series) > 0:
                    prediction_error = self.global_kalman_filter.spread_series[-1]  # e_t
                    prediction_std = self.global_kalman_filter.spread_std_series[-1]  # sqrt(Q_t)

                    if prediction_std > 0:
                        current_zscore = prediction_error / prediction_std  # 官方标准化方法
                    else:
                        current_zscore = 0.0

                    # 计算traditional spread（仅用于可视化对比）
                    current_spread = current_p2 - current_alpha - current_beta * current_p1
                else:
                    continue
            else:
                continue

            # 存储当前点的结果
            results.append({
                'timestamp': current_timestamp,
                'position': i,
                'alpha': current_alpha,
                'beta': current_beta,
                'current_spread': current_spread,
                'prediction_error': prediction_error,
                'prediction_std': prediction_std,
                'current_zscore': current_zscore,
                'spread_std': prediction_std
            })

        # 整理分析结果（更新结构）
        self.analysis_results = {
            'results': results,
            'timestamps': [r['timestamp'] for r in results],
            'current_spreads': [r['current_spread'] for r in results],
            'prediction_errors': [r['prediction_error'] for r in results],
            'prediction_stds': [r['prediction_std'] for r in results],
            'current_zscores': [r['current_zscore'] for r in results],
            'alphas': [r['alpha'] for r in results],
            'betas': [r['beta'] for r in results],
            'spread_stds': [r['spread_std'] for r in results]
        }

        print(f"成功分析{len(results)}个数据点")

        # 获取官方交易信号并裁剪到与分析结果匹配
        official_signals_full = self.get_official_trading_signals(entry_std=2.0, exit_std=0.0)
        if official_signals_full is not None:
            # 裁剪信号以匹配分析结果的时间范围(从start_idx开始)
            # 确保不超出官方信号的边界
            available_length = len(official_signals_full['target_quantity']) - start_idx
            actual_length = min(len(results), available_length)
            signal_end_idx = start_idx + actual_length

            self.official_signals = {
                'errors': official_signals_full['errors'].iloc[start_idx:signal_end_idx].reset_index(drop=True),
                'target_quantity': official_signals_full['target_quantity'].iloc[start_idx:signal_end_idx].reset_index(drop=True)
            }
        else:
            self.official_signals = None

        self._plot_analysis()
        self._print_summary()

        return self.analysis_results

    def _plot_analysis(self):
        """绘制分析结果"""
        if self.analysis_results is None:
            return

        timestamps = self.analysis_results['timestamps']
        current_zscores = self.analysis_results['current_zscores']
        current_spreads = self.analysis_results['current_spreads']
        alphas = self.analysis_results['alphas']
        betas = self.analysis_results['betas']

        # 创建分析图表
        _, axes = plt.subplots(5, 1, figsize=(20, 15))

        # 第一图：Z-Score时间序列（交易决策信号）
        axes[0].plot(timestamps, current_zscores, 'b-', alpha=0.8, linewidth=1.0)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label=r'±0.5$\sigma$')
        axes[0].axhline(y=-0.5, color='orange', linestyle='--', alpha=0.7)
        axes[0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label=r'±1$\sigma$')
        axes[0].axhline(y=-1, color='red', linestyle='--', alpha=0.7)
        axes[0].axhline(y=1.5, color='darkred', linestyle=':', alpha=0.7, label=r'±1.5$\sigma$')
        axes[0].axhline(y=-1.5, color='darkred', linestyle=':', alpha=0.7)
        axes[0].set_ylabel('Current Z-Score')
        axes[0].set_ylim(-3, 3)
        axes[0].set_title(f'{self.coin1}-{self.coin2} Current Z-Score (Trading Decision Signal)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 第二图：官方交易信号
        target_quantity = self.official_signals['target_quantity']

        # 确保时间戳和信号长度匹配
        min_length = min(len(timestamps), len(target_quantity))
        timestamps_matched = timestamps[:min_length]
        target_quantity_matched = target_quantity[:min_length]

        # 绘制交易信号
        axes[1].plot(timestamps_matched, target_quantity_matched, 'purple', alpha=0.8, linewidth=1.5)
        axes[1].fill_between(timestamps_matched, target_quantity_matched, 0,
                             where=(target_quantity_matched > 0), color='green', alpha=0.3, label='Long Signal')
        axes[1].fill_between(timestamps_matched, target_quantity_matched, 0,
                             where=(target_quantity_matched < 0), color='red', alpha=0.3, label='Short Signal')

        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1].axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Long')
        axes[1].axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='Short')
        axes[1].set_ylabel('Trading Signal')
        axes[1].set_title('ArbitrageLab Trading Signal (entry_std=2.0)')
        axes[1].set_ylim(-1.2, 1.2)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 第三图：Spread时间序列
        axes[2].plot(timestamps, current_spreads, 'g-', alpha=0.7, linewidth=1.0)
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_ylabel('Current Spread')
        axes[2].set_title('Current Spread Time Series')
        axes[2].grid(True, alpha=0.3)

        # 第四图：动态Alpha和Beta参数
        ax3_alpha = axes[3]
        ax3_beta = ax3_alpha.twinx()
        line1 = ax3_alpha.plot(timestamps, alphas, 'g-', alpha=0.7, linewidth=1.0, label='Alpha')
        line2 = ax3_beta.plot(timestamps, betas, 'r-', alpha=0.7, linewidth=1.0, label='Beta')
        ax3_alpha.set_ylabel('Alpha', color='g')
        ax3_beta.set_ylabel('Beta', color='r')
        ax3_alpha.set_title('Dynamic Kalman Parameters')
        ax3_alpha.grid(True, alpha=0.3)

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3_alpha.legend(lines, labels, loc='upper left')

        # 第五图：Z-Score分布 - 过滤NaN值
        zscores_for_hist = np.array(current_zscores)
        zscores_clean = zscores_for_hist[~np.isnan(zscores_for_hist)]
        if len(zscores_clean) > 0:
            axes[4].hist(zscores_clean, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[4].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[4].axvline(x=1, color='orange', linestyle='--', alpha=0.7)
        axes[4].axvline(x=-1, color='orange', linestyle='--', alpha=0.7)
        axes[4].axvline(x=2, color='red', linestyle='--', alpha=0.7)
        axes[4].axvline(x=-2, color='red', linestyle='--', alpha=0.7)

        axes[4].set_xlabel('Current Z-Score')
        axes[4].set_ylabel('Frequency')
        axes[4].set_title('Current Z-Score Distribution (Trading Decision Basis)')
        axes[4].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _print_summary(self):
        """打印分析结果摘要"""
        results = self.analysis_results['results']
        current_zscores = np.array(self.analysis_results['current_zscores'])
        alphas = np.array(self.analysis_results['alphas'])
        betas = np.array(self.analysis_results['betas'])

        # 过滤NaN值
        valid_mask = ~np.isnan(current_zscores)
        current_zscores_clean = current_zscores[valid_mask]
        alphas_clean = alphas[valid_mask]
        betas_clean = betas[valid_mask]

        print(f"\n=== {self.coin1}-{self.coin2} Kalman滤波分析摘要 ===")
        print(f"分析参数: 逐点更新, Z-Score回看={self.lookback_zscore}")
        print(f"Kalman参数: obs_cov={self.global_kalman_filter.observation_covariance}, trans_cov={self.global_kalman_filter.transition_covariance}")
        print(f"总数据点: {len(results)}, 有效信号点: {len(current_zscores_clean)} (跳过暖启动{self.lookback_zscore}点)")

        # Z-Score统计
        print(f"📊 Z-Score统计 (交易决策信号):")
        print(f"范围: {np.min(current_zscores_clean):.2f} 至 {np.max(current_zscores_clean):.2f}")
        print(f"均值: {np.mean(current_zscores_clean):.3f}, 标准差: {np.std(current_zscores_clean):.3f}")
        print(f"|Z|>0.5 信号比例: {np.mean(np.abs(current_zscores_clean) > 0.5):.2%}")
        print(f"|Z|>1.0 中信号比例: {np.mean(np.abs(current_zscores_clean) > 1.0):.2%}")
        print(f"|Z|>1.5 强信号比例: {np.mean(np.abs(current_zscores_clean) > 1.5):.2%}")
        print(f"|Z|>2.0 极端信号比例: {np.mean(np.abs(current_zscores_clean) > 2.0):.2%}")

        # 参数稳定性
        alpha_cv = np.std(alphas_clean) / abs(np.mean(alphas_clean)) * 100
        beta_cv = np.std(betas_clean) / abs(np.mean(betas_clean)) * 100
        print(f"🎛️ 参数稳定性:")
        print(f"Alpha变异系数(CV): {alpha_cv:.2f}%")
        print(f"Beta变异系数(CV): {beta_cv:.2f}%")

        # Kalman滤波质量
        spread_stds = self.analysis_results['spread_stds']
        print(f"📈 Kalman滤波质量:")
        print(f"Spread标准差范围: {np.min(spread_stds):.4f} - {np.max(spread_stds):.4f}")
        print(f"平均Spread标准差: {np.mean(spread_stds):.4f}")

        # 交易信号统计 - 修正信号频率计算
        weak_signals = np.mean(np.abs(current_zscores_clean) > 0.5)
        medium_signals = np.mean(np.abs(current_zscores_clean) > 1.0)
        strong_signals = np.mean(np.abs(current_zscores_clean) > 1.5)
        extreme_strong_signals = np.mean(np.abs(current_zscores_clean) > 2.0)

        print(f"🎯 交易信号分布:")
        print(f"  弱信号 (|Z|>0.5): {weak_signals:.1%} - 考虑建仓")
        print(f"  中等信号 (|Z|>1.0): {medium_signals:.1%} - 建议建仓")
        print(f"  强信号 (|Z|>1.5): {strong_signals:.1%} - 强烈建仓")
        print(f"  超强信号 (|Z|>2): {extreme_strong_signals:.1%} - 强烈建仓")
        # 信号频率计算
        if medium_signals > 0:
            print(f"预计每 {1.0 / medium_signals:.1f} 分钟出现一次中等信号（≈ {1440 * medium_signals:.1f} 次/日）")
        else:
            print("预计中等信号频率极低")

        # 官方交易信号统计
        if hasattr(self, 'official_signals') and self.official_signals is not None:
            print(f"\n📈 ArbitrageLab官方交易信号统计:")
            target_quantity = self.official_signals['target_quantity']

            # 信号统计
            long_signals = np.sum(target_quantity > 0)
            short_signals = np.sum(target_quantity < 0)
            neutral_signals = np.sum(target_quantity == 0)
            total_signals = len(target_quantity)

            print(f"  总信号数: {total_signals}")
            print(f"  做多信号: {long_signals} ({long_signals / total_signals:.1%})")
            print(f"  做空信号: {short_signals} ({short_signals / total_signals:.1%})")
            print(f"  中性信号: {neutral_signals} ({neutral_signals / total_signals:.1%})")

            # 交易活跃度
            active_signals = long_signals + short_signals
            if active_signals > 0:
                print(f"  交易活跃度: {active_signals / total_signals:.1%} (非中性信号比例)")
                print(f"  预计每 {total_signals / active_signals:.1f} 分钟出现交易信号")
            else:
                print("  交易活跃度: 0% - 无交易信号")

    def get_official_trading_signals(self, entry_std=2.0, exit_std=0.0):
        """
        获取ArbitrageLab官方库标准交易信号

        :param entry_std: (float) 入场标准差倍数
        :param exit_std: (float) 出场标准差倍数
        :return: (pd.DataFrame) 官方交易信号DataFrame，包含errors和target_quantity
        """
        return self.global_kalman_filter.trading_signals(entry_std, exit_std)


def main():
    loader = DataLoader()

    # Kalman滤波分析
    coin1, coin2 = 'MANA', 'SAND'  # 可以修改这里选择不同的交易对

    print(f"=== Kalman滤波分析 {coin1}-{coin2} ===")

    # Kalman滤波参数配置
    kalman_obs_cov = None  # 1.0      # 观测协方差
    kalman_trans_cov = None  # 0.001  # 转换协方差

    # 价格变换参数配置
    price_transform = 'log'  # 'log'=对数变换, 其他=原始价格

    # 加载1分钟数据
    start_date = '2025-08-10'
    data = loader.load_pair_data(coin1, coin2, start_date=start_date)
    print(f"1m数据长度: {len(data)}")

    # 创建分析器
    kf_analyzer = KalmanFilterAnalyzer(
        lookback_zscore=100,             # z-score历史基准
        kalman_obs_cov=kalman_obs_cov,   # Kalman观测协方差
        kalman_trans_cov=kalman_trans_cov  # Kalman转换协方差
    )

    print("\n✅ 开始分析...")
    _ = kf_analyzer.analyze_pair(data, coin1, coin2, price_transform=price_transform)
    print("🎉 分析完成！")


if __name__ == "__main__":
    main()

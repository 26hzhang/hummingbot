import warnings
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ArbitrageLab半衰期计算
from arbitragelab.cointegration_approach.utils import get_half_life_of_mean_reversion
from sklearn.linear_model import LinearRegression

# 协整检验相关包
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import datetime
import itertools
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ARBITRAGELAB_AVAILABLE = True


def calculate_half_life(spread):
    """包装ArbitrageLab半衰期计算，确保数据类型兼容"""
    if isinstance(spread, np.ndarray):
        spread_series = pd.Series(spread)
    else:
        spread_series = spread
    return get_half_life_of_mean_reversion(spread_series)


class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.pairs = [f.stem.replace("_5m", "") for f in self.data_dir.glob("*_5m.csv")]
        print(f"Found {len(self.pairs)} symbols")

    def load_pair_data(self, coin1, coin2, start_date=None, end_date=None, silent=False):
        """加载两个币对的对齐数据，返回干净的DataFrame"""
        pair1 = f"{coin1}USDT"
        pair2 = f"{coin2}USDT"

        file1 = self.data_dir / f"{pair1}_5m.csv"
        file2 = self.data_dir / f"{pair2}_5m.csv"

        if not file1.exists() or not file2.exists():
            print(f"Files not found for {pair1} or {pair2}")
            return None

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # 处理时间戳，统一时区并排序
        df1['timestamp'] = pd.to_datetime(df1['open_time'], utc=True)
        df2['timestamp'] = pd.to_datetime(df2['open_time'], utc=True)

        # 按时间排序
        df1 = df1.sort_values('timestamp').reset_index(drop=True)
        df2 = df2.sort_values('timestamp').reset_index(drop=True)

        common_start = max(df1['timestamp'].min(), df2['timestamp'].min())
        common_end = min(df1['timestamp'].max(), df2['timestamp'].max())

        # 时区对齐的日期处理
        if start_date:
            start_date_tz = pd.to_datetime(start_date, utc=True)
            common_start = max(common_start, start_date_tz)
        if end_date:
            end_date_tz = pd.to_datetime(end_date, utc=True)
            common_end = min(common_end, end_date_tz)

        df1_filtered = df1[(df1['timestamp'] >= common_start) & (df1['timestamp'] <= common_end)]
        df2_filtered = df2[(df2['timestamp'] >= common_start) & (df2['timestamp'] <= common_end)]

        df1_clean = df1_filtered[['timestamp', 'close', 'volume']].rename(columns={
            'close': f'{coin1}_price', 'volume': f'{coin1}_volume'
        })
        df2_clean = df2_filtered[['timestamp', 'close', 'volume']].rename(columns={
            'close': f'{coin2}_price', 'volume': f'{coin2}_volume'
        })

        # 内连接并再次排序确保最终结果有序
        merged_df = pd.merge(df1_clean, df2_clean, on='timestamp', how='inner')
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)

        if not silent:
            print(f"Loaded {len(merged_df)} aligned records for {coin1}-{coin2}")
            print(f"Time range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
        return merged_df


class RollingCointegrationAnalyzer:
    def __init__(self, window_size=300, step_size=None, is_pvalue_threshold=0.1, os_pvalue_threshold=0.2):
        self.window_size = window_size
        if not step_size:
            self.step_size = window_size
        else:
            self.step_size = step_size
        self.is_pvalue_threshold = is_pvalue_threshold  # IS阶段协整检验的p-value阈值
        self.os_pvalue_threshold = os_pvalue_threshold  # OS阶段协整检验的p-value阈值

    def analyze_pair_rolling(self, data, coin1, coin2, silent=False):
        """
        对指定pair进行完整的rolling window协整分析
        使用对数价格，IS估计参数，OS验证协整

        Args:
            data: 包含两个币种价格的DataFrame
            coin1, coin2: 币种名称
            silent: 是否静默执行（不显示图表和详细输出）
        """
        # 提取对数价格
        log_price1 = np.log(data[f'{coin1}_price'])
        log_price2 = np.log(data[f'{coin2}_price'])

        results = []
        total_points = len(data)

        if not silent:
            print(f"Analyzing {coin1}-{coin2} with {self.window_size}K IS + {self.window_size}K OS windows...")

        for start_idx in range(0, total_points - 2 * self.window_size, self.step_size):
            # IS数据 (前window_size根K线)
            is_end = start_idx + self.window_size
            is_log_p1 = log_price1.iloc[start_idx:is_end]
            is_log_p2 = log_price2.iloc[start_idx:is_end]

            # OS数据 (后window_size根K线)
            os_start = is_end
            os_end = os_start + self.window_size

            if os_end > total_points:
                break

            os_log_p1 = log_price1.iloc[os_start:os_end]
            os_log_p2 = log_price2.iloc[os_start:os_end]

            # IS阶段：估计协整参数
            is_result = self._estimate_cointegration(is_log_p1, is_log_p2)

            if is_result is None:
                continue

            # OS阶段：使用IS的参数验证
            os_result = self._validate_cointegration(
                os_log_p1, os_log_p2, is_result['alpha'], is_result['beta'])

            if os_result is None:
                continue

            # 记录结果
            window_result = {
                'window_start': start_idx,
                'is_start_time': data['timestamp'].iloc[start_idx],
                'is_end_time': data['timestamp'].iloc[is_end - 1],
                'os_start_time': data['timestamp'].iloc[os_start],
                'os_end_time': data['timestamp'].iloc[os_end - 1],
                'alpha': is_result['alpha'],
                'beta': is_result['beta'],
                'r_squared': is_result['r_squared'],
                'is_pvalue': is_result['pvalue'],
                'os_pvalue': os_result['pvalue'],
                'is_half_life': is_result['half_life'],
                'os_half_life': os_result['half_life'],
                'is_cointegrated': is_result['pvalue'] < self.is_pvalue_threshold,
                'os_cointegrated': os_result['pvalue'] < self.os_pvalue_threshold,
                'both_cointegrated': (is_result['pvalue'] < self.is_pvalue_threshold) and (os_result['pvalue'] < self.os_pvalue_threshold)
            }

            results.append(window_result)

        results_df = pd.DataFrame(results)

        # 计算统计摘要
        summary = self._calculate_summary(results_df, coin1, coin2, silent=silent)

        # 生成可视化（仅非静默模式）
        if not silent:
            self._plot_results(results_df, coin1, coin2)

        return {
            'results': results_df,
            'summary': summary
        }

    def _estimate_cointegration(self, log_p1, log_p2):
        """IS阶段：估计协整参数并检验"""
        try:
            # 线性回归估计协整参数
            X = log_p1.values.reshape(-1, 1)
            y = log_p2.values
            reg = LinearRegression().fit(X, y)

            alpha = reg.intercept_
            beta = reg.coef_[0]

            # 计算R-squared
            y_pred = reg.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # 计算spread并检验平稳性
            spread = y - alpha - beta * log_p1.values
            adf_stat, pvalue, _, _, _, _ = adfuller(spread, regression="n", autolag='AIC')

            # 计算半衰期
            try:
                half_life = calculate_half_life(spread)
                if np.isnan(half_life) or np.isinf(half_life):
                    half_life = None
            except:
                half_life = None

            return {
                'alpha': alpha,
                'beta': beta,
                'r_squared': r_squared,
                'pvalue': pvalue,
                'spread': spread,
                'half_life': half_life
            }
        except:
            return None

    def _validate_cointegration(self, log_p1, log_p2, alpha, beta):
        """OS阶段：使用IS参数验证协整"""
        try:
            # 使用IS的参数计算OS的spread
            spread = log_p2.values - alpha - beta * log_p1.values

            # 检验OS spread的平稳性
            adf_stat, pvalue, _, _, _, _ = adfuller(spread, regression="n", autolag='AIC')

            # 计算OS半衰期
            try:
                half_life = calculate_half_life(spread)
                if np.isnan(half_life) or np.isinf(half_life):
                    half_life = None
            except:
                half_life = None

            return {
                'pvalue': pvalue,
                'spread': spread,
                'half_life': half_life
            }
        except:
            return None

    def _calculate_summary(self, results_df, coin1, coin2, silent=False):
        """计算协整分析摘要统计"""
        if len(results_df) == 0:
            return None

        is_coint_rate = results_df['is_cointegrated'].mean()
        os_coint_rate = results_df['os_cointegrated'].mean()
        both_coint_rate = results_df['both_cointegrated'].mean()

        # 参数稳定性
        beta_cv = results_df['beta'].std() / results_df['beta'].mean()

        # R-squared统计
        avg_r_squared = results_df['r_squared'].mean()

        # 半衰期统计
        is_half_life_valid = results_df['is_half_life'].dropna()
        os_half_life_valid = results_df['os_half_life'].dropna()
        avg_is_half_life = is_half_life_valid.mean() if len(is_half_life_valid) > 0 else None
        avg_os_half_life = os_half_life_valid.mean() if len(os_half_life_valid) > 0 else None
        half_life_stability = abs(avg_is_half_life - avg_os_half_life) / avg_is_half_life if avg_is_half_life and avg_os_half_life and avg_is_half_life > 0 else None

        # 综合评分
        stability_score = 1 / (1 + beta_cv)  # 变异系数越小越好
        overall_score = is_coint_rate * 0.3 + os_coint_rate * 0.5 + stability_score * 0.2

        summary = {
            'pair': f'{coin1}-{coin2}',
            'total_windows': len(results_df),
            'is_cointegration_rate': is_coint_rate,
            'os_cointegration_rate': os_coint_rate,
            'both_cointegration_rate': both_coint_rate,
            'beta_mean': results_df['beta'].mean(),
            'beta_std': results_df['beta'].std(),
            'beta_cv': beta_cv,
            'avg_r_squared': avg_r_squared,
            'avg_is_half_life': avg_is_half_life,
            'avg_os_half_life': avg_os_half_life,
            'half_life_stability': half_life_stability,
            'overall_score': overall_score
        }

        if not silent:
            print(f"\n=== {coin1}-{coin2} 协整分析摘要 ===")
            print(f"总窗口数: {len(results_df)}")
            print(f"IS协整率: {is_coint_rate:.2%}")
            print(f"OS协整率: {os_coint_rate:.2%}")
            print(f"双重协整率: {both_coint_rate:.2%}")
            print(f"Beta稳定性(CV): {beta_cv:.3f}")
            print(f"平均R-squared: {avg_r_squared:.3f}")
            if avg_is_half_life:
                print(f"IS平均半衰期: {avg_is_half_life:.1f} periods")
            if avg_os_half_life:
                print(f"OS平均半衰期: {avg_os_half_life:.1f} periods")
            if half_life_stability:
                print(f"半衰期稳定性: {half_life_stability:.3f}")
            print(f"综合评分: {overall_score:.3f}")

        return summary

    def _plot_results(self, results_df, coin1, coin2):
        """生成协整分析可视化图表"""
        if len(results_df) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{coin1}-{coin2} Rolling Cointegration Analysis', fontsize=16)

        # 1. IS/OS p-value时间序列
        axes[0, 0].plot(results_df.index, results_df['is_pvalue'], 'b-', label='IS p-value', alpha=0.7)
        axes[0, 0].plot(results_df.index, results_df['os_pvalue'], 'r-', label='OS p-value', alpha=0.7)
        axes[0, 0].axhline(y=0.01, color='b', linestyle='--', alpha=0.5, label='IS threshold')
        axes[0, 0].axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='OS threshold')
        axes[0, 0].set_ylabel('P-value')
        axes[0, 0].set_title('Cointegration P-values Over Time')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')

        # 2. R-squared时间序列
        axes[0, 1].plot(results_df.index, results_df['r_squared'], 'purple', alpha=0.7)
        axes[0, 1].axhline(y=results_df['r_squared'].mean(), color='purple', linestyle='--', alpha=0.5, label='Mean')
        axes[0, 1].set_ylabel('R-squared')
        axes[0, 1].set_title('R-squared Over Time')
        axes[0, 1].legend()

        # 3. Beta系数稳定性
        axes[1, 0].plot(results_df.index, results_df['beta'], 'g-', alpha=0.7)
        axes[1, 0].axhline(y=results_df['beta'].mean(), color='g', linestyle='--', alpha=0.5, label='Mean')
        axes[1, 0].set_ylabel('Beta Coefficient')
        axes[1, 0].set_title('Beta Coefficient Stability')
        axes[1, 0].legend()

        # 4. IS vs OS散点图，颜色表示R-squared
        scatter = axes[1, 1].scatter(
            -np.log(results_df['is_pvalue']), -np.log(results_df['os_pvalue']), 
            c=results_df['r_squared'], alpha=0.6, cmap='viridis')
        axes[1, 1].axvline(x=-np.log(0.01), color='b', linestyle='--', alpha=0.5, label='IS threshold')
        axes[1, 1].axhline(y=-np.log(0.05), color='r', linestyle='--', alpha=0.5, label='OS threshold')
        axes[1, 1].set_xlabel('-log(IS p-value)')
        axes[1, 1].set_ylabel('-log(OS p-value)')
        axes[1, 1].set_title('IS vs OS Cointegration (color=R²)')
        axes[1, 1].legend()
        plt.colorbar(scatter, ax=axes[1, 1], label='R-squared')

        plt.tight_layout()
        plt.show()


def analyze_single_pair(args):
    """单个pair的协整分析函数，用于多线程"""
    data_loader, coin1, coin2, start_date, analyzer, thread_id = args
    try:
        # 加载数据
        data = data_loader.load_pair_data(coin1, coin2, start_date=start_date, silent=True)

        if data is None or len(data) < 2 * analyzer.window_size:
            return None, {'pair': f'{coin1}-{coin2}', 'reason': 'insufficient_data'}

        # 执行协整分析（静默模式，使用传入的analyzer）
        result = analyzer.analyze_pair_rolling(data, coin1, coin2, silent=True)

        if result and result['summary']:
            return result['summary'], None
        else:
            return None, {'pair': f'{coin1}-{coin2}', 'reason': 'analysis_failed'}

    except Exception as e:
        return None, {'pair': f'{coin1}-{coin2}', 'reason': f'error: {str(e)}'}


def single_cointegration_analysis(data_loader, coin_pair, output_dir, analyzer, start_date="2023-01-01"):
    """单配对协整分析函数"""
    try:
        coin1, coin2 = coin_pair.split('-')
        print(f"=== 开始单配对协整分析: {coin1}-{coin2} ===")
        
        # 加载数据
        data = data_loader.load_pair_data(coin1, coin2, start_date=start_date, silent=False)
        
        if data is None or len(data) < 2 * analyzer.window_size:
            print(f"数据不足，需要至少 {2 * analyzer.window_size} 条记录，实际获得 {len(data) if data is not None else 0} 条")
            return None
        
        # 执行协整分析（使用传入的analyzer）
        result = analyzer.analyze_pair_rolling(data, coin1, coin2, silent=False)
        
        if result and result['summary']:
            # 保存单配对结果
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存详细结果
            detailed_file = output_dir / f'single_pair_analysis_{coin1}_{coin2}_{timestamp}.csv'
            result['results'].to_csv(detailed_file, index=False)
            
            # 保存摘要
            summary_file = output_dir / f'single_pair_summary_{coin1}_{coin2}_{timestamp}.csv'
            summary_df = pd.DataFrame([result['summary']])
            summary_df.to_csv(summary_file, index=False)
            
            print(f"\n结果已保存:")
            print(f"- 详细分析: {detailed_file}")
            print(f"- 摘要结果: {summary_file}")
            
            return result
        else:
            print("协整分析失败")
            return None
            
    except Exception as e:
        print(f"单配对分析出错: {str(e)}")
        return None


def batch_cointegration_analysis(data_loader, output_dir, analyzer, start_date="2023-01-01", max_workers=8):
    """批量分析所有可能的交易对组合"""
    # 获取所有可用币种
    available_coins = [pair.replace('USDT', '') for pair in data_loader.pairs]
    # available_coins = [i for i in available_coins if i in top_symbols]

    print(f"可用币种: {len(available_coins)} 个")
    print(f"币种列表: {available_coins}")

    # 生成所有可能的pair组合（包括不同方向）
    all_pairs = []
    for coin1, coin2 in itertools.permutations(available_coins, 2):
        all_pairs.append((coin1, coin2))

    print(f"总共需要分析 {len(all_pairs)} 个pair组合")
    print(f"使用 {max_workers} 个线程并行处理")

    # 存储所有分析结果
    all_summaries = []
    failed_pairs = []

    # 准备多线程参数
    thread_args = [(data_loader, coin1, coin2, start_date, analyzer, i % max_workers) for i, (coin1, coin2) in enumerate(all_pairs)]

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_pair = {executor.submit(analyze_single_pair, args): args for args in thread_args}

        # 使用进度条收集结果
        for future in tqdm(as_completed(future_to_pair), total=len(all_pairs), desc="协整分析进度"):
            try:
                summary, failed = future.result()
                if summary:
                    all_summaries.append(summary)
                if failed:
                    failed_pairs.append(failed)
            except Exception as e:
                coin1, coin2, _ = future_to_pair[future]
                failed_pairs.append({'pair': f'{coin1}-{coin2}', 'reason': f'thread_error: {str(e)}'})
                continue

    # 转换为DataFrame
    summaries_df = pd.DataFrame(all_summaries)
    failed_df = pd.DataFrame(failed_pairs)

    # 按综合评分排序
    if len(summaries_df) > 0:
        summaries_df = summaries_df.sort_values('overall_score', ascending=False)

        print(f"\n=== 批量分析完成 ===")
        print(f"成功分析: {len(summaries_df)} 个pairs")
        print(f"失败分析: {len(failed_df)} 个pairs")

        # 显示top 10结果
        print(f"\n=== TOP 10 协整Pairs ===")
        top_cols = ['pair', 'both_cointegration_rate', 'avg_r_squared', 'beta_cv', 'avg_is_half_life', 'avg_os_half_life', 'overall_score']
        print(summaries_df[top_cols].head(10).to_string(index=False))

        # 保存结果
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = output_dir / f'cointegration_analysis_summary_{timestamp}.csv'
        failed_file = output_dir / f'failed_pairs_{timestamp}.csv'

        summaries_df.to_csv(summary_file, index=False)
        if len(failed_df) > 0:
            failed_df.to_csv(failed_file, index=False)

        print(f"\n结果已保存:")
        print(f"- 成功分析: {summary_file}")
        if len(failed_df) > 0:
            print(f"- 失败记录: {failed_file}")

        return summaries_df, failed_df
    else:
        print("没有成功分析的pairs")
        return None, failed_df


def main():
    parser = argparse.ArgumentParser(description='统计套利配对选择分析工具')
    parser.add_argument(
        'mode', choices=['single_coint', 'batch_coint'], help='分析模式：单配对分析或批量分析')
    parser.add_argument('--coin_pair', type=str, help='币种对，格式为 COIN1-COIN2 (例如：BTC-ETH)，仅在single模式下需要')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='数据起始日期 (默认: 2023-01-01)')
    parser.add_argument('--max_workers', type=int, default=8, help='批量分析时的并行线程数 (默认: 8)')
    parser.add_argument('--window_size', type=int, default=600, help='分析窗口大小 (默认: 600)')
    parser.add_argument('--step_size', type=int, default=None, help='滚动步长，默认等于window_size (默认: None)')
    parser.add_argument('--is_pvalue_threshold', type=float, default=0.1, help='IS阶段协整检验p-value阈值 (默认: 0.1)')
    parser.add_argument('--os_pvalue_threshold', type=float, default=0.2, help='OS阶段协整检验p-value阈值 (默认: 0.2)')
    parser.add_argument('--data_dir', type=str, default='/Users/zhanghao/GitHub/hummingbot/data/futures_5m', help='数据目录路径')
    parser.add_argument(
        '--output_dir', type=str, default='/Users/zhanghao/GitHub/hummingbot/data/analysis/futures_5m', help='输出目录路径')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.mode == 'single_coint' and not args.coin_pair:
        parser.error("单配对分析模式需要指定 --coin_pair 参数")
    
    if args.coin_pair and '-' not in args.coin_pair:
        parser.error("币种对格式错误，应为 COIN1-COIN2 格式 (例如：BTC-ETH)")
    
    # 初始化数据加载器和输出目录
    data_loader = DataLoader(data_dir=args.data_dir)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建协整分析器
    analyzer = RollingCointegrationAnalyzer(
        window_size=args.window_size,
        step_size=args.step_size,
        is_pvalue_threshold=args.is_pvalue_threshold,
        os_pvalue_threshold=args.os_pvalue_threshold
    )
    
    if args.mode == 'single_coint':
        result = single_cointegration_analysis(
            data_loader=data_loader,
            coin_pair=args.coin_pair,
            output_dir=output_dir,
            analyzer=analyzer,
            start_date=args.start_date
        )
        
    elif args.mode == 'batch_coint':
        print("=== 开始批量协整分析 ===")
        summary_results, failed_results = batch_cointegration_analysis(
            data_loader=data_loader,
            output_dir=output_dir,
            analyzer=analyzer,
            start_date=args.start_date,
            max_workers=args.max_workers
        )


if __name__ == "__main__":
    main()

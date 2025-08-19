import argparse
import asyncio
import csv
import logging
import pickle
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm.asyncio import tqdm

# 添加hummingbot路径以使用现有基础设施 / Add hummingbot path to use existing infrastructure
current_dir = Path(__file__).parent
hummingbot_root = current_dir if current_dir.name == "hummingbot" else current_dir.parent
sys.path.insert(0, str(hummingbot_root))

# Hummingbot imports for futures trading
from hummingbot.connector.derivative.binance_perpetual import (
    binance_perpetual_constants as FUTURES_CONSTANTS,
    binance_perpetual_web_utils as futures_web_utils,
)

# Hummingbot imports for spot trading
from hummingbot.connector.exchange.binance import (
    binance_constants as SPOT_CONSTANTS,
    binance_web_utils as spot_web_utils,
)
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.api_throttler.data_types import LinkedLimitWeightPair, RateLimit
from hummingbot.core.web_assistant.connections.data_types import RESTMethod

# 初始化日志（具体配置在main函数中设置）/ Initialize logging (specific config set in main function)
logger = logging.getLogger(__name__)


@dataclass
class DownloadTask:
    """下载任务数据类 / Download task data class"""
    symbol: str
    interval: str
    start_time: datetime
    end_time: datetime
    market_type: str
    chunk_id: str
    status: str = "pending"  # pending, downloading, completed, failed
    retry_count: int = 0
    file_path: Optional[str] = None
    total_records: int = 0
    downloaded_records: int = 0
    error_message: Optional[str] = None


@dataclass
class DownloadProgress:
    """下载进度数据类 / Download progress data class"""
    symbol: str
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    total_records: int
    downloaded_records: int
    start_time: datetime
    last_update: datetime
    estimated_completion: Optional[datetime] = None


class BinanceDataDownloader:
    """基于Hummingbot的Binance数据下载器 / Binance Data Downloader based on Hummingbot"""

    def __init__(self, data_dir: str = "data", temp_dir: str = "data/temp", max_concurrent: int = 5,
                 api_key: str = None, api_secret: str = None):
        self.data_dir = Path(data_dir)
        self.temp_data_dir = Path(temp_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_data_dir.mkdir(parents=True, exist_ok=True)

        # 并发控制 / Concurrency control
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # API 密钥 / API credentials
        self.api_key = api_key
        self.api_secret = api_secret

        # 创建优化的throttler配置，专为大批量下载设计 / Create optimized throttler config for bulk downloads
        self.SPOT_KLINES_URL = "/klines"
        self.FUTURES_KLINES_URL = "v1/klines"

        # 现货市场高性能限流配置 / Spot market high-performance rate limits
        spot_custom_rate_limits = [
            # 优化配置，目标1000请求/分钟 / Optimized config targeting 1000 requests/minute
            RateLimit(limit_id=SPOT_CONSTANTS.REQUEST_WEIGHT, limit=5000, time_interval=60),  # 基于官方6000/分钟限制
            RateLimit(limit_id=self.SPOT_KLINES_URL, limit=1000, time_interval=60, weight=2,
                      linked_limits=[LinkedLimitWeightPair(SPOT_CONSTANTS.REQUEST_WEIGHT, weight=2)]),
            RateLimit(limit_id=SPOT_CONSTANTS.TICKER_PRICE_CHANGE_PATH_URL, limit=20, time_interval=60, weight=2,
                      linked_limits=[LinkedLimitWeightPair(SPOT_CONSTANTS.REQUEST_WEIGHT, weight=2)]),
        ]

        # 期货市场高性能限流配置 / Futures market high-performance rate limits
        futures_custom_rate_limits = [
            # 优化配置，目标1000请求/分钟 / Optimized config targeting 1000 requests/minute
            RateLimit(limit_id=FUTURES_CONSTANTS.REQUEST_WEIGHT, limit=2000, time_interval=60),  # 基于官方2400/分钟限制
            RateLimit(limit_id=self.FUTURES_KLINES_URL, limit=1000, time_interval=60, weight=2,
                      linked_limits=[LinkedLimitWeightPair(FUTURES_CONSTANTS.REQUEST_WEIGHT, weight=2)]),
            RateLimit(limit_id=FUTURES_CONSTANTS.TICKER_PRICE_CHANGE_URL, limit=20, time_interval=60, weight=1,
                      linked_limits=[LinkedLimitWeightPair(FUTURES_CONSTANTS.REQUEST_WEIGHT, weight=1)]),
        ]

        # 创建自定义throttler / Create custom throttlers
        self.spot_throttler = AsyncThrottler(spot_custom_rate_limits)
        self.futures_throttler = AsyncThrottler(futures_custom_rate_limits)

        # API factories with custom throttlers
        # 如果提供了API密钥，创建带认证的factory / If API keys provided, create authenticated factory
        if self.api_key and self.api_secret:
            from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_auth import BinancePerpetualAuth
            from hummingbot.connector.exchange.binance.binance_auth import BinanceAuth
            from hummingbot.connector.time_synchronizer import TimeSynchronizer

            time_provider = TimeSynchronizer()
            self.spot_auth = BinanceAuth(api_key=self.api_key, secret_key=self.api_secret, time_provider=time_provider)
            self.futures_auth = BinancePerpetualAuth(api_key=self.api_key, api_secret=self.api_secret, time_provider=time_provider)

            self.spot_api_factory = spot_web_utils.build_api_factory(throttler=self.spot_throttler, auth=self.spot_auth)
            self.futures_api_factory = futures_web_utils.build_api_factory(throttler=self.futures_throttler, auth=self.futures_auth)
            logger.info("API 认证已启用 / API authentication enabled")
        else:
            # 公开API模式 / Public API mode
            self.spot_api_factory = spot_web_utils.build_api_factory(throttler=self.spot_throttler)
            self.futures_api_factory = futures_web_utils.build_api_factory(throttler=self.futures_throttler)
            logger.info("使用公开API模式 / Using public API mode")

        # 进度跟踪 / Progress tracking
        self.progress_file = hummingbot_root / "logs/download_progress.pkl"
        self.progress_data: Dict[str, DownloadProgress] = {}
        self.load_progress()

        # 错误处理和退避策略 / Error handling and backoff strategy
        self.max_retries = 5
        self.base_delay = 60  # 基础延迟60秒 / Base delay 60 seconds
        self.max_delay = 1800  # 最大延迟30分钟 / Max delay 30 minutes
        self.ban_detected = False
        self.last_ban_time = 0

        # 稳定性控制机制（从fast版本借鉴）/ Stability control mechanism (from fast version)
        self.consecutive_errors = 0
        self.last_success_time = time.time()
        self.warning_timestamps = {}  # 警告类型 -> 最后警告时间
        self.warning_cooldown = 300  # 5分钟内不重复相同警告
        self.global_silence_until = 0  # 全局静默到某个时间

    def _get_temp_dir(self, symbol: str) -> Path:
        """获取指定交易对的临时目录 / Get temporary directory for symbol"""
        temp_symbol_dir = self.temp_data_dir / symbol
        temp_symbol_dir.mkdir(exist_ok=True)
        return temp_symbol_dir

    def _cleanup_temp_dir(self, symbol: str):
        """清理指定交易对的临时目录 / Clean up temporary directory for symbol"""
        temp_symbol_dir = self.temp_data_dir / symbol
        if temp_symbol_dir.exists():
            shutil.rmtree(temp_symbol_dir)
            logger.info(f"Cleaned up temporary directory: {temp_symbol_dir}")

    def load_progress(self):
        """加载下载进度 / Load download progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    self.progress_data = pickle.load(f)
                logger.info("Loaded previous download progress")
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")
                self.progress_data = {}
        else:
            self.progress_data = {}

    def save_progress(self):
        """保存下载进度 / Save download progress"""
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(self.progress_data, f)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    async def get_top_volume_symbols(self, market_type: str, limit: int = 50, volume_type: str = "24h") -> List[str]:
        """获取交易量最高的币种 / Get top volume symbols

        Args:
            market_type: 市场类型 ('spot' 或 'futures')
            limit: 返回的币种数量
            volume_type: 成交量类型 ('24h', '7d', '30d', '1y')
        """
        try:
            if volume_type == "24h":
                # 使用24小时ticker数据 / Use 24hr ticker data
                if market_type == "spot":
                    rest_assistant = await self.spot_api_factory.get_rest_assistant()
                    async with self.spot_throttler.execute_task(SPOT_CONSTANTS.TICKER_PRICE_CHANGE_PATH_URL):
                        response = await rest_assistant.execute_request(
                            url=spot_web_utils.public_rest_url(SPOT_CONSTANTS.TICKER_PRICE_CHANGE_PATH_URL),
                            method=RESTMethod.GET,
                            throttler_limit_id=SPOT_CONSTANTS.TICKER_PRICE_CHANGE_PATH_URL
                        )

                    usdt_pairs = [item for item in response if item['symbol'].endswith('USDT')]
                    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
                    symbols = [pair['symbol'] for pair in sorted_pairs[:limit]]

                else:  # futures
                    rest_assistant = await self.futures_api_factory.get_rest_assistant()
                    async with self.futures_throttler.execute_task(FUTURES_CONSTANTS.TICKER_PRICE_CHANGE_URL):
                        response = await rest_assistant.execute_request(
                            url=futures_web_utils.public_rest_url(FUTURES_CONSTANTS.TICKER_PRICE_CHANGE_URL),
                            method=RESTMethod.GET,
                            throttler_limit_id=FUTURES_CONSTANTS.TICKER_PRICE_CHANGE_URL
                        )

                    usdt_pairs = [item for item in response if item['symbol'].endswith('USDT')]
                    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
                    symbols = [pair['symbol'] for pair in sorted_pairs[:limit]]
            else:
                # 基于指定时间范围计算成交量 / Calculate volume based on specified time period
                symbols = await self._get_top_volume_by_period(market_type, limit, volume_type)

            logger.info(f"Retrieved top {len(symbols)} {market_type} symbols (volume_type: {volume_type})")
            logger.info(f"Symbols: {symbols}")
            return symbols

        except Exception as e:
            logger.error(f"Failed to get top volume symbols for {market_type}: {e}")
            return []

    async def _get_top_volume_by_period(self, market_type: str, limit: int, volume_type: str) -> List[str]:
        """基于指定成交量类型计算交易量最高的币种 / Calculate top volume symbols for specified volume type"""
        from datetime import timedelta

        # 计算时间范围 / Calculate time range
        now = datetime.now()
        if volume_type == '24h':
            start_time = now - timedelta(hours=24)
            interval = '1h'
        elif volume_type == '7d':
            start_time = now - timedelta(days=7)
            interval = '4h'
        elif volume_type == '30d':
            start_time = now - timedelta(days=30)
            interval = '1d'
        elif volume_type == '1y':
            start_time = now - timedelta(days=365)
            interval = '1w'
        else:
            # 默认24小时 / Default 24 hours
            start_time = now - timedelta(hours=24)
            interval = '1h'

        try:
            # 首先获取所有USDT交易对 / First get all USDT pairs
            if market_type == "spot":
                rest_assistant = await self.spot_api_factory.get_rest_assistant()
                # 获取交易信息 / Get exchange info
                async with self.spot_throttler.execute_task(SPOT_CONSTANTS.REQUEST_WEIGHT):
                    exchange_info = await rest_assistant.execute_request(
                        url=spot_web_utils.public_rest_url("/exchangeInfo"),
                        method=RESTMethod.GET,
                        throttler_limit_id=SPOT_CONSTANTS.REQUEST_WEIGHT
                    )
                all_symbols = [s['symbol'] for s in exchange_info['symbols']
                               if s['symbol'].endswith('USDT') and s['status'] == 'TRADING']
            else:  # futures
                rest_assistant = await self.futures_api_factory.get_rest_assistant()
                async with self.futures_throttler.execute_task(FUTURES_CONSTANTS.REQUEST_WEIGHT):
                    exchange_info = await rest_assistant.execute_request(
                        url=futures_web_utils.public_rest_url("/exchangeInfo"),
                        method=RESTMethod.GET,
                        throttler_limit_id=FUTURES_CONSTANTS.REQUEST_WEIGHT
                    )
                all_symbols = [s['symbol'] for s in exchange_info['symbols']
                               if s['symbol'].endswith('USDT') and s['status'] == 'TRADING']

            # 限制检查的币种数量以避免过多API调用 / Limit symbols to check to avoid too many API calls
            check_symbols = all_symbols[:200]  # 检查前200个币种

            # 计算每个币种的成交量 / Calculate volume for each symbol
            symbol_volumes = []
            for symbol in check_symbols:
                try:
                    start_ts = int(start_time.timestamp() * 1000)
                    end_ts = int(now.timestamp() * 1000)

                    params = {
                        'symbol': symbol,
                        'interval': interval,
                        'startTime': start_ts,
                        'endTime': end_ts,
                        'limit': 1000
                    }

                    if market_type == "spot":
                        rest_assistant = await self.spot_api_factory.get_rest_assistant()
                        url = spot_web_utils.public_rest_url("/klines")
                        throttler_id = SPOT_CONSTANTS.REQUEST_WEIGHT
                        async with self.spot_throttler.execute_task(throttler_id):
                            response = await rest_assistant.execute_request(
                                url=url, method=RESTMethod.GET, params=params,
                                throttler_limit_id=throttler_id
                            )
                    else:  # futures
                        rest_assistant = await self.futures_api_factory.get_rest_assistant()
                        url = futures_web_utils.public_rest_url(FUTURES_CONSTANTS.KLINES_URL)
                        throttler_id = FUTURES_CONSTANTS.REQUEST_WEIGHT
                        async with self.futures_throttler.execute_task(throttler_id):
                            response = await rest_assistant.execute_request(
                                url=url, method=RESTMethod.GET, params=params,
                                throttler_limit_id=throttler_id
                            )

                    # 计算总成交量 / Calculate total volume
                    total_volume = sum(float(kline[7]) for kline in response)  # quoteVolume
                    symbol_volumes.append((symbol, total_volume))

                    # 简短延迟避免过快请求 / Brief delay to avoid too fast requests
                    await asyncio.sleep(0.01)

                except Exception as e:
                    logger.debug(f"Failed to get volume for {symbol}: {e}")
                    continue

            # 按成交量排序并返回前N个 / Sort by volume and return top N
            symbol_volumes.sort(key=lambda x: x[1], reverse=True)
            top_symbols = [symbol for symbol, _ in symbol_volumes[:limit]]

            logger.info(f"Calculated top {len(top_symbols)} symbols for {volume_type} volume")
            return top_symbols

        except Exception as e:
            logger.error(f"Failed to calculate top volume by period: {e}")
            # 回退到24小时数据 / Fallback to 24hr data
            return await self.get_top_volume_symbols(market_type, limit)

    async def test_private_api(self, market_type: str = "futures") -> bool:
        """测试私有API访问 / Test private API access"""
        if not self.api_key or not self.api_secret:
            logger.warning("No API credentials provided, cannot test private API")
            return False

        try:
            if market_type == "spot":
                rest_assistant = await self.spot_api_factory.get_rest_assistant()
                # 测试账户信息API / Test account info API
                async with self.spot_throttler.execute_task(SPOT_CONSTANTS.REQUEST_WEIGHT):
                    response = await rest_assistant.execute_request(
                        url=spot_web_utils.private_rest_url("/account"),
                        method=RESTMethod.GET,
                        throttler_limit_id=SPOT_CONSTANTS.REQUEST_WEIGHT,
                        is_auth_required=True
                    )
                logger.info(f"Spot account balances count: {len(response['balances'])}")

            else:  # futures
                rest_assistant = await self.futures_api_factory.get_rest_assistant()
                # 测试余额信息API / Test balance info API (simpler endpoint)
                async with self.futures_throttler.execute_task(FUTURES_CONSTANTS.REQUEST_WEIGHT):
                    response = await rest_assistant.execute_request(
                        url=futures_web_utils.private_rest_url("/balance"),
                        method=RESTMethod.GET,
                        throttler_limit_id=FUTURES_CONSTANTS.REQUEST_WEIGHT,
                        is_auth_required=True
                    )
                usdt_balance = next((b for b in response if b['asset'] == 'USDT'), {})
                logger.info(f"Futures USDT balance: {usdt_balance.get('balance', 'N/A')}")

            logger.info("Private API test successful")
            return True

        except Exception as e:
            logger.error(f"Private API test failed: {e}")
            return False

    def get_existing_file_time_range(self, symbol: str, market_type: str, interval: str) -> Optional[Tuple[datetime, datetime]]:
        """获取已存在文件的时间范围 / Get time range of existing file"""
        market_dir = self.data_dir / f"{market_type}_{interval}"
        filename = f"{symbol}_{interval}.csv"
        file_path = market_dir / filename
        
        if not file_path.exists():
            return None
            
        try:
            # 读取CSV文件的第一行和最后一行来获取时间范围 / Read first and last row to get time range
            df = pd.read_csv(file_path)
            if len(df) == 0:
                return None
                
            # 解析时间字符串 / Parse time strings
            first_time_str = df.iloc[0]['open_time']
            last_time_str = df.iloc[-1]['open_time']
            
            first_time = datetime.strptime(first_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            last_time = datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            
            return (first_time, last_time)
            
        except Exception as e:
            logger.warning(f"Failed to read existing file time range for {symbol}: {e}")
            return None
    
    def calculate_download_ranges(self, symbol: str, market_type: str, interval: str, 
                                 requested_start: datetime, requested_end: datetime) -> List[Tuple[datetime, datetime]]:
        """计算需要下载的时间范围，实现增量下载逻辑 / Calculate download ranges for incremental download"""
        existing_range = self.get_existing_file_time_range(symbol, market_type, interval)
        
        if existing_range is None:
            # 文件不存在，下载整个请求范围 / File doesn't exist, download entire requested range
            return [(requested_start, requested_end)]
            
        file_start, file_end = existing_range
        download_ranges = []
        
        logger.info(f"Existing file range for {symbol}: {file_start.strftime('%Y-%m-%d %H:%M:%S')} to {file_end.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Requested range: {requested_start.strftime('%Y-%m-%d %H:%M:%S')} to {requested_end.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 情况1: 请求的开始时间小于文件开始时间 / Case 1: requested start < file start
        if requested_start < file_start:
            if requested_end < file_start:
                # 请求范围在文件范围之前，需要连接 / Requested range before file range, need to connect
                download_ranges.append((requested_start, file_start))
                logger.info(f"Adding pre-file range: {requested_start.strftime('%Y-%m-%d %H:%M:%S')} to {file_start.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # 补充下载从请求开始到文件开始的部分 / Download from requested start to file start
                download_ranges.append((requested_start, file_start))
                logger.info(f"Adding prefix range: {requested_start.strftime('%Y-%m-%d %H:%M:%S')} to {file_start.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 情况2: 请求的结束时间大于文件结束时间 / Case 2: requested end > file end
        if requested_end > file_end:
            if requested_start > file_end:
                # 请求范围在文件范围之后，需要连接 / Requested range after file range, need to connect
                download_ranges.append((file_end, requested_end))
                logger.info(f"Adding post-file range: {file_end.strftime('%Y-%m-%d %H:%M:%S')} to {requested_end.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # 补充下载从文件结束到请求结束的部分 / Download from file end to requested end
                download_ranges.append((file_end, requested_end))
                logger.info(f"Adding suffix range: {file_end.strftime('%Y-%m-%d %H:%M:%S')} to {requested_end.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not download_ranges:
            logger.info(f"No additional data needed for {symbol}, existing file covers requested range")
        
        return download_ranges

    def check_file_exists(self, symbol: str, market_type: str, interval: str, start_date: str = None, end_date: str = None, force_redownload: bool = False) -> bool:
        """检查文件是否已存在 / Check if file already exists"""
        if force_redownload:
            return False

        # 使用新的文件名格式 / Use new filename format
        market_dir = self.data_dir / f"{market_type}_{interval}"
        filename = f"{symbol}_{interval}.csv"
        file_path = market_dir / filename

        if file_path.exists():
            logger.info(f"文件已存在，将进行增量检查: {file_path} / File exists, will perform incremental check: {file_path}")
            return True
        return False

    def generate_chunks(self, start_dt: datetime, end_dt: datetime, chunk_hours: int = 24) -> List[Tuple[datetime, datetime]]:
        """生成时间分块 / Generate time chunks"""
        chunks = []
        current = start_dt

        while current < end_dt:
            chunk_end = min(current + timedelta(hours=chunk_hours), end_dt)
            chunks.append((current, chunk_end))
            current = chunk_end

        return chunks

    async def download_klines_chunk(self, symbol: str, interval: str, start_time: int, end_time: int, market_type: str) -> List[List]:
        """下载单个时间块的K线数据 / Download klines data for a single time chunk"""
        for attempt in range(self.max_retries):
            try:
                if market_type == "spot":
                    rest_assistant = await self.spot_api_factory.get_rest_assistant()
                    # 现货使用优化的klines接口 / Spot uses optimized klines endpoint
                    klines_url = self.SPOT_KLINES_URL
                    throttler = self.spot_throttler
                    throttler_limit_id = self.SPOT_KLINES_URL
                else:
                    rest_assistant = await self.futures_api_factory.get_rest_assistant()
                    # 期货使用优化的klines接口 / Futures uses optimized klines endpoint
                    klines_url = self.FUTURES_KLINES_URL
                    throttler = self.futures_throttler
                    throttler_limit_id = self.FUTURES_KLINES_URL

                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_time * 1000,
                    "endTime": end_time * 1000,
                    "limit": 1500
                }

                async with throttler.execute_task(throttler_limit_id):
                    if market_type == "spot":
                        response = await rest_assistant.execute_request(
                            url=spot_web_utils.public_rest_url(klines_url),
                            method=RESTMethod.GET,
                            params=params,
                            throttler_limit_id=throttler_limit_id
                        )
                    else:
                        response = await rest_assistant.execute_request(
                            url=futures_web_utils.public_rest_url(klines_url),
                            method=RESTMethod.GET,
                            params=params,
                            throttler_limit_id=throttler_limit_id
                        )

                # 成功时重置错误计数 / Reset error count on success
                self.consecutive_errors = 0
                self.last_success_time = time.time()
                return response

            except Exception as e:
                error_msg = str(e)

                # 处理IP封禁 / Handle IP ban
                if "418" in error_msg or "1003" in error_msg or "banned" in error_msg.lower():
                    await self._handle_rate_limit_error(error_msg, symbol)
                    continue

                # 其他错误重试 / Retry for other errors
                wait_time = min(self.base_delay * (2 ** attempt) + random.uniform(0, 10), self.max_delay)
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to download {symbol} chunk {start_time}-{end_time} after {self.max_retries} attempts: {e}")
                    return []
                else:
                    logger.warning(f"Download attempt {attempt + 1}/{self.max_retries} failed for {symbol}: {e}, retrying in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)

        return []

    def should_show_warning(self, warning_type: str) -> bool:
        """检查是否应该显示警告（避免重复）/ Check if warning should be shown (avoid duplicates)"""
        current_time = time.time()

        # 全局静默期间不显示任何警告 / No warnings during global silence period
        if current_time < self.global_silence_until:
            return False

        # 检查特定警告类型的冷却期 / Check cooldown for specific warning type
        last_warning_time = self.warning_timestamps.get(warning_type, 0)
        if current_time - last_warning_time < self.warning_cooldown:
            return False

        # 更新最后警告时间 / Update last warning time
        self.warning_timestamps[warning_type] = current_time
        return True

    async def _handle_rate_limit_error(self, error_msg: str, symbol: str):
        """处理rate limit错误和IP封禁 / Handle rate limit errors and IP bans"""
        current_time = time.time()

        if "418" in str(error_msg) or "1003" in str(error_msg) or "banned" in str(error_msg).lower():
            self.ban_detected = True
            self.last_ban_time = current_time
            self.consecutive_errors += 1

            # 设置全局静默期，避免重复警告 / Set global silence to avoid repeated warnings
            self.global_silence_until = current_time + 600  # 10分钟静默期

            # 检查是否需要显示警告 / Check if warning should be shown
            should_show = self.should_show_warning("ip_ban")

            # 解析封禁时间 / Parse ban time
            ban_until = None
            if "banned until" in str(error_msg):
                try:
                    import re
                    match = re.search(r'banned until (\d+)', str(error_msg))
                    if match:
                        ban_until_ms = int(match.group(1))
                        ban_until = ban_until_ms / 1000
                        wait_time = max(0, ban_until - current_time)

                        if should_show:
                            logger.warning(f"IP banned until {datetime.fromtimestamp(ban_until).strftime('%Y-%m-%d %H:%M:%S')}, waiting {wait_time / 60:.1f} minutes (attempt {self.consecutive_errors})")

                        await asyncio.sleep(wait_time + 60)  # 额外等待1分钟 / Wait extra 1 minute
                        self.ban_detected = False
                        self.consecutive_errors = 0  # 重置错误计数
                        return
                except Exception as parse_error:
                    logger.error(f"Failed to parse ban time: {parse_error}")

            # 如果无法解析时间，使用递增等待时间 / If unable to parse time, use incremental wait
            wait_time = min(300 + (self.consecutive_errors * 180), 1800)  # 5分钟到30分钟
            if should_show:
                logger.warning(f"Rate limit detected, waiting {wait_time} seconds... (attempt {self.consecutive_errors})")

            await asyncio.sleep(wait_time)
            self.ban_detected = False

    def save_to_csv(self, symbol: str, market_type: str, interval: str, data: List[List], start_date: str, end_date: str) -> str:
        """保存数据到CSV文件 / Save data to CSV file"""
        # 创建市场类型目录 / Create market type directory
        market_dir = self.data_dir / market_type
        market_dir.mkdir(exist_ok=True)

        # 生成文件名 / Generate filename
        filename = f"{symbol}_{interval}_{start_date}_{end_date}.csv"
        file_path = market_dir / filename

        # CSV列名 / CSV column names
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ]

        # 检查文件是否存在 / Check if file exists
        file_exists = file_path.exists()

        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 如果是新文件，写入表头 / Write header for new file
            if not file_exists:
                writer.writerow(columns)

            # 转换时间戳并写入数据 / Convert timestamps and write data
            for row in data:
                # 转换毫秒时间戳为可读格式 / Convert millisecond timestamps to readable format
                row[0] = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                row[6] = datetime.fromtimestamp(row[6] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                # 移除最后一列（忽略字段）/ Remove last column (ignore field)
                writer.writerow(row[:-1])

        return str(file_path)

    def merge_temp_files_with_existing(self, symbol: str, market_type: str, interval: str) -> int:
        """合并临时文件与已存在文件并清理 / Merge temporary files with existing file and cleanup"""
        temp_dir = self._get_temp_dir(symbol)
        temp_files = sorted(temp_dir.glob(f"*_{symbol}_{interval}_*.csv"))

        if not temp_files:
            # 清理空的临时目录 / Clean up empty temp directory
            self._cleanup_temp_dir(symbol)
            return 0

        all_data = []
        total_records = 0

        # 加载现有文件数据（如果存在）/ Load existing file data if exists
        market_dir = self.data_dir / f"{market_type}_{interval}"
        existing_file = market_dir / f"{symbol}_{interval}.csv"
        
        if existing_file.exists():
            try:
                existing_df = pd.read_csv(existing_file)
                all_data.append(existing_df)
                total_records += len(existing_df)
                logger.info(f"Loaded {len(existing_df)} existing records from {existing_file}")
            except Exception as e:
                logger.warning(f"Failed to read existing file {existing_file}: {e}")

        # 加载临时文件数据 / Load temporary file data
        for temp_file in temp_files:
            try:
                df = pd.read_csv(temp_file)
                all_data.append(df)
                total_records += len(df)
                logger.debug(f"Loaded {len(df)} records from {temp_file}")
            except Exception as e:
                logger.error(f"Failed to read temp file {temp_file}: {e}")

        final_records = 0
        if all_data:
            # 合并数据并去重，按时间排序 / Merge data, remove duplicates, and sort by time
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['open_time']).sort_values('open_time')

            # 保存最终文件 / Save final file
            final_records = len(combined_df)
            final_file = self.save_csv_from_dataframe(combined_df, symbol, market_type, interval)

            logger.info(f"Merged {symbol} data: {total_records} -> {final_records} records (removed duplicates), saved to {final_file}")

        # 清理临时文件 / Clean up temporary files
        self._cleanup_temp_dir(symbol)

        return final_records

    def merge_temp_files(self, symbol: str, market_type: str, interval: str) -> int:
        """合并临时文件并清理 / Merge temporary files and cleanup"""
        # 使用新的合并函数，支持与现有文件合并 / Use new merge function that supports merging with existing file
        return self.merge_temp_files_with_existing(symbol, market_type, interval)

    def save_csv_from_dataframe(self, df: pd.DataFrame, symbol: str, market_type: str, interval: str) -> str:
        """从DataFrame保存CSV文件 / Save CSV file from DataFrame"""
        market_dir = self.data_dir / f"{market_type}_{interval}"
        market_dir.mkdir(exist_ok=True)

        filename = f"{symbol}_{interval}.csv"
        file_path = market_dir / filename

        df.to_csv(file_path, index=False)
        return str(file_path)

    async def download_symbol_data(self, symbol: str, market_type: str, interval: str, start_dt: datetime, end_dt: datetime, force_redownload: bool = False) -> Tuple[int, int]:
        """下载单个币种的数据，支持增量下载 / Download data for a single symbol with incremental support"""
        # 检查文件是否已存在 / Check if file already exists
        start_date_str = start_dt.strftime('%Y%m%d')
        end_date_str = end_dt.strftime('%Y%m%d')

        if force_redownload:
            # 强制重新下载，使用原来的逻辑 / Force redownload, use original logic
            download_ranges = [(start_dt, end_dt)]
        else:
            # 计算需要下载的时间范围（增量逻辑）/ Calculate download ranges (incremental logic)
            download_ranges = self.calculate_download_ranges(symbol, market_type, interval, start_dt, end_dt)
            
            if not download_ranges:
                # 不需要下载新数据，返回现有文件的记录数 / No new data needed, return existing file record count
                try:
                    market_dir = self.data_dir / f"{market_type}_{interval}"
                    file_path = market_dir / f"{symbol}_{interval}.csv"
                    df = pd.read_csv(file_path)
                    logger.info(f"No download needed for {symbol}, existing file has {len(df)} records")
                    return len(df), 0
                except Exception as e:
                    logger.warning(f"Failed to read existing file {symbol}: {e}, will re-download")
                    download_ranges = [(start_dt, end_dt)]

        async with self.semaphore:  # 控制并发 / Control concurrency
            try:
                temp_dir = self._get_temp_dir(symbol)
                downloaded_records = 0
                failed_chunks = 0
                total_chunks = 0

                # 处理每个需要下载的时间范围 / Process each download range
                for range_idx, (range_start, range_end) in enumerate(download_ranges):
                    logger.info(f"Downloading range {range_idx + 1}/{len(download_ranges)} for {symbol}: {range_start.strftime('%Y-%m-%d %H:%M:%S')} to {range_end.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # 生成时间分块 / Generate time chunks
                    chunks = self.generate_chunks(range_start, range_end, chunk_hours=24)
                    total_chunks += len(chunks)

                    # 创建进度条 / Create progress bar
                    progress_bar = tqdm(
                        total=len(chunks),
                        desc=f"{symbol:>12s} R{range_idx+1}",
                        leave=True,
                        unit="chunk",
                        ncols=100
                    )

                    try:
                        for i, (chunk_start, chunk_end) in enumerate(chunks):
                            start_ts = int(chunk_start.timestamp())
                            end_ts = int(chunk_end.timestamp())

                            # 检查是否已下载 / Check if already downloaded
                            temp_filename = f"chunk_{range_idx}_{i:04d}_{symbol}_{interval}_{start_ts}_{end_ts}.csv"
                            temp_file_path = temp_dir / temp_filename

                            if temp_file_path.exists():
                                progress_bar.update(1)
                                continue

                            # 下载数据 / Download data
                            data = await self.download_klines_chunk(symbol, interval, start_ts, end_ts, market_type)

                            if data:
                                # 保存到临时文件 / Save to temporary file
                                self._save_temp_chunk(temp_file_path, data)
                                downloaded_records += len(data)
                            else:
                                failed_chunks += 1
                                logger.warning(f"Failed to download chunk {i} for {symbol}")

                            progress_bar.update(1)

                            # 短暂延迟避免请求过快 / Brief delay to avoid requests too fast
                            await asyncio.sleep(0.1)

                    finally:
                        progress_bar.close()

                # 合并临时文件与现有文件 / Merge temporary files with existing file
                final_records = self.merge_temp_files(symbol, market_type, interval)
                
                if download_ranges and final_records > 0:
                    logger.info(f"Successfully downloaded {downloaded_records} new records for {symbol}, total records: {final_records}")

                return final_records, failed_chunks

            except Exception as e:
                logger.error(f"Error downloading {symbol} data: {e}")
                return 0, 1

    def _save_temp_chunk(self, file_path: Path, data: List[List]):
        """保存临时数据块 / Save temporary data chunk"""
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]

        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)

            for row in data:
                # 转换时间戳 / Convert timestamps
                row[0] = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                row[6] = datetime.fromtimestamp(row[6] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                # 移除最后一列（忽略字段）/ Remove last column (ignore field)
                writer.writerow(row[:-1])

    async def run_downloads(self, symbols: Optional[List[str]] = None, top_volume: Optional[int] = None,
                            market_type: str = "spot", interval: str = "1h", start_time: Optional[str] = None,
                            end_time: Optional[str] = None, volume_type: str = "24h",
                            force_redownload: bool = False) -> Dict[str, int]:
        """运行下载任务 / Run download tasks"""
        try:
            logger.info("=== Binance高性能数据下载器启动 / Binance High-Performance Data Downloader Started ===")

            # 确定下载的币种 / Determine symbols to download
            if symbols:
                target_symbols = symbols
                logger.info(f"命令行参数: symbols={symbols}, top_volume={top_volume}, market_type={market_type}")
            elif top_volume:
                target_symbols = await self.get_top_volume_symbols(market_type, top_volume, volume_type)
                logger.info(f"命令行参数: symbols={symbols}, top_volume={top_volume}, market_type={market_type}")
            else:
                logger.error("必须指定symbols或top_volume参数 / Must specify either symbols or top_volume parameter")
                return {}

            if not target_symbols:
                logger.error("未找到可下载的币种 / No symbols found for download")
                return {}

            # 处理时间范围 / Handle time range
            if start_time:
                start_dt = datetime.strptime(start_time, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            else:
                start_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)

            # 处理结束时间 / Handle end time
            if end_time is None:
                end_dt = datetime.now(timezone.utc)
            else:
                end_dt = datetime.strptime(end_time, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                # 如果用户指定了具体日期，默认到当天结束 / If user specified date, default to end of day
                if end_dt.hour == 0 and end_dt.minute == 0:
                    end_dt = end_dt.replace(hour=12)  # 设置为中午12点 / Set to 12:00 PM

            # 确保结束时间有时区信息 / Ensure end time has timezone info
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)

            logger.info(f"时间范围: {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d %H:%M:%S')}, 间隔: {interval}")

            logger.info("Binance高性能数据下载器初始化完成 / Advanced downloader initialized")
            logger.info(f"最大并发数: {self.max_concurrent}, 分块大小: 24小时")

            # 创建下载任务 / Create download tasks
            tasks = []
            for symbol in target_symbols:
                task = asyncio.create_task(
                    self.download_symbol_data(symbol, market_type, interval, start_dt, end_dt, force_redownload),
                    name=f"download_{symbol}"
                )
                tasks.append((symbol, task))

            # 执行下载任务 / Execute download tasks
            results = {}
            successful_symbols = []
            failed_symbols = []

            for symbol, task in tasks:
                try:
                    final_records, failed_chunks = await task
                    results[symbol] = final_records

                    if final_records > 0:
                        successful_symbols.append(symbol)
                    else:
                        failed_symbols.append(symbol)

                except Exception as e:
                    logger.error(f"Task for {symbol} failed: {e}")
                    failed_symbols.append(symbol)
                    results[symbol] = 0

            # 保存进度 / Save progress
            self.save_progress()

            # 记录完成统计 / Log completion statistics
            total_successful = len(successful_symbols)
            total_failed = len(failed_symbols)

            logger.info("=== 下载任务完成 / Download Task Completed ===")
            logger.info(f"成功下载: {total_successful} 个交易对")
            logger.info(f"失败下载: {total_failed} 个交易对")

            if successful_symbols:
                logger.info(f"成功的交易对: {', '.join(successful_symbols)}")
            if failed_symbols:
                logger.warning(f"失败的交易对: {', '.join(failed_symbols)}")

            return results

        except Exception as e:
            logger.error(f"Download process failed: {e}", exc_info=True)
            return {}


def generate_log_filename(symbols, top_volume, market_type, start_time, end_time, interval):
    """生成日志文件名 / Generate log filename"""
    # 创建logs目录 / Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # 确定symbols部分 / Determine symbols part
    if symbols:
        if len(symbols) == 1:
            symbols_part = symbols[0]
        else:
            symbols_part = f"multi_{len(symbols)}"
    elif top_volume:
        symbols_part = f"top_volume_{top_volume}"
    else:
        symbols_part = "unknown"

    # 格式化时间字符串 / Format time strings
    start_str = start_time.replace('-', '') if isinstance(start_time, str) else start_time.strftime('%Y%m%d')
    end_str = end_time.replace('-', '') if isinstance(end_time, str) else end_time.strftime('%Y%m%d')

    # 生成文件名 / Generate filename
    filename = f"binance_{market_type}_{symbols_part}_{start_str}_{end_str}_{interval}.log"
    return logs_dir / filename


def setup_logging(log_file_path):
    """设置日志配置 / Setup logging configuration"""
    # 创建logger / Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除现有handlers / Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建formatter / Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件handler / File handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台handler / Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


async def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(description='Binance高性能数据下载器 / Binance High-Performance Data Downloader')
    parser.add_argument('--symbols', nargs='+', help='指定要下载的币种 / Specify symbols to download')
    parser.add_argument('--top-volume', type=int, help='下载成交量最高的N个币种 / Download top N volume symbols')
    parser.add_argument('--volume-type', default="24h", choices=['24h', '7d', '30d', '1y'], help='计算top-volume时使用的成交量类型 / Volume type for calculating top-volume (24h, 7d, 30d, 1y)')
    parser.add_argument('--market-type', choices=['spot', 'futures'], default='futures', help='市场类型 / Market type (default: futures)')
    parser.add_argument('--interval', default='5m', help='时间间隔 / Time interval (default: 1h). 可选: 1m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d, 3d, 1w, 1M')
    parser.add_argument('--start-time', default="2023-01-01", help='开始时间 YYYY-MM-DD / Start time YYYY-MM-DD')
    parser.add_argument('--end-time', default=None, help='结束时间 YYYY-MM-DD / End time YYYY-MM-DD')
    parser.add_argument('--api-key', default=None, help='Binance API Key (可选，用于认证API) / Binance API Key (optional, for authenticated API)')
    parser.add_argument('--api-secret', default=None, help='Binance API Secret (可选，用于认证API) / Binance API Secret (optional, for authenticated API)')
    parser.add_argument('--test-private-api', action='store_true', help='测试私有API访问（需要API密钥）/ Test private API access (requires API keys)')
    parser.add_argument('--force-redownload', action='store_true', help='强制重新下载，忽略已存在的文件 / Force re-download, ignore existing files')
    parser.add_argument('--incremental', action='store_true', default=True, help='启用增量下载模式，自动补充缺失的时间段数据 / Enable incremental download mode, automatically fill missing time periods (default: True)')
    parser.add_argument('--no-incremental', action='store_true', help='禁用增量下载模式，总是下载完整时间范围 / Disable incremental download mode, always download complete time range')

    args = parser.parse_args()

    # 验证参数 / Validate arguments
    if args.symbols and args.top_volume:  # 如果手动指定了不同的top_volume值
        print("错误: 不能同时指定 --symbols 和 --top-volume 参数 / Error: Cannot specify both --symbols and --top-volume")
        return

    # 验证API密钥参数 / Validate API key parameters
    if (args.api_key and not args.api_secret) or (args.api_secret and not args.api_key):
        print("错误: 必须同时提供 --api-key 和 --api-secret 参数 / Error: Both --api-key and --api-secret must be provided together")
        return
    
    # 处理增量下载设置 / Handle incremental download settings
    if args.no_incremental:
        args.incremental = False
        args.force_redownload = True  # 禁用增量模式时自动启用强制重下载
        logger.info("增量下载已禁用，将下载完整时间范围 / Incremental download disabled, will download complete time range")

    # 如果指定了symbols，则不使用top_volume
    if args.symbols:
        args.top_volume = None

    # 生成日志文件名 / Generate log filename
    log_file_path = generate_log_filename(
        symbols=args.symbols,
        top_volume=args.top_volume,
        market_type=args.market_type,
        start_time=args.start_time or "20230101",
        end_time=args.end_time or datetime.now().strftime('%Y%m%d'),
        interval=args.interval
    )

    # 设置日志 / Setup logging
    setup_logging(log_file_path)
    logger.info(f"日志文件: {log_file_path}")

    try:
        # 创建下载器实例 / Create downloader instance
        downloader = BinanceDataDownloader(api_key=args.api_key, api_secret=args.api_secret)

        # 如果请求测试私有API / If private API test requested
        if args.test_private_api:
            logger.info("=== 测试私有API访问 / Testing Private API Access ===")
            success = await downloader.test_private_api(args.market_type)
            if success:
                logger.info("私有API测试成功，可以访问账户信息 / Private API test successful, account access working")
            else:
                logger.error("私有API测试失败，请检查API密钥 / Private API test failed, please check API keys")
            return

        # 运行下载任务 / Run download tasks
        results = await downloader.run_downloads(
            symbols=args.symbols,
            top_volume=args.top_volume,
            market_type=args.market_type,
            interval=args.interval,
            start_time=args.start_time,
            end_time=args.end_time,
            volume_type=args.volume_type,
            force_redownload=args.force_redownload
        )
        
        if args.incremental and not args.force_redownload:
            logger.info("增量下载模式已启用，仅下载缺失的时间段数据 / Incremental download mode enabled, only downloading missing time period data")

        logger.info(f"日志文件已保存到: {log_file_path}")

        # 显示下载结果 / Display download results
        if results:
            print("\n=== 下载结果 / Download Results ===")
            for symbol, record_count in results.items():
                print(f"{symbol}: {record_count:,} 条记录 / {record_count:,} records")

    except KeyboardInterrupt:
        logger.info("用户中断下载 / User interrupted download")
    except Exception as e:
        logger.error(f"程序执行出错: {e} / Program execution error: {e}", exc_info=True)


if __name__ == "__main__":
    # 禁用其他模块的警告日志 / Disable warning logs from other modules
    logging.getLogger('hummingbot').setLevel(logging.ERROR)
    logging.getLogger('aiohttp').setLevel(logging.ERROR)
    logging.getLogger('asyncio').setLevel(logging.ERROR)

    asyncio.run(main())

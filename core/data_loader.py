import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import ccxt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceDataLoader:
    """
    Клас для завантаження та кешування OHLCV-даних з Binance через CCXT.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        data_dir: str = "data",
        cache_file: str = "btc_1m.parquet"
    ):
        self.data_dir = data_dir
        self.cache_file = os.path.join(data_dir, cache_file)
        os.makedirs(self.data_dir, exist_ok=True)

        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'adjustForTimeDifference': True}
        })

    def get_top_btc_pairs(self, top_n: int = 100) -> List[str]:
        """
        Повертає top_n пар до BTC з найбільшим об'ємом торгів.

        :param top_n: Кількість пар
        :return: Список символів пар
        """
        try:
            markets = self.exchange.load_markets()
            btc_pairs = [s for s in markets if s.endswith("/BTC")]
            sorted_pairs = sorted(
                btc_pairs,
                key=lambda x: markets[x].get('quoteVolume24h', 0),
                reverse=True
            )
            return sorted_pairs[:top_n]
        except Exception as e:
            logger.error(f"Error fetching top BTC pairs: {e}")
            return []

    def fetch_ohlcv(
        self, pair: str, timeframe: str = '1m',
        since: Optional[int] = None, limit: int = 1000
    ) -> pd.DataFrame:
        """
        Отримує OHLCV-дані для вказаної пари з Binance.

        :return: DataFrame з колонками: timestamp, open, high, low, close, volume
        """
        try:
            candles = self.exchange.fetch_ohlcv(
                symbol=pair, timeframe=timeframe, since=since, limit=limit
            )
            if not candles:
                return pd.DataFrame()

            df = pd.DataFrame(
                candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            return pd.DataFrame()

    def fetch_historical_data(
        self, pair: str, timeframe: str = '1m',
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Історичне завантаження даних для пари за вказаний період.

        :return: DataFrame з усіма OHLCV-даними за період
        """
        start_dt = pd.to_datetime(start_date) if start_date else datetime.now() - timedelta(days=30)
        end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
        total_days = (end_dt - start_dt).days + 1

        all_data = []
        current_dt = start_dt
        last_timestamp = None

        with tqdm(total=total_days, desc=f"Fetching {pair}") as pbar:
            while current_dt < end_dt:
                try:
                    since = int(current_dt.timestamp() * 1000)
                    df = self.fetch_ohlcv(pair, timeframe, since)

                    if df.empty:
                        pbar.update(total_days - pbar.n)
                        break

                    if last_timestamp is not None:
                        df = df[df.index > last_timestamp]

                    if df.empty:
                        current_dt += timedelta(days=1)
                        pbar.update(1)
                        continue

                    all_data.append(df)
                    last_timestamp = df.index[-1]

                    processed_days = (last_timestamp.to_pydatetime() - start_dt).days
                    pbar.update(processed_days - pbar.n)

                    current_dt = last_timestamp.to_pydatetime() + timedelta(minutes=1)
                    time.sleep(self.exchange.rateLimit / 1000)

                except Exception as e:
                    logger.error(f"Error during historical fetch for {pair}: {e}")
                    pbar.update(total_days - pbar.n)
                    break

        if all_data:
            return pd.concat(all_data).sort_index()
        return pd.DataFrame()

    def load_or_download_data(
        self, timeframe: str = '1m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Завантажує дані з кешу або з Binance API.

        :return: MultiIndex DataFrame: (pair, [OHLCV])
        """
        if os.path.exists(self.cache_file):
            try:
                logger.info("Loading cached data...")
                cached = pd.read_parquet(self.cache_file)
                cached_pairs = cached.columns.get_level_values(0).unique()
                if len(cached_pairs) >= top_n:
                    logger.info(f"Using cached data with {len(cached_pairs)} pairs.")
                    return cached
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        logger.info("Downloading fresh data from Binance API...")
        pairs = self.get_top_btc_pairs(top_n)
        all_data = {}
        failed = []

        for pair in tqdm(pairs, desc="Downloading pairs"):
            try:
                df = self.fetch_historical_data(pair, timeframe, start_date, end_date)
                if not df.empty:
                    df.columns = pd.MultiIndex.from_product([[pair], df.columns])
                    all_data[pair] = df
                else:
                    logger.warning(f"No data for {pair}")
            except Exception as e:
                failed.append(pair)
                logger.error(f"Error fetching data for {pair}: {e}")

            time.sleep(self.exchange.rateLimit / 1000)

        if failed:
            logger.warning(f"Failed to fetch data for: {failed}")
        if not all_data:
            raise ValueError("No data was downloaded.")

        combined = pd.concat(all_data.values(), axis=1).sort_index(axis=1)

        try:
            combined.to_parquet(self.cache_file, compression='gzip')
            logger.info(f"Data saved to cache: {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

        return combined

    def verify_data_integrity(self, data: pd.DataFrame) -> bool:
        """
        Перевіряє цілісність даних: порожні значення, некоректні ціни.

        :return: True, якщо все ок
        """
        if data.empty:
            logger.error("Data is empty.")
            return False

        missing = data.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Missing values found: {missing}")

        for pair in data.columns.get_level_values(0).unique():
            close = data[(pair, 'close')]
            if (close <= 0).any():
                logger.warning(f"Pair {pair} contains non-positive close prices.")
                return False

        return True

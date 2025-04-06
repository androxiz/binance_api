import pandas as pd
from core.data_loader import BinanceDataLoader
from core.backtester import Backtester
from strategies.sma_cross import SMACrossoverStrategy
from strategies.rsi_bb import RSIBollingerStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
import logging
import warnings
import os
from dotenv import load_dotenv


load_dotenv()

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def main(api_key: str, api_secret: str):
    """
    Основна функція для запуску системи бектестингу.

    Вона ініціалізує завантаження даних, налаштовує стратегії та запускає бектестинг для кожної стратегії.

    Параметри:
    api_key (str): API-ключ для доступу до Binance.
    api_secret (str): API-секрет для доступу до Binance.
    """
    try:
        logger.info("Starting backtesting system")
        
        # Завантаження даних
        logger.info("Loading data...")
        data_loader = BinanceDataLoader(
            api_key=api_key,
            api_secret=api_secret,
            data_dir='data',
            cache_file='btc_1m.parquet'
        )
        
        price_data = data_loader.load_or_download_data(
            timeframe='1m',
            start_date='2025-02-01',  # Явне вказання дат
            end_date='2025-02-28',
            top_n=3
        )
        logger.info(f"Data range: {price_data.index.min()} - {price_data.index.max()}")
        logger.info(f"Total pairs: {len(price_data.columns.get_level_values(0).unique())}")


        # Ініціалізація стратегій
        logger.info("Initializing strategies...")
        strategies = [
            SMACrossoverStrategy(price_data, fast_window=10, slow_window=50),
            RSIBollingerStrategy(price_data),
            VWAPReversionStrategy(price_data)
        ]

        # Запуск бектестингу
        logger.info("Running backtests...")
        backtester = Backtester()
        all_metrics = []
        
        for strategy in strategies:
            result = backtester.run_backtest(strategy)
            if result is not None:
                backtester.save_results(result, strategy.__class__.__name__)
                all_metrics.append(result['metrics'])
                logger.info(f"Completed {strategy.__class__.__name__}")

        # Порівняння стратегій
        if all_metrics:
            logger.info("Comparing strategies...")
            metrics_df = pd.DataFrame(all_metrics)
            backtester.compare_strategies(metrics_df)
            
        logger.info("Backtesting completed successfully")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    API_KEY = os.getenv("API_KEY")
    API_SECRET = os.getenv("API_SECRET")

    
    main(api_key=API_KEY, api_secret=API_SECRET)
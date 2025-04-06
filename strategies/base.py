from abc import ABC, abstractmethod
import pandas as pd
import vectorbt as vbt
from typing import Dict, Union
import logging
from core.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class StrategyBase(ABC):
    """
    Абстрактний базовий клас для реалізації торгових стратегій.

    Містить загальну логіку для обробки даних, запуску бектесту та обчислення метрик.
    """

    def __init__(self, price_data: pd.DataFrame, 
                 commission: float = 0.001, 
                 slippage: float = 0.0005):
        """
        Ініціалізація базової стратегії.

        Аргументи:
            price_data (pd.DataFrame): Історичні дані цін (OHLCV) з мультиіндексом.
            commission (float): Комісія на одну угоду.
            slippage (float): Прослизання при виконанні угоди.
        """
        self.price_data = price_data
        self.commission = commission
        self.slippage = slippage
        self.pairs = price_data.columns.get_level_values(0).unique()

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Генерація торгових сигналів.

        Повертає:
            pd.DataFrame: Сигнали для входу/виходу в позиції.
        """
        pass

    @abstractmethod
    def run_backtest(self) -> vbt.Portfolio:
        """
        Запуск бектесту.

        Повертає:
            vbt.Portfolio: Об'єкт портфеля з результатами стратегії.
        """
        pass

    def calculate_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, Union[float, int]]:
        """
        Обчислення ключових метрик для портфеля.

        Аргументи:
            portfolio (vbt.Portfolio): Портфель з результатами бектесту.

        Повертає:
            Dict[str, Union[float, int]]: Метрики стратегії (прибутковість, win-rate, drawdown тощо).
        """
        try:
            trades = portfolio.trades.records
            base_metrics = {
                'total_return': portfolio.total_return(),
                'sharpe_ratio': portfolio.sharpe_ratio(),
                'max_drawdown': portfolio.max_drawdown(),
                'win_rate': len(trades[trades['pnl'] > 0])/len(trades) if len(trades) > 0 else 0.0,
                'total_trades': len(trades),
                'exposure_time': self._calculate_exposure(portfolio)
            }

            # Метрики з MetricsCalculator
            trade_stats = MetricsCalculator.calculate_trade_duration_stats(portfolio)
            profit_factor = MetricsCalculator.calculate_profit_factor(portfolio)

            return {**base_metrics, **trade_stats, 'profit_factor': profit_factor}

        except Exception as e:
            logger.error(f"Metrics calculation error: {str(e)}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'exposure_time': 0.0,
                'avg_trade_duration': 0.0,
                'median_trade_duration': 0.0,
                'max_trade_duration': 0.0,
                'min_trade_duration': 0.0,
                'profit_factor': 0.0
            }

    def _calculate_exposure(self, portfolio: vbt.Portfolio) -> float:
        """
        Обчислення часу експозиції (середній час перебування у позиції).

        Аргументи:
            portfolio (vbt.Portfolio): Портфель з результатами бектесту.

        Повертає:
            float: Час експозиції у відносних одиницях.
        """
        try:
            positions = portfolio.positions.records
            if len(positions) == 0:
                return 0.0
            return (positions['exit_idx'] - positions['entry_idx']).mean() / len(portfolio.orders)
        except Exception as e:
            logger.error(f"Exposure calculation error: {str(e)}")
            return 0.0

    @abstractmethod
    def get_metrics(self) -> Dict[str, Union[float, int]]:
        """
        Отримання метрик стратегії після бектесту.

        Повертає:
            Dict[str, Union[float, int]]: Словник з обчисленими метриками.
        """
        pass

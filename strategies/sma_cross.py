import pandas as pd
import numpy as np
import vectorbt as vbt
from .base import StrategyBase
from ta.trend import SMAIndicator
from typing import Dict, Union

class SMACrossoverStrategy(StrategyBase):
    """
    Стратегія перехрестя SMA (Simple Moving Average).

    Генерує сигнали на основі перехрестя швидкої та повільної SMA.
    """
    
    def __init__(self, price_data: pd.DataFrame,
                 fast_window: int = 10,
                 slow_window: int = 50,
                 **kwargs):
        """
        Ініціалізація стратегії перехрестя SMA.

        Аргументи:
            price_data (pd.DataFrame): Історичні дані цін (OHLCV) з мультиіндексом.
            fast_window (int): Розмір вікна для швидкої SMA.
            slow_window (int): Розмір вікна для повільної SMA.
        """
        super().__init__(price_data, **kwargs)
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self) -> pd.DataFrame:
        """
        Генерація торгових сигналів на основі перехрестя SMA.

        Повертає:
            pd.DataFrame: Сигнали для входу та виходу в позиції (1 - купівля, -1 - продаж, 0 - без дії).
        """
        signals = pd.DataFrame(
            index=self.price_data.index,
            columns=pd.MultiIndex.from_product([self.pairs, ['signal']])
        )

        for pair in self.pairs:
            close = self.price_data[(pair, 'close')].dropna()
            
            # Обчислення швидкої та повільної SMA
            fast_sma = SMAIndicator(close, window=self.fast_window).sma_indicator()
            slow_sma = SMAIndicator(close, window=self.slow_window).sma_indicator()
            
            # Генерація сигналів на купівлю та продаж
            signals[(pair, 'signal')] = np.where(
                fast_sma > slow_sma, 1, 
                np.where(fast_sma < slow_sma, -1, 0)
            )
            
        return signals

    def run_backtest(self) -> vbt.Portfolio:
        """
        Запуск бектесту для стратегії перехрестя SMA.

        Повертає:
            vbt.Portfolio: Портфель з результатами стратегії.
        """
        signals = self.generate_signals()
        close = self.price_data.xs('close', axis=1, level=1)
        return vbt.Portfolio.from_signals(
            close=close,
            entries=signals.xs('signal', axis=1, level=1) == 1,
            exits=signals.xs('signal', axis=1, level=1) == -1,
            fees=self.commission,
            slippage=self.slippage,
            freq='1m'
        )

    def get_metrics(self) -> Dict[str, Union[float, int]]:
        """
        Отримання метрик стратегії перехрестя SMA.

        Повертає:
            Dict[str, Union[float, int]]: Словник з обчисленими метриками стратегії.
        """
        portfolio = self.run_backtest()
        metrics = self.calculate_metrics(portfolio)
        metrics.update({
            'strategy': 'SMA Crossover',
            'fast_window': self.fast_window,
            'slow_window': self.slow_window
        })
        return metrics

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Union
from .base import StrategyBase

class VWAPReversionStrategy(StrategyBase):
    """
    Стратегія повернення до VWAP.

    Генерує сигнали на основі значного відхилення ціни від VWAP.
    Відкриває позиції на купівлю, коли ціна опускається нижче VWAP, 
    і на продаж, коли ціна піднімається вище VWAP.
    """
    
    def __init__(self, price_data: pd.DataFrame, 
                 lookback_window: int = 50,
                 deviation_threshold: float = 0.01,
                 **kwargs):
        """
        Ініціалізація стратегії повернення до VWAP.

        Аргументи:
            price_data (pd.DataFrame): Історичні дані цін (OHLCV).
            lookback_window (int): Вікно для обчислення VWAP.
            deviation_threshold (float): Поріг відхилення ціни від VWAP для генерування сигналу.
        """
        super().__init__(price_data, **kwargs)
        self.lookback_window = lookback_window
        self.deviation_threshold = deviation_threshold
        
    def calculate_vwap(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Обчислення VWAP (Volume Weighted Average Price).

        Аргументи:
            high (pd.Series): Ціни високих значень.
            low (pd.Series): Ціни низьких значень.
            close (pd.Series): Ціни закриття.
            volume (pd.Series): Обсяги торгів.

        Повертає:
            pd.Series: Значення VWAP.
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).rolling(self.lookback_window).sum() / \
               volume.rolling(self.lookback_window).sum()
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Генерація торгових сигналів на основі відхилення ціни від VWAP.

        Повертає:
            pd.DataFrame: Сигнали для входу та виходу в позиції (1 - купівля, -1 - продаж, 0 - без дії).
        """
        signals = pd.DataFrame(
            index=self.price_data.index,
            columns=pd.MultiIndex.from_product([self.pairs, ['signal']])
        )
        
        for pair in self.pairs:
            high = self.price_data[(pair, 'high')]
            low = self.price_data[(pair, 'low')]
            close = self.price_data[(pair, 'close')]
            volume = self.price_data[(pair, 'volume')]
            
            # Обчислення VWAP та відхилення
            vwap = self.calculate_vwap(high, low, close, volume)
            deviation = (close - vwap) / vwap
            
            # Генерація сигналів на купівлю та продаж
            signals[(pair, 'signal')] = np.select(
                [
                    deviation < -self.deviation_threshold,  # Позиція на купівлю
                    deviation > self.deviation_threshold   # Позиція на продаж
                ],
                [1, -1],
                default=0
            )
            
        return signals
    
    def run_backtest(self) -> vbt.Portfolio:
        """
        Запуск бектесту для стратегії повернення до VWAP.

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
        Отримання метрик стратегії повернення до VWAP.

        Повертає:
            Dict[str, Union[float, int]]: Словник з обчисленими метриками стратегії.
        """
        portfolio = self.run_backtest()
        metrics = self.calculate_metrics(portfolio)
        metrics.update({
            'strategy': 'VWAP Reversion',
            'lookback_window': self.lookback_window,
            'deviation_threshold': self.deviation_threshold
        })
        return metrics

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Union
from .base import StrategyBase
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class RSIBollingerStrategy(StrategyBase):
    """
    Стратегія на основі RSI з підтвердженням за допомогою Bollinger Bands.

    Генерує сигнали на купівлю/продаж при перетині рівнів RSI та меж Bollinger Bands.
    """
    
    def __init__(self, price_data: pd.DataFrame, 
                 rsi_window: int = 14,
                 bb_window: int = 20,
                 bb_std: float = 2.0,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 **kwargs):
        """
        Ініціалізація стратегії RSI з підтвердженням за допомогою Bollinger Bands.

        Аргументи:
            price_data (pd.DataFrame): Історичні дані цін (OHLCV) з мультиіндексом.
            rsi_window (int): Розмір вікна для обчислення RSI.
            bb_window (int): Розмір вікна для обчислення Bollinger Bands.
            bb_std (float): Кількість стандартних відхилень для Bollinger Bands.
            rsi_oversold (float): Рівень перепроданості для RSI.
            rsi_overbought (float): Рівень перекупленості для RSI.
        """
        super().__init__(price_data, **kwargs)
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
    def generate_signals(self) -> pd.DataFrame:
        """
        Генерація торгових сигналів на основі RSI та Bollinger Bands.

        Повертає:
            pd.DataFrame: Сигнали для входу та виходу в позиції (1 - купівля, -1 - продаж, 0 - без дії).
        """
        signals = pd.DataFrame(
            index=self.price_data.index,
            columns=pd.MultiIndex.from_product([self.pairs, ['signal']])
        )
        
        for pair in self.pairs:
            close = self.price_data[(pair, 'close')].dropna()
            
            # Обчислення RSI
            rsi = RSIIndicator(close, window=self.rsi_window).rsi()
            
            # Обчислення Bollinger Bands
            bb = BollingerBands(close, window=self.bb_window, window_dev=self.bb_std)
            bb_lower = bb.bollinger_lband()
            bb_upper = bb.bollinger_hband()
            
            # Генерація сигналів на купівлю та продаж
            long_cond = (rsi < self.rsi_oversold) & (close <= bb_lower)
            short_cond = (rsi > self.rsi_overbought) & (close >= bb_upper)
            
            signals[(pair, 'signal')] = np.select(
                [long_cond, short_cond],
                [1, -1],
                default=0
            )
            
        return signals
    
    def run_backtest(self) -> vbt.Portfolio:
        """
        Запуск бектесту для стратегії RSI з підтвердженням по Bollinger Bands.

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
        Отримання метрик стратегії RSI з підтвердженням по Bollinger Bands.

        Повертає:
            Dict[str, Union[float, int]]: Словник з обчисленими метриками стратегії.
        """
        portfolio = self.run_backtest()
        metrics = self.calculate_metrics(portfolio)
        metrics.update({
            'strategy': 'RSI with Bollinger Bands',
            'rsi_window': self.rsi_window,
            'bb_window': self.bb_window,
            'bb_std': self.bb_std,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought
        })
        return metrics

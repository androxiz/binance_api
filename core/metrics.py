from typing import Dict

import pandas as pd
import vectorbt as vbt

class MetricsCalculator:
    """
    Калькулятор метрик для портфелів VectorBT.
    
    Містить методи для обчислення тривалості угод та фактору прибутку.
    """

    @staticmethod
    def calculate_trade_duration_stats(portfolio: vbt.Portfolio) -> Dict:
        """
        Обчислює статистику тривалості угод у хвилинах.

        Аргументи:
            portfolio (vbt.Portfolio): Портфель VectorBT з інформацією про угоди.

        Повертає:
            Dict: Словник з середньою, медіанною, максимальною та мінімальною тривалістю угод.
        """
        trades = portfolio.trades.records
        if trades.shape[0] == 0:
            return {}

        index = portfolio.wrapper.index
        if len(index) > 1 and isinstance(index[0], pd.Timestamp):
            bar_duration = (index[1] - index[0]).total_seconds() / 60  # у хвилинах
        else:
            bar_duration = 1  # запасний варіант

        durations_bars = trades['exit_idx'] - trades['entry_idx']
        durations_minutes = durations_bars * bar_duration

        return {
            'avg_trade_duration': durations_minutes.mean(),
            'median_trade_duration': float(pd.Series(durations_minutes).median()),
            'max_trade_duration': durations_minutes.max(),
            'min_trade_duration': durations_minutes.min()
        }

    @staticmethod
    def calculate_profit_factor(portfolio: vbt.Portfolio) -> float:
        """
        Обчислює фактор прибутку (Profit Factor) — відношення валового прибутку до валових збитків.

        Аргументи:
            portfolio (vbt.Portfolio): Портфель VectorBT з інформацією про угоди.

        Повертає:
            float: Значення фактору прибутку. Якщо збитків немає — повертає infinity.
        """
        trades = portfolio.trades.records_readable
        if trades.empty:
            return 0.0

        gross_profits = trades[trades['PnL'] > 0]['PnL'].sum()
        gross_losses = trades[trades['PnL'] < 0]['PnL'].sum()

        if gross_losses == 0:
            return float('inf')

        return gross_profits / abs(gross_losses)

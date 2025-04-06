import pytest
import pandas as pd
import numpy as np
from strategies.sma_cross import SMACrossoverStrategy
from strategies.rsi_bb import RSIBollingerStrategy, RSIIndicator, BollingerBands
from strategies.vwap_reversion import VWAPReversionStrategy

@pytest.fixture
def sample_price_data():
    """
    Генерує тестові дані про ціни для стратегії.

    Функція створює зразок цінових даних для трьох торгових пар
    ('BTC/USDT', 'ETH/BTC', 'LTC/BTC') на основі випадкових змін в
    цінах та обсягах торгів, створюючи DataFrame, який використовуватиметься 
    для тестування стратегій.

    Параметри:
        Немає.

    Повертає:
        pd.DataFrame: Датафрейм з цінами та обсягами для кожної пари.
    """
    date_rng = pd.date_range(start='2025-02-01', end='2025-02-03', freq='1min')
    pairs = ['BTC/USDT', 'ETH/BTC', 'LTC/BTC']
    
    data = {}
    for pair in pairs:
        np.random.seed(42)
        close = np.cumprod(1 + np.random.normal(0, 0.0001, len(date_rng)))
        high = close * 1.001
        low = close * 0.999
        volume = np.random.randint(100, 1000, len(date_rng))
        
        data[(pair, 'open')] = close * 0.999
        data[(pair, 'high')] = high
        data[(pair, 'low')] = low
        data[(pair, 'close')] = close
        data[(pair, 'volume')] = volume
    
    return pd.DataFrame(data, index=date_rng)

def test_sma_crossover_strategy(sample_price_data):
    """
    Тестує стратегію перехрестя SMA.

    Перевіряється правильність генерації сигналів, виконання бектесту 
    та обчислення метрик стратегії SMA Crossover.

    Параметри:
        sample_price_data (pd.DataFrame): Тестові дані для стратегій.

    Повертає:
        Нічого.
    """
    strategy = SMACrossoverStrategy(sample_price_data, fast_window=5, slow_window=15)
    signals = strategy.generate_signals()
    
    assert not signals.empty
    assert all(col[1] == 'signal' for col in signals.columns)
    assert set(col[0] for col in signals.columns) == {'BTC/USDT', 'ETH/BTC', 'LTC/BTC'}
    assert signals.isin([-1, 0, 1]).all().all()
    
    portfolio = strategy.run_backtest()
    assert portfolio is not None
    
    metrics = strategy.get_metrics()
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics

def test_rsi_bollinger_strategy(sample_price_data):
    """
    Тестує стратегію RSI з Bollinger Bands.

    Перевіряється правильність генерації сигналів, а також відповідність 
    сигналів умовам RSI та Bollinger Bands для кожної торгової пари.

    Параметри:
        sample_price_data (pd.DataFrame): Тестові дані для стратегій.

    Повертає:
        Нічого.
    """
    strategy = RSIBollingerStrategy(sample_price_data, rsi_window=14, bb_window=20)
    signals = strategy.generate_signals()

    assert not signals.empty
    assert all(col[1] == 'signal' for col in signals.columns)
    assert set(col[0] for col in signals.columns) == {'BTC/USDT', 'ETH/BTC', 'LTC/BTC'}
    assert signals.isin([-1, 0, 1]).all().all()

    for pair in ['BTC/USDT', 'ETH/BTC', 'LTC/BTC']:
        close = sample_price_data[(pair, 'close')].dropna()
        rsi = RSIIndicator(close, window=14).rsi()
        bb = BollingerBands(close, window=20, window_dev=2)
        bb_lower = bb.bollinger_lband()
        bb_upper = bb.bollinger_hband()

        long_cond = (rsi < 30) & (close <= bb_lower)
        short_cond = (rsi > 70) & (close >= bb_upper)

        long_signal = signals[(pair, 'signal')] == 1
        short_signal = signals[(pair, 'signal')] == -1

        assert (long_signal == long_cond).all(), f"Не співпали умови для купівлі по {pair}"
        assert (short_signal == short_cond).all(), f"Не співпали умови для продажу по {pair}"

    portfolio = strategy.run_backtest()
    assert portfolio is not None

    metrics = strategy.get_metrics()
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics


def test_vwap_reversion_strategy(sample_price_data):
    """
    Тестує стратегію VWAP Reversion.

    Перевіряється правильність генерації сигналів, обчислення VWAP та 
    відповідність сигналів умовам відхилення від VWAP для кожної торгової пари.

    Параметри:
        sample_price_data (pd.DataFrame): Тестові дані для стратегій.

    Повертає:
        Нічого.
    """
    strategy = VWAPReversionStrategy(sample_price_data, lookback_window=50, deviation_threshold=0.01)
    signals = strategy.generate_signals()

    assert not signals.empty
    assert all(col[1] == 'signal' for col in signals.columns)
    assert set(col[0] for col in signals.columns) == {'BTC/USDT', 'ETH/BTC', 'LTC/BTC'}
    assert signals.isin([-1, 0, 1]).all().all()

    # Перевірка обчислення VWAP
    for pair in ['BTC/USDT', 'ETH/BTC', 'LTC/BTC']:
        high = sample_price_data[(pair, 'high')].dropna()
        low = sample_price_data[(pair, 'low')].dropna()
        close = sample_price_data[(pair, 'close')].dropna()
        volume = sample_price_data[(pair, 'volume')].dropna()

        # Ручний розрахунок VWAP
        typical_price = (high + low + close) / 3
        vwap_manual = (typical_price * volume).rolling(50).sum() / volume.rolling(50).sum()

        # Перевірка, що обчислений VWAP співпадає з розрахунками стратегії
        vwap_strategy = strategy.calculate_vwap(high, low, close, volume)
        
        pd.testing.assert_series_equal(vwap_manual, vwap_strategy, check_exact=False)

        deviation = (close - vwap_strategy) / vwap_strategy
        long_signal = signals[(pair, 'signal')] == 1
        short_signal = signals[(pair, 'signal')] == -1

        long_cond = deviation < -0.01
        short_cond = deviation > 0.01

        assert (long_signal == long_cond).all(), f"Не співпали умови для купівлі по {pair}"
        assert (short_signal == short_cond).all(), f"Не співпали умови для продажу по {pair}"

    portfolio = strategy.run_backtest()
    assert portfolio is not None

    metrics = strategy.get_metrics()
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics

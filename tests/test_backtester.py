import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import os
import matplotlib.pyplot as plt
import vectorbt as vbt
from core.backtester import Backtester

@pytest.fixture
def backtester(tmp_path):
    """Фікстура для створення екземпляра Backtester з тимчасовою директорією."""
    results_dir = tmp_path / "results"
    return Backtester(results_dir=str(results_dir))

@pytest.fixture
def mock_portfolio():
    """Фікстура для мока портфеля vectorbt."""
    portfolio = MagicMock(spec=vbt.Portfolio)
    
    dates = pd.date_range("2023-01-01", periods=3)
    portfolio.value.return_value = pd.Series(
        [100, 110, 105],
        index=dates
    )
    
    portfolio.returns.return_value = pd.DataFrame({'asset1': [0.01, -0.02, 0.03]})
    portfolio.drawdown.return_value = pd.DataFrame({'asset1': [0.0, -0.05, -0.03]})
    return portfolio

def test_init_creates_directories(tmp_path):
    """Тест ініціалізації: перевірка створення директорій."""
    results_dir = tmp_path / "results"
    backtester = Backtester(results_dir=str(results_dir))
    
    assert os.path.exists(results_dir)
    assert os.path.exists(results_dir / "screenshots")

def test_run_backtest_success(backtester):
    """Тест успішного виконання бектесту."""
    mock_strategy = Mock()
    mock_strategy.run_backtest.return_value = "portfolio"
    mock_strategy.get_metrics.return_value = {"total_return": 0.1}
    
    result = backtester.run_backtest(mock_strategy)
    
    assert result == {"metrics": {"total_return": 0.1}, "portfolio": "portfolio"}
    mock_strategy.run_backtest.assert_called_once()

def test_run_backtest_failure(backtester, caplog):
    """Тест обробки помилки при бектесті."""
    mock_strategy = Mock()
    mock_strategy.run_backtest.side_effect = Exception("Test error")
    
    result = backtester.run_backtest(mock_strategy)
    
    assert result is None
    assert "Backtest failed: Test error" in caplog.text

def test_save_results(backtester, mock_portfolio, caplog):
    """Тест збереження результатів: запис у CSV і виклики графіків."""
    results = {
        "metrics": {"total_return": 0.1},
        "portfolio": mock_portfolio
    }
    
    with patch.object(backtester, '_plot_equity_curve') as mock_equity, \
         patch.object(backtester, '_plot_performance') as mock_perf, \
         patch.object(backtester, '_plot_drawdown') as mock_drawdown, \
         patch.object(backtester, '_plot_heatmap') as mock_heatmap:
        
        backtester.save_results(results, "TestStrategy")
        
        # Перевірка запису метрик
        metrics_path = os.path.join(backtester.results_dir, 'metrics.csv')
        assert os.path.exists(metrics_path)
        
        mock_equity.assert_called_once_with(mock_portfolio, "TestStrategy")
        mock_perf.assert_called_once_with(mock_portfolio, "TestStrategy")
        mock_drawdown.assert_called_once_with(mock_portfolio, "TestStrategy")
        mock_heatmap.assert_called_once_with(mock_portfolio, "TestStrategy")

def test_save_plot(backtester):
    """Тест збереження графіка."""
    plt.figure()
    with patch('matplotlib.pyplot.savefig') as mock_save, \
         patch('matplotlib.pyplot.close') as mock_close:
        backtester._save_plot("test_plot.png")
        
        expected_path = os.path.join(backtester.results_dir, 'screenshots', 'test_plot.png')
        mock_save.assert_called_once_with(expected_path, bbox_inches='tight', dpi=300)
        mock_close.assert_called_once()

def test_plot_equity_curve(backtester, mock_portfolio, caplog):
    """Тест побудови кривої капіталу."""
    backtester._plot_equity_curve(mock_portfolio, "TestStrategy")
    
    assert "Equity curve error" not in caplog.text
    assert os.path.exists(os.path.join(backtester.results_dir, 'screenshots', 'TestStrategy_equity.png'))

def test_compare_strategies(backtester, caplog):
    """Тест порівняння стратегій."""
    metrics_df = pd.DataFrame({
        'strategy': ['Strategy1', 'Strategy2'],
        'total_return': [0.1, 0.2],
        'sharpe_ratio': [1.5, 2.0]
    })
    
    backtester.compare_strategies(metrics_df)
    
    assert os.path.exists(os.path.join(backtester.results_dir, 'screenshots', 'strategy_comparison.png'))
    assert "No valid metrics to compare" not in caplog.text

def test_compare_strategies_empty(backtester, caplog):
    """Тест порівняння стратегій з порожніми даними."""
    backtester.compare_strategies(pd.DataFrame())
    assert "No valid metrics to compare" in caplog.text

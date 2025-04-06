import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
import vectorbt as vbt
from typing import Dict, Any

matplotlib.use('Agg')  # Використовуємо backend без віконного інтерфейсу
logger = logging.getLogger(__name__)

class Backtester:
    """
    Клас для запуску бектестів торгових стратегій, збереження результатів та візуалізації метрик.
    """

    def __init__(self, results_dir: str = 'results'):
        """
        Ініціалізація класу Backtester.

        :param results_dir: Каталог для збереження результатів бектестів.
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'screenshots'), exist_ok=True)
        self._configure_plot_styles()

    def _configure_plot_styles(self):
        """
        Налаштування стилів для побудови графіків.
        """
        plt.style.use('ggplot')
        sns.set_theme(style="whitegrid", font_scale=1.1)
        plt.rcParams.update({
            'figure.figsize': (12, 6),
            'font.family': 'DejaVu Sans',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def _save_plot(self, filename: str):
        """
        Збереження графіку у файл.

        :param filename: Назва файлу для збереження графіка.
        """
        try:
            path = os.path.join(self.results_dir, 'screenshots', filename)
            plt.tight_layout()
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Saved plot: {filename}")
        except Exception as e:
            logger.error(f"Error saving plot: {str(e)}")
            plt.close()

    def run_backtest(self, strategy) -> Dict[str, Any]:
        """
        Запуск бектесту заданої стратегії.

        :param strategy: Об'єкт стратегії з методами `run_backtest` та `get_metrics`.
        :return: Словник з портфелем і метриками або None у випадку помилки.
        """
        try:
            portfolio = strategy.run_backtest()
            return {
                'metrics': strategy.get_metrics(),
                'portfolio': portfolio
            }
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            return None

    def save_results(self, results: Dict[str, Any], strategy_name: str):
        """
        Збереження результатів бектесту та побудова графіків.

        :param results: Словник з портфелем та метриками.
        :param strategy_name: Назва стратегії.
        """
        try:
            # Збереження метрик
            metrics_file = os.path.join(self.results_dir, 'metrics.csv')
            metrics_df = pd.DataFrame([results['metrics']])

            if os.path.exists(metrics_file):
                metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
            else:
                metrics_df.to_csv(metrics_file, index=False)

            # Побудова графіків
            self._plot_equity_curve(results['portfolio'], strategy_name)
            self._plot_performance(results['portfolio'], strategy_name)
            self._plot_drawdown(results['portfolio'], strategy_name)
            self._plot_heatmap(results['portfolio'], strategy_name)

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def _plot_heatmap(self, portfolio: vbt.Portfolio, strategy_name: str):
        """
        Побудова теплової карти (heatmap) кореляції прибутків активів.

        :param portfolio: Об'єкт портфеля.
        :param strategy_name: Назва стратегії.
        """
        try:
            returns = portfolio.returns().corr()

            plt.figure()
            sns.heatmap(returns, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title(f'{strategy_name} - Heatmap of Asset Returns Correlation')
            self._save_plot(f"{strategy_name}_heatmap.png")
        except Exception as e:
            logger.error(f"Heatmap plot error: {str(e)}")

    def _plot_equity_curve(self, portfolio: vbt.Portfolio, strategy_name: str):
        """
        Побудова кривої капіталу.

        :param portfolio: Об'єкт портфеля.
        :param strategy_name: Назва стратегії.
        """
        try:
            equity = portfolio.value()

            # Обробка дат
            try:
                equity.index = pd.to_datetime(equity.index, unit='ms', errors='coerce')
                equity = equity[equity.index.notnull()]
            except Exception as e:
                logger.error(f"Date processing failed: {str(e)}")
                return

            logger.info(f"{strategy_name} date range: {equity.index.min()} - {equity.index.max()}")

            plt.figure()
            plt.plot(equity.values, linewidth=2, color='steelblue')
            plt.title(f'{strategy_name} - Equity Curve')
            plt.xlabel('Trading Periods')
            plt.ylabel('Portfolio Value')
            plt.grid(True)
            self._save_plot(f"{strategy_name}_equity.png")

        except Exception as e:
            logger.error(f"Equity curve error: {str(e)}")

    def _plot_performance(self, portfolio: vbt.Portfolio, strategy_name: str):
        """
        Побудова графіку прибутків активів.

        :param portfolio: Об'єкт портфеля.
        :param strategy_name: Назва стратегії.
        """
        try:
            returns = portfolio.returns()

            if returns.empty:
                logger.warning(f"No returns data for {strategy_name}")
                return

            total_returns = returns.sum()
            valid_returns = total_returns[~total_returns.isna()]

            plt.figure()
            valid_returns.sort_values().plot(
                kind='barh',
                color=sns.color_palette("viridis", len(valid_returns))
            )
            plt.title(f'{strategy_name} - Asset Returns')
            plt.xlabel('Total Return')
            plt.grid(True, axis='x')
            self._save_plot(f"{strategy_name}_asset_returns.png")

        except Exception as e:
            logger.error(f"Error plotting performance: {str(e)}")

    def _plot_drawdown(self, portfolio: vbt.Portfolio, strategy_name: str):
        """
        Побудова графіку просадок портфеля.

        :param portfolio: Об'єкт портфеля.
        :param strategy_name: Назва стратегії.
        """
        try:
            drawdowns = portfolio.drawdown()

            plt.figure()
            ax = plt.gca()
            palette = sns.color_palette("husl", n_colors=len(drawdowns.columns))

            for i, (pair, dd_series) in enumerate(drawdowns.items()):
                dd_series.plot(
                    alpha=0.7,
                    linewidth=1,
                    color=palette[i],
                    ax=ax,
                    label=pair
                )

            plt.title(f'{strategy_name} - Drawdown by Asset')
            plt.xlabel('Date')
            plt.ylabel('Drawdown, %')
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self._save_plot(f"{strategy_name}_drawdown.png")

        except Exception as e:
            logger.error(f"Drawdown plot error: {str(e)}")

    def compare_strategies(self, metrics_df: pd.DataFrame):
        """
        Порівняння стратегій за ключовими метриками з побудовою графіків.

        :param metrics_df: DataFrame з метриками для кожної стратегії.
        """
        try:
            if metrics_df.empty or not isinstance(metrics_df, pd.DataFrame):
                logger.warning("No valid metrics to compare")
                return

            numeric_metrics = [
                'total_return', 'sharpe_ratio', 'max_drawdown',
                'win_rate', 'profit_factor', 'avg_trade_duration',
                'median_trade_duration', 'max_trade_duration',
                'min_trade_duration'
            ]

            # Перетворення типів для числових метрик
            for metric in numeric_metrics:
                if metric in metrics_df.columns:
                    metrics_df[metric] = pd.to_numeric(metrics_df[metric], errors='coerce')

            valid_metrics = [
                m for m in numeric_metrics
                if m in metrics_df.columns and not metrics_df[m].isnull().all()
            ]

            if not valid_metrics:
                logger.warning("No valid metrics available for comparison")
                return

            # Побудова сітки графіків
            ncols = min(len(valid_metrics), 2)
            nrows = (len(valid_metrics) + 1) // 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
            axes = axes.flatten() if len(valid_metrics) > 1 else [axes]

            for i, metric in enumerate(valid_metrics):
                ax = axes[i]
                try:
                    plot_data = metrics_df[['strategy', metric]].dropna()
                    plot_data[metric] = plot_data[metric].astype(float)

                    sns.barplot(
                        x=metric,
                        y='strategy',
                        data=plot_data.sort_values(metric),
                        ax=ax,
                        palette='viridis',
                        orient='h'
                    )
                    ax.set_title(f"{metric.replace('_', ' ').title()}")

                    if 'duration' in metric:
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f} min"))

                    ax.grid(True, alpha=0.3)

                    if metric == 'win_rate':
                        ax.set_xlim(0, 1)
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

                    if metric == 'profit_factor':
                        ax.set_xlim(left=0)

                except Exception as e:
                    logger.error(f"Plot error for {metric}: {str(e)}")
                    continue

            # Видалення зайвих вісей, якщо графіків менше, ніж місць
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            self._save_plot('strategy_comparison.png')

        except Exception as e:
            logger.error(f"Strategy comparison failed: {str(e)}")

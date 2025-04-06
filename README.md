# Проект: Бектестинг торгових стратегій

Цей проект націлений на бектестинг трьох торгових стратегій на основі даних з Binance. Він включає в себе завантаження даних, застосування стратегій (SMA Crossover, RSI + Bollinger Bands, VWAP Reversion) та аналіз результатів. Мета проекту - перевірити ефективність кожної стратегії за допомогою бектестингу та порівняння результатів.

## Інструкція по запуску

### 1. Клонуйте репозиторій

```bash
git clone https://github.com/androxiz/binance_api.git
cd binance_api
```

### 2. Встановіть залежності
```bash
python -m venv venv
venv/Scripts/activate  (для Windows)
pip install -r requirements.txt
```

### 3. Налаштуйте змінні середовища
Створіть файл .env у корні проекту та додайте ваш API-ключ та секрет для доступу до Binance
Приклад .env файлу:
```bash
API_KEY='your_api_key'
API_SECRET='your_api_secret'
```

### 4. Запуск проекту
Запустіть проект через main.py


### 5. Тестування
```bash
pytest -v -s
```
---
Програма завантажить історичні дані, застосує стратегії та проведе бектестинг. Результати будуть збережені в директорії results.

---
Для тестування застосунку я встановив завантаження 3х валютних пар. За необхідності можна змінити це значення в main.py:
```bash
price_data = data_loader.load_or_download_data(
            timeframe='1m',
            start_date='2025-02-01',
            end_date='2025-02-28',
            top_n=3 # Вкажіть кількість валютних пар (Максимум 100)
        )
```

---

## Опис стратегій
### 1. SMA Crossover Strategy
Ця стратегія використовує два ковзаючих середніх (SMA) з різними періодами: швидке (fast) та повільне (slow). Коли швидке SMA перетинає повільне SMA знизу вгору, генерується сигнал на покупку, коли зверху вниз — сигнал на продаж.
### 2. RSI + Bollinger Bands Strategy
Ця стратегія використовує індикатор RSI та Банди Боллінджера для генерування сигналів. Сигнал на покупку генерується, коли RSI знаходиться нижче 30 і ціна закривається нижче нижньої межі Боллінджера. Сигнал на продаж генерується, коли RSI вище 70 і ціна закривається вище верхньої межі Боллінджера.

### 3. VWAP Reversion Strategy 
Ця стратегія використовує VWAP (середня ціна за обсягом) для визначення відхилень від середнього рівня. Коли ціна відхиляється від VWAP на більше ніж заданий поріг, генерується сигнал на покупку або продаж. Сигнал на покупку виникає, коли відхилення менше -0.01, а на продаж — коли відхилення більше 0.01.

---
## Висновки по результатах
Після запуску бектестингу для кожної стратегії, результати будуть збережені в директорії results. Для кожної стратегії буде згенеровано CSV файл з результатами бектестингу, а також графіки, які можна використовувати для порівняння ефективності стратегій.
Програма також проводить порівняння стратегій за допомогою метрик, таких як загальний прибуток. Це дозволяє зрозуміти, яка з стратегій має найкращу продуктивність на даних за обраний період.

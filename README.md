# Smart Bookshelf

Прототип книжного рекомендательного сервиса на датасете Goodbooks-10k.
Реализует и сравнивает четыре подхода: popularity baseline, content-based
(TF-IDF по названиям и тегам), item-based collaborative filtering и SVD
матричное разложение. Качество оценивается по Precision@K, Recall@K и nDCG@K
на отложенной выборке.

## Структура

- `archive/` — исходные CSV (в .gitignore)
- `src/` — код: data_loader, eda, metrics, модели
- `scripts/` — тонкие оркестраторы
- `outputs/figures` — графики EDA
- `outputs/metrics` — сводные таблицы метрик
- `notebooks/` — финальный отчёт

## Пайплайн

1. `python src/data_loader.py` — профайлинг датасета
2. `python scripts/run_eda.py` — графики EDA в `outputs/figures`
3. Модели: Popularity → Content-Based → Item-CF → SVD
4. Оценка: P@K, R@K, nDCG@K на holdout
5. Гибрид + выводы

## Запуск

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\data_loader.py
python scripts\run_eda.py
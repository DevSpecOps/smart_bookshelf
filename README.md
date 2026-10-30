# Smart Bookshelf

ÐŸÑ€Ð¾Ñ‚Ð¾Ñ‚Ð¸Ð¿ ÐºÐ½Ð¸Ð¶Ð½Ð¾Ð³Ð¾ Ñ€ÐµÐºÐ¾Ð¼ÐµÐ½Ð´Ð°Ñ‚ÐµÐ»ÑŒÐ½Ð¾Ð³Ð¾ ÑÐµÑ€Ð²Ð¸ÑÐ° Ð½Ð° Ð´Ð°Ñ‚Ð°ÑÐµÑ‚Ðµ Goodbooks-10k.
Ð ÐµÐ°Ð»Ð¸Ð·ÑƒÐµÑ‚ Ð¸ ÑÑ€Ð°Ð²Ð½Ð¸Ð²Ð°ÐµÑ‚ Ñ‡ÐµÑ‚Ñ‹Ñ€Ðµ Ð¿Ð¾Ð´Ñ…Ð¾Ð´Ð°: popularity baseline, content-based
(TF-IDF Ð¿Ð¾ Ð½Ð°Ð·Ð²Ð°Ð½Ð¸ÑÐ¼ Ð¸ Ñ‚ÐµÐ³Ð°Ð¼), item-based collaborative filtering Ð¸ SVD
Ð¼Ð°Ñ‚Ñ€Ð¸Ñ‡Ð½Ð¾Ðµ Ñ€Ð°Ð·Ð»Ð¾Ð¶ÐµÐ½Ð¸Ðµ. ÐšÐ°Ñ‡ÐµÑÑ‚Ð²Ð¾ Ð¾Ñ†ÐµÐ½Ð¸Ð²Ð°ÐµÑ‚ÑÑ Ð¿Ð¾ Precision@K, Recall@K Ð¸ nDCG@K
Ð½Ð° Ð¾Ñ‚Ð»Ð¾Ð¶ÐµÐ½Ð½Ð¾Ð¹ Ð²Ñ‹Ð±Ð¾Ñ€ÐºÐµ.

## Ð¡Ñ‚Ñ€ÑƒÐºÑ‚ÑƒÑ€Ð°

- `archive/` â€” Ð¸ÑÑ…Ð¾Ð´Ð½Ñ‹Ðµ CSV (Ð² .gitignore)
- `src/` â€” ÐºÐ¾Ð´: data_loader, eda, metrics, Ð¼Ð¾Ð´ÐµÐ»Ð¸
- `scripts/` â€” Ñ‚Ð¾Ð½ÐºÐ¸Ðµ Ð¾Ñ€ÐºÐµÑÑ‚Ñ€Ð°Ñ‚Ð¾Ñ€Ñ‹
- `outputs/figures` â€” Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¸ EDA
- `outputs/metrics` â€” ÑÐ²Ð¾Ð´Ð½Ñ‹Ðµ Ñ‚Ð°Ð±Ð»Ð¸Ñ†Ñ‹ Ð¼ÐµÑ‚Ñ€Ð¸Ðº
- `notebooks/` â€” Ñ„Ð¸Ð½Ð°Ð»ÑŒÐ½Ñ‹Ð¹ Ð¾Ñ‚Ñ‡Ñ‘Ñ‚

## ÐŸÐ°Ð¹Ð¿Ð»Ð°Ð¹Ð½

1. `python src/data_loader.py` â€” Ð¿Ñ€Ð¾Ñ„Ð°Ð¹Ð»Ð¸Ð½Ð³ Ð´Ð°Ñ‚Ð°ÑÐµÑ‚Ð°
2. `python scripts/run_eda.py` â€” Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¸ EDA Ð² `outputs/figures`
3. ÐœÐ¾Ð´ÐµÐ»Ð¸: Popularity â†’ Content-Based â†’ Item-CF â†’ SVD
4. ÐžÑ†ÐµÐ½ÐºÐ°: P@K, R@K, nDCG@K Ð½Ð° holdout
5. Ð“Ð¸Ð±Ñ€Ð¸Ð´ + Ð²Ñ‹Ð²Ð¾Ð´Ñ‹

## Ð—Ð°Ð¿ÑƒÑÐº

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\data_loader.py
python scripts\run_eda.py

## Данные

Источник — Kaggle-датасет `zygmunt/goodbooks-10k` (как указано в задании).
`archive/books.csv` использует схему: `book_id` — локальный ID (1…10000),
`goodreads_book_id` — глобальный ID Goodreads. `ratings.book_id` ссылается
на `books.book_id`, `book_tags.goodreads_book_id` — на `books.goodreads_book_id`.

`scripts/download_data.py` скачивает пять CSV в `archive/` и проверяет, что
`ratings.csv` содержит 981 756 строк (каноническая версия). Если `archive/`
уже наполнен, загрузка пропускается.

## Оценка

`src/evaluate.py` выполняет полный цикл offline-оценки:

1. Дедупликация `(user_id, book_id)` — 981 756 → 979 478 оценок (2 278 дубликатов агрегируются средним).
2. Per-user split 80/20 (`random_state=42`): из каждого пользователя с ≥5 оценками 20% уходят в тест.
3. Обучение пяти моделей: `popularity_bayes`, `popularity_count`, `content_based`, `item_cf`, `svd`.
4. Генерация top-N рекомендаций для случайной выборки 5 000 пользователей (все имеют ≥1 релевантную оценку в тесте).
5. Метрики P@K, R@K, nDCG@K для K ∈ {5, 10, 20} × threshold ∈ {4, 5}.
6. Покрытие каталога `coverage@10` для каждой модели.

Артефакты: `outputs/metrics/summary.csv`, `outputs/metrics/coverage.csv`,
`outputs/figures/09_model_comparison_thr4.png`, `09_model_comparison_thr5.png`,
`10_coverage.png`.

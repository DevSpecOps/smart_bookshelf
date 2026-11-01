# Smart Bookshelf

Prototype of a book recommender system built on the Goodbooks-10k dataset.

Compares four approaches: popularity baseline, content-based (TF-IDF over titles and tags), item-based collaborative filtering, and SVD matrix factorization. Quality is measured with Precision@K, Recall@K and nDCG@K on a held-out per-user split.

## Structure

- `archive/` - raw CSV files (gitignored)
- `src/` - data_loader, eda, metrics, models
- `scripts/` - thin orchestrators and `run_all.py`
- `outputs/figures` - EDA and evaluation plots
- `outputs/metrics` - metric tables
- `notebooks/` - final report (executed)

## Quick start

Local:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/run_all.py

On Windows use `.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

Make:

    make all

Targets: `data`, `eda`, `eval`, `notebook`, `clean`. `make all` is equivalent to `python scripts/run_all.py`.

Docker:

    docker build -t smart-bookshelf .
    docker run --rm -v "$(pwd)/outputs:/app/outputs" smart-bookshelf

The container downloads the dataset, generates EDA figures, trains all models, computes metrics, and executes the report notebook. Figures and metrics are written to `outputs/` on the host.

## Pipeline

1. `python scripts/download_data.py` - download data into `archive/`
2. `python scripts/run_eda.py` - EDA plots in `outputs/figures`
3. `python src/evaluate.py` - train models and compute offline metrics
4. `jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb` - build the executed report

## Models

- `popularity` - top-N by Bayesian-smoothed mean.
- `content_based` - TF-IDF over titles and tags, cosine similarity.
- `item_cf` - item-based collaborative filtering (sparse cosine, top-K neighbours).
- `svd` - matrix factorization (surprise SVD, 100 factors).
- `hybrid` - reciprocal rank fusion of content, CF and SVD.

## Data

Source: Kaggle `zygmunt/goodbooks-10k` (as required by the assignment).

`archive/books.csv` uses the schema: `book_id` is the local ID (1..10000), `goodreads_book_id` is the global Goodreads ID. `ratings.book_id` references `books.book_id`, and `book_tags.goodreads_book_id` references `books.goodreads_book_id`.

`scripts/download_data.py` downloads five CSVs into `archive/` and verifies that `ratings.csv` has exactly 981,756 rows (canonical version). If `archive/` is already populated, downloads are skipped.

## Evaluation protocol

`src/evaluate.py` performs the full offline evaluation cycle:

1. Deduplicate `(user_id, book_id)` - 981,756 -> 979,478 ratings (2,278 duplicates aggregated by mean).
2. Per-user 80/20 split with `random_state=42`.
3. Train five models on the train split.
4. Generate top-N recommendations for a random sample of 5,000 users (each with at least one relevant rating in the test set).
5. Compute P@K, R@K, nDCG@K for K in {5, 10, 20} and threshold in {4, 5}.
6. Compute catalog coverage@10 for each model.

Artifacts: `outputs/metrics/summary.csv`, `outputs/metrics/coverage.csv`, `outputs/figures/09_model_comparison_thr4.png`, `outputs/figures/09_model_comparison_thr5.png`, `outputs/figures/10_coverage.png`, `outputs/figures/11_precision_coverage.png`.

## Results (threshold >= 4, k = 10)

| Model            | Precision@10 | Recall@10 | nDCG@10 | Coverage@10 |
|------------------|--------------|-----------|---------|-------------|
| hybrid           | 0.0476       | 0.1667    | 0.1169  | 0.8292      |
| content_based    | 0.0457       | 0.1432    | 0.1154  | 0.6733      |
| item_cf          | 0.0197       | 0.0739    | 0.0512  | 0.9360      |
| popularity_bayes | 0.00062      | 0.0018    | 0.0012  | 0.0016      |
| svd              | 0.00044      | 0.0013    | 0.0008  | 0.0394      |

Notes:

- Hybrid (RRF fusion) wins on all ranking metrics: +16% recall vs content-based.
- SVD minimises RMSE, not ranking quality - poor top-N results are expected without a ranking loss (BPR / WARP / ALS-implicit).
- Coverage and precision trade off strongly: item_cf is the most diverse model (0.936), popularity the least (0.0016).

## Reproducibility

- `.python-version` pins Python 3.11.
- `requirements.lock.txt` records exact installed versions.
- `random_state=42` is used in the per-user split and in SVD.

## Report

`notebooks/report.ipynb` contains the full analysis in Russian: EDA, data problems, model demos, offline evaluation, Item-CF complexity analysis, conclusions, and improvement ideas.
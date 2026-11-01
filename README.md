# Smart Bookshelf

Book recommender prototype on Goodbooks-10k (Kaggle). Compares popularity baseline, content-based (TF-IDF over titles and tags), item-based collaborative filtering, SVD matrix factorization, and a hybrid reciprocal-rank-fusion recommender. Quality is measured with P@K, R@K and nDCG@K on a held-out per-user split.

## Quick start

### Option 1 - Docker (recommended, works on any OS)

    docker build -t smart-bookshelf .
    docker run --rm -v "$(pwd)/outputs:/app/outputs" smart-bookshelf

The container installs pinned dependencies, downloads the dataset, trains all models, computes offline metrics, and executes the report notebook. Figures and metric tables land in `outputs/` on the host.

### Option 2 - Native (Python 3.11 or 3.12, C++ compiler required)

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/run_all.py

On Windows use `.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

`scikit-surprise` (used for SVD) needs a C++ toolchain on native installs: MSVC Build Tools on Windows, Xcode Command Line Tools on macOS, `build-essential` + `python3-dev` on Linux. Python 3.13 has no prebuilt wheels for `scikit-surprise` yet - use Docker in that case.

## Pipeline

    scripts/download_data.py      -> archive/
    scripts/run_eda.py            -> outputs/figures/01..08_*.png
    src/evaluate.py               -> outputs/metrics/summary.csv, coverage.csv
    scripts/build_notebook.py     -> notebooks/report.ipynb
    jupyter nbconvert --execute --inplace notebooks/report.ipynb

`python scripts/run_all.py` runs the full sequence. `make all` is an alias for the same.

## Models

- `popularity` - top-N by Bayesian-smoothed mean.
- `content_based` - TF-IDF over titles and tags, cosine similarity.
- `item_cf` - item-based collaborative filtering (sparse cosine, top-K neighbours).
- `svd` - biased matrix factorization (`surprise.SVD`, 100 factors).
- `hybrid` - reciprocal rank fusion of content, CF and SVD.

## Data

Source: Kaggle `zygmunt/goodbooks-10k`.

`scripts/download_data.py` downloads five CSVs into `archive/` and verifies that `ratings.csv` contains exactly 981,756 rows (canonical version). If `archive/` is already populated, downloads are skipped.

## Results (threshold >= 4, K = 10)

| Model            | P@10    | R@10    | nDCG@10 | Coverage@10 |
|------------------|---------|---------|---------|-------------|
| hybrid           | 0.0476  | 0.1667  | 0.1169  | 0.8292      |
| content_based    | 0.0457  | 0.1432  | 0.1154  | 0.6733      |
| item_cf          | 0.0197  | 0.0739  | 0.0512  | 0.9360      |
| popularity_bayes | 0.00062 | 0.0018  | 0.0012  | 0.0016      |
| svd              | 0.00044 | 0.0013  | 0.0008  | 0.0394      |

## Reproducibility

- `random_state=42` is used in the per-user split and in SVD.
- Docker image pins Python 3.11 and installs exactly `requirements.txt`.
- Native installs work on Python 3.11 and 3.12 with a C++ compiler for `scikit-surprise`.

## See also

- `notebooks/report.ipynb` - full analysis in Russian: EDA, data problems, model demos, offline evaluation, Item-CF complexity, conclusions, and improvement ideas.
- `outputs/metrics/summary.csv` - all metrics for K in {5, 10, 20} and threshold in {4, 5}.
- `outputs/figures/` - EDA and evaluation plots.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import SEED, K_VALUES, RELEVANCE_THRESHOLDS, MIN_RATINGS_POPULARITY
from data_loader import load_raw, PROJECT_ROOT
from metrics import precision_at_k, recall_at_k, ndcg_at_k
from popularity import PopularityRecommender
from content_based import ContentBasedRecommender
from item_cf import ItemBasedCF
from svd_model import SVDRecommender
from hybrid import HybridRecommender

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

MAX_K = max(K_VALUES)
N_EVAL_USERS = 5000


def per_user_split(ratings, test_size=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, group in ratings.groupby("user_id", sort=False):
        idx = group.index.to_numpy()
        if len(idx) < 5:
            train_parts.append(idx)
            continue
        n_test = max(1, int(round(len(idx) * test_size)))
        perm = rng.permutation(len(idx))
        test_parts.append(idx[perm[:n_test]])
        train_parts.append(idx[perm[n_test:]])
    train = ratings.loc[np.concatenate(train_parts)].reset_index(drop=True)
    test = ratings.loc[np.concatenate(test_parts)].reset_index(drop=True)
    return train, test


def build_models(train, books, book_tags, tags):
    models = {}
    models["popularity_bayes"] = PopularityRecommender(min_ratings=MIN_RATINGS_POPULARITY, score="bayesian").fit(train)
    models["popularity_count"] = PopularityRecommender(min_ratings=MIN_RATINGS_POPULARITY, score="count").fit(train)
    models["content_based"] = ContentBasedRecommender().fit(books, book_tags, tags, train)
    models["item_cf"] = ItemBasedCF().fit(train)
    models["svd"] = SVDRecommender().fit(train)
    hybrid = HybridRecommender(
        content_model=models["content_based"],
        cf_model=models["item_cf"],
        svd_model=models["svd"],
        popularity_model=models["popularity_bayes"],
    ).fit(train)
    models["hybrid"] = hybrid
    return models


def generate_recs(models, eligible):
    cache = {}
    for name, model in models.items():
        print(f"generating recs: {name}")
        cache[name] = {int(uid): model.recommend(int(uid), MAX_K) for uid in eligible}
    return cache


def compute_metrics(recs_cache, eligible, test):
    relevance = {thr: test[test["rating"] >= thr].groupby("user_id")["book_id"].apply(set).to_dict()
                 for thr in RELEVANCE_THRESHOLDS}
    rows = []
    for name, cache in recs_cache.items():
        for thr in RELEVANCE_THRESHOLDS:
            rel_map = relevance[thr]
            users_thr = [u for u in eligible if rel_map.get(int(u))]
            if not users_thr:
                continue
            for k in K_VALUES:
                ps, rs, nds = [], [], []
                for uid in users_thr:
                    recs = cache.get(int(uid), [])
                    rel = rel_map[int(uid)]
                    ps.append(precision_at_k(recs, rel, k))
                    rs.append(recall_at_k(recs, rel, k))
                    nds.append(ndcg_at_k(recs, rel, k))
                rows.append({"model": name, "threshold": thr, "k": k,
                             "precision": float(np.mean(ps)),
                             "recall": float(np.mean(rs)),
                             "ndcg": float(np.mean(nds)),
                             "n_users": len(users_thr)})
    return pd.DataFrame(rows)


def compute_coverage(recs_cache, n=10):
    rows = []
    for name, cache in recs_cache.items():
        exposed = set()
        for recs in cache.values():
            exposed.update(recs[:n])
        rows.append({"model": name, "coverage_at_10": len(exposed) / 10000.0})
    return pd.DataFrame(rows)


def plot_comparison(summary, threshold):
    subset = summary[summary["threshold"] == threshold]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["precision", "recall", "ndcg"]):
        sns.barplot(data=subset, x="k", y=metric, hue="model", ax=ax)
        ax.set_title(f"{metric.capitalize()}@K (threshold>={threshold})")
        ax.set_xlabel("K")
        ax.set_ylabel(metric)
        ax.legend(title="model", fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / f"09_model_comparison_thr{threshold}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def plot_coverage(cov_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=cov_df, x="model", y="coverage_at_10", ax=ax, color="teal")
    ax.set_title("Catalog coverage@10")
    ax.set_xlabel("model")
    ax.set_ylabel("coverage@10")
    for i, row in enumerate(cov_df.itertuples()):
        ax.text(i, row.coverage_at_10 + 0.01, f"{row.coverage_at_10:.4f}", ha="center", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = FIG_DIR / "10_coverage.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def plot_precision_coverage(summary, cov_df, k=10, threshold=4):
    subset = summary[(summary["threshold"] == threshold) & (summary["k"] == k)]
    merged = subset.merge(cov_df, on="model")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=merged, x="coverage_at_10", y="precision", hue="model", s=140, ax=ax)
    for row in merged.itertuples():
        ax.annotate(row.model, (row.coverage_at_10, row.precision),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_title(f"Precision@{k} vs coverage@{k} (threshold>={threshold})")
    ax.set_xlabel(f"coverage@{k}")
    ax.set_ylabel(f"precision@{k}")
    fig.tight_layout()
    path = FIG_DIR / "11_precision_coverage.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def main():
    frames = load_raw()
    ratings = frames["ratings"].groupby(["user_id", "book_id"], as_index=False)["rating"].mean()
    print(f"after dedup: {len(ratings)} rows (was {len(frames['ratings'])})")

    train, test = per_user_split(ratings)
    print(f"train: {len(train)} test: {len(test)}")

    models = build_models(train, frames["books"], frames["book_tags"], frames["tags"])

    min_thr = min(RELEVANCE_THRESHOLDS)
    eligible = test[test["rating"] >= min_thr]["user_id"].unique()
    rng = np.random.default_rng(SEED)
    if len(eligible) > N_EVAL_USERS:
        eligible = rng.choice(eligible, size=N_EVAL_USERS, replace=False)
    print(f"evaluating on {len(eligible)} users")

    recs_cache = generate_recs(models, eligible)
    summary = compute_metrics(recs_cache, eligible, test)
    summary.to_csv(METRICS_DIR / "summary.csv", index=False)

    cov_df = compute_coverage(recs_cache, n=10)
    cov_df.to_csv(METRICS_DIR / "coverage.csv", index=False)

    for thr in RELEVANCE_THRESHOLDS:
        plot_comparison(summary, thr)
    plot_coverage(cov_df)
    plot_precision_coverage(summary, cov_df, k=10, threshold=4)

    print("\nSummary (threshold=4, k=10):")
    print(summary[(summary["threshold"] == 4) & (summary["k"] == 10)].to_string(index=False))
    print("\nCoverage@10:")
    print(cov_df.to_string(index=False))


if __name__ == "__main__":
    main()

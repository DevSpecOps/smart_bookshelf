import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_raw, PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def _summary_stats(ratings):
    dist = ratings["rating"].value_counts(normalize=True).sort_index()
    n_users = ratings["user_id"].nunique()
    n_books = ratings["book_id"].nunique()
    n = len(ratings)
    density = n / (n_users * n_books)
    per_user = ratings.groupby("user_id").size()
    per_book = ratings.groupby("book_id").size()
    print("\nEDA summary")
    print(f"  interactions         : {n:,}")
    print(f"  users / books        : {n_users:,} / {n_books:,}")
    print(f"  density              : {density:.6f}  (sparsity {1 - density:.6f})")
    print(f"  mean rating          : {ratings['rating'].mean():.4f}")
    print(f"  share >=4 / >=5      : {dist.loc[4:].sum():.4f} / {dist.loc[5]:.4f}")
    print(f"  ratings per user     : p50={per_user.median():.0f} p90={per_user.quantile(0.90):.0f} p99={per_user.quantile(0.99):.0f}")
    print(f"  ratings per book     : p50={per_book.median():.0f} p90={per_book.quantile(0.90):.0f} p99={per_book.quantile(0.99):.0f}")
    print(f"  users with <=5       : {(per_user <= 5).sum():,} ({(per_user <= 5).mean()*100:.1f}%)")


def plot_rating_distribution(ratings):
    counts = ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="steelblue")
    ax.set_title("Rating distribution")
    ax.set_xlabel("rating")
    ax.set_ylabel("count")
    _save(fig, "01_rating_distribution.png")


def plot_rating_cdf(ratings):
    sorted_ratings = np.sort(ratings["rating"].values)
    cdf = np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_ratings, cdf, drawstyle="steps-post", color="crimson")
    ax.set_title("CDF of ratings")
    ax.set_xlabel("rating")
    ax.set_ylabel("cumulative share")
    ax.set_xticks([1, 2, 3, 4, 5])
    for thr in (4, 5):
        share = (ratings["rating"] >= thr).mean()
        ax.axvline(thr, color="gray", linestyle="--", linewidth=0.8)
        ax.text(thr + 0.03, 0.08, f">={thr}: {share*100:.1f}%", color="gray", fontsize=9)
    _save(fig, "02_rating_cdf.png")


def plot_relevance_by_threshold(ratings):
    rows = []
    for thr in (4, 5):
        rows.append((f"rating >= {thr}", (ratings["rating"] >= thr).mean()))
    labels = [r[0] for r in rows]
    values = [r[1] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=labels, y=values, ax=ax, color="mediumseagreen")
    ax.set_title("Share of relevant interactions by threshold")
    ax.set_xlabel("threshold")
    ax.set_ylabel("share, %")
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    _save(fig, "03_relevance_thresholds.png")


def plot_user_activity(ratings):
    per_user = ratings.groupby("user_id").size().sort_values(ascending=False).values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(per_user) + 1), per_user, color="darkorange")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("User activity: rank vs ratings count (log-log)")
    ax.set_xlabel("user rank")
    ax.set_ylabel("ratings count")
    _save(fig, "04_user_activity_loglog.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_user, bins=60, color="darkorange", edgecolor="black")
    ax.set_yscale("log")
    ax.set_title("Ratings per user histogram")
    ax.set_xlabel("ratings per user")
    ax.set_ylabel("users")
    _save(fig, "05_user_activity_hist.png")


def plot_book_popularity(ratings):
    per_book = ratings.groupby("book_id").size().sort_values(ascending=False).values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(per_book) + 1), per_book, color="seagreen")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Book popularity: rank vs ratings count (log-log)")
    ax.set_xlabel("book rank")
    ax.set_ylabel("ratings count")
    _save(fig, "06_book_popularity_loglog.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_book, bins=60, color="seagreen", edgecolor="black")
    ax.set_yscale("log")
    ax.set_title("Ratings per book histogram")
    ax.set_xlabel("ratings per book")
    ax.set_ylabel("books")
    _save(fig, "07_book_popularity_hist.png")


def plot_top_tags(book_tags, tags, top_n=20):
    merged = book_tags.merge(tags, on="tag_id")
    agg = merged.groupby("tag_name")["count"].sum().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(x=agg.values, y=agg.index, ax=ax, color="mediumpurple")
    ax.set_title(f"Top-{top_n} tags by total count")
    ax.set_xlabel("total count")
    ax.set_ylabel("tag")
    _save(fig, "08_top_tags.png")


def main():
    frames = load_raw()
    _summary_stats(frames["ratings"])
    plot_rating_distribution(frames["ratings"])
    plot_rating_cdf(frames["ratings"])
    plot_relevance_by_threshold(frames["ratings"])
    plot_user_activity(frames["ratings"])
    plot_book_popularity(frames["ratings"])
    plot_top_tags(frames["book_tags"], frames["tags"])


if __name__ == "__main__":
    main()

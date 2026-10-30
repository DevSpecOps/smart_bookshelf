import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "archive"

FILES = {
    "books": "books.csv",
    "ratings": "ratings.csv",
    "tags": "tags.csv",
    "book_tags": "book_tags.csv",
    "to_read": "to_read.csv",
}


def load_raw(data_dir=None):
    data_dir = data_dir or DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")
    frames = {}
    for name, fname in FILES.items():
        path = data_dir / fname
        if not path.exists():
            print(f"[WARN] missing: {path}", file=sys.stderr)
            continue
        frames[name] = pd.read_csv(path)
    return frames


def profile_frame(name, df, n=3):
    sep = "=" * 88
    print(f"\n{sep}\n[{name}] shape={df.shape}  memory={df.memory_usage(deep=True).sum()/1e6:.1f} MB\n{sep}")
    print("dtypes:")
    print(df.dtypes.to_string())
    print("\nnulls per column:")
    print(df.isna().sum().to_string())
    print(f"\nhead({n}):")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.head(n).to_string())
    num = df.select_dtypes(include="number")
    if not num.empty:
        print("\ndescribe (numeric, transposed):")
        with pd.option_context("display.width", 220):
            print(num.describe().T.to_string())


def _key_checks(frames):
    sep = "=" * 88
    print(f"\n{sep}\nKEY CHECKS\n{sep}")

    if "books" in frames and "book_tags" in frames:
        books, bt = frames["books"], frames["book_tags"]
        b_global = set(books["book_id"].astype("int64"))
        bt_ids = set(bt["goodreads_book_id"].astype("int64"))
        print(f"books.book_id uniq               = {len(b_global)}")
        print(f"book_tags.goodreads_book_id uniq = {len(bt_ids)}")
        print(f"overlap                          = {len(b_global & bt_ids)}")

    if "books" in frames and "ratings" in frames:
        books, r = frames["books"], frames["ratings"]
        b_local = set(books["id"].astype("int64"))
        r_ids = set(r["book_id"].astype("int64"))
        print(f"books.id uniq        = {len(b_local)}")
        print(f"ratings.book_id uniq = {len(r_ids)}")
        print(f"overlap              = {len(b_local & r_ids)}")

    if "ratings" in frames:
        r = frames["ratings"]
        n_users = r["user_id"].nunique()
        n_books = r["book_id"].nunique()
        n = len(r)
        density = n / (n_users * n_books)
        print(f"\nratings: users={n_users:,}, books={n_books:,}, interactions={n:,}")
        print(f"density = {density:.6f}  -> sparsity = {1 - density:.6f}")
        print("rating distribution:")
        print(r["rating"].value_counts(normalize=True).sort_index().to_string())
        print(f"mean rating = {r['rating'].mean():.3f}")

    if "ratings" in frames and "to_read" in frames:
        tr = frames["to_read"]
        print(f"\nto_read: users={tr['user_id'].nunique():,}, books={tr['book_id'].nunique():,}, n={len(tr):,}")


def main():
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"DATA_DIR     = {DATA_DIR}")
    frames = load_raw()
    for name, df in frames.items():
        profile_frame(name, df)
    _key_checks(frames)


if __name__ == "__main__":
    main()

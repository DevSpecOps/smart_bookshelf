import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "archive"

SOURCES = {
    "books.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv",
    "ratings.csv": "https://github.com/Alireza-Akhavan/datasets_and_models/raw/main/ratings.csv",
    "tags.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv",
    "book_tags.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv",
    "to_read.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/to_read.csv",
}

EXPECTED_RATINGS_ROWS = 981756


def _row_count(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def _download(name, url):
    dest = DATA_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {name} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"downloading {name}")
    urllib.request.urlretrieve(url, dest)
    print(f"saved {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        _download(name, url)
    rows = _row_count(DATA_DIR / "ratings.csv")
    if rows != EXPECTED_RATINGS_ROWS:
        print(f"ratings.csv: expected {EXPECTED_RATINGS_ROWS} rows, got {rows}", file=sys.stderr)
        sys.exit(1)
    print(f"\nall files in {DATA_DIR}, ratings.csv verified ({rows} rows)")


if __name__ == "__main__":
    main()

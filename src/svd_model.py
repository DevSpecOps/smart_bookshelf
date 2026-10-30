import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader


class SVDRecommender:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.model = None
        self.trainset = None
        self.known_items = set()
        self.seen = {}

    def fit(self, ratings):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)
        self.trainset = data.build_full_trainset()
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        )
        self.model.fit(self.trainset)
        self.known_items = set(int(b) for b in ratings["book_id"].unique())
        self.seen = ratings.groupby("user_id")["book_id"].apply(set).to_dict()
        return self

    def predict(self, user_id, book_id):
        return float(self.model.predict(user_id, book_id).est)

    def rmse(self, test_df):
        est = np.array([self.model.predict(r.user_id, r.book_id).est for r in test_df.itertuples()])
        actual = test_df["rating"].to_numpy(dtype=float)
        return float(np.sqrt(np.mean((est - actual) ** 2)))

    def get_recommendations(self, user_id, N=5):
        seen = self.seen.get(user_id, set())
        candidates = [b for b in self.known_items if b not in seen]
        if not candidates:
            return []
        scored = [(b, self.model.predict(user_id, b).est) for b in candidates]
        scored.sort(key=lambda kv: -kv[1])
        return [int(b) for b, _ in scored[:N]]

    def coverage(self, n=10, max_users=500):
        exposed = set()
        sample = list(self.seen.keys())[:max_users]
        for uid in sample:
            exposed.update(self.get_recommendations(int(uid), n))
        return len(exposed) / len(self.known_items) if self.known_items else 0.0

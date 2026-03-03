from pathlib import Path
import joblib
import numpy as np

ARTIFACTS_PATH = Path("artifacts/collab")

class CollabRecommender:

    def __init__(self, path: Path = ARTIFACTS_PATH):
        self.item_similarity = joblib.load(path / "item_similarity.joblib")
        self.R_centered = joblib.load(path / "R_centered.joblib")
        self.user_means = joblib.load(path / "user_means.joblib")
        self.userid_to_index = joblib.load(path / "userid_to_index.joblib")
        self.movieid_to_index = joblib.load(path / "movieid_to_index.joblib")
        self.index_to_movieid = joblib.load(path / "index_to_movieid.joblib")  

    def recommend(self, user_id: int, n_recs: int = 10, k_neighbors: int = 50, reverse: bool = True):
        if user_id not in self.userid_to_index:
            raise ValueError(f"user_id {user_id} not found")

        uidx = self.userid_to_index[user_id]
        user_row = self.R_centered.getrow(uidx)  # sparse (1 x movies)

        
        rated_items = user_row.indices
        rated_vals = user_row.data

        if rated_items.size == 0:
            return []

        n_movies = self.item_similarity.shape[0]
        scores = np.zeros(n_movies, dtype=np.float32)
        

        # Weighted sum of similarities from rated items
        for i, r_ui in zip(rated_items, rated_vals):
            sim_row = self.item_similarity.getrow(i)
            if sim_row.nnz == 0:
                continue

            # keep only top neighbors for this item for speed
            if sim_row.nnz > k_neighbors:
                topk = np.argpartition(sim_row.data, -k_neighbors)[-k_neighbors:]
                neigh_cols = sim_row.indices[topk]
                neigh_sims = sim_row.data[topk]
            else:
                neigh_cols = sim_row.indices
                neigh_sims = sim_row.data

            scores[neigh_cols] += neigh_sims * r_ui


        # Exclude already-rated movies
        if reverse:
            scores[rated_items] = -np.inf  # exclude from best
        else:
            scores[rated_items] = np.inf   # exclude from worst
        
        #top_N
        finite = np.isfinite(scores)
        if not finite.any():
            return []

        take = min(n_recs, int(finite.sum()))
    
        if reverse:
            # BEST (highest scores)
            idx = np.argpartition(scores, -take)[-take:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        else:
            # WORST (lowest scores)
            idx = np.argpartition(scores, take)[:take]
            idx = idx[np.argsort(scores[idx])]

        recs = []
        for i in idx:
            mid = int(self.index_to_movieid[int(i)])
            score = float(scores[int(i)])
            recs.append({"movieId": mid, "predicted_score": round(score, 4)})

        return recs
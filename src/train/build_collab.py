import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("data")
ARTIFACTS_PATH = Path("artifacts/collab")
ARTIFACTS_PATH.mkdir( exist_ok=True)


def main():

    ratings = pd.read_csv(DATA_PATH / "ratings.csv") 

    unique_users = ratings["userId"].unique()
    unique_movies = ratings["movieId"].unique()

    userid_to_index = {uid: i for i, uid in enumerate(unique_users)}
    movieid_to_index = {mid: i for i, mid in enumerate(unique_movies)}
    index_to_movieid = {i: mid for mid, i in movieid_to_index.items()}

    # Sparse matrix indices
    user_idx = ratings["userId"].map(userid_to_index).to_numpy()
    movie_idx = ratings["movieId"].map(movieid_to_index).to_numpy()
    values = ratings["rating"].to_numpy(dtype=np.float32)

    num_users = len(unique_users)
    num_movies = len(unique_movies)

    R = csr_matrix((values, (user_idx, movie_idx)), shape=(num_users, num_movies))

    # Mean centering for bias removal (reduces "this user rates high/low" bias)
    R_centered = R.copy().tocsr()

    user_means = np.zeros(num_users, dtype=np.float32)
    for u in range(num_users):
        start, end = R_centered.indptr[u], R_centered.indptr[u + 1]
        if start == end:
            continue
        mean_u = R_centered.data[start:end].mean()
        user_means[u] = mean_u
        R_centered.data[start:end] -= mean_u


   # Transpose so: Movies x Users
    M = R_centered.T.tocsr()
    M_norm = normalize(M, norm="l2", axis=1, copy=True)

    # Movie-movie similarity
    item_sim = cosine_similarity(M_norm, dense_output=False)

    # Save artifacts
    joblib.dump(item_sim, ARTIFACTS_PATH / "item_similarity.joblib")
    joblib.dump(R_centered, ARTIFACTS_PATH / "R_centered.joblib")
    joblib.dump(user_means, ARTIFACTS_PATH / "user_means.joblib")
    joblib.dump(userid_to_index, ARTIFACTS_PATH / "userid_to_index.joblib")
    joblib.dump(movieid_to_index, ARTIFACTS_PATH / "movieid_to_index.joblib")
    joblib.dump(index_to_movieid, ARTIFACTS_PATH / "index_to_movieid.joblib")

    print("Saved to:", ARTIFACTS_PATH)
    print("Users:", num_users, "Movies:", num_movies)
    print("Similarity shape:", item_sim.shape, "nnz:", item_sim.nnz)

 

if __name__ == "__main__":
    main()
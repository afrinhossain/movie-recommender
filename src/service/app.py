import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pathlib import Path
import joblib
from src.service.collab_recommender import CollabRecommender



app = FastAPI()

ARTIFACTS = Path("artifacts")

movies = pd.read_csv(ARTIFACTS /'movies.csv')
similarity_matrix = joblib.load(ARTIFACTS / "similarity_matrix.joblib")
movieid_to_index = joblib.load(ARTIFACTS / "movieid_to_index.joblib")

rec_collab = CollabRecommender()
movieid_to_title = dict(zip(movies["movieId"], movies["title"]))

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/similar/{movie_id}")
def get_similar(movie_id: int, k:int = 5):

    if movie_id not in movieid_to_index:
        raise HTTPException(status_code=404, detail=f"movie_id {movie_id} not found")
    

    idx = movieid_to_index[movie_id]

    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_k = scores[1:k+1]

    query_movie = {
        'Queried movie': movies.loc[idx, "title"] + ' (movieId: ' + str(movies.loc[idx, "movieId"]) + ')',
        'genres': movies.loc[idx, "genres"],
    }

    results = []
    for rank, (i, score) in enumerate(top_k, start=1):
        title = movies.loc[i, "title"]
        mid = movies.loc[i, "movieId"]
        genres = movies.loc[i, "genres"]
        results.append(
            f"{rank}. {title} - movieId: ({mid}) with similarity score: {round(float(score),3)}"
        )

    return {
        'query_movie': query_movie,
        'top_k': k,
        'recommendations': results
    }


@app.get("/movies")
def search_movies(name:str):
    name = name.lower()
    matches = movies[movies["text"].str.contains(name)]
    if len(matches) == 0:
        raise HTTPException(status_code=404, detail=f"No movies found matching '{name}'")
    
    return matches[["movieId", "title"]].to_dict(orient="records")


@app.get("/recommend/collab/{user_id}")
def recommend_collab(user_id: int, n_recs: int = 10, reverse: bool = True):
    try:
        rec_ids = rec_collab.recommend(user_id=user_id, n_recs=n_recs, reverse=reverse)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    
    for r in rec_ids:
        mid = r["movieId"]
        r["title"] = movieid_to_title.get(mid, "Unknown title")

    return {"user_id": user_id, "top_k": n_recs, "recommendations": rec_ids}
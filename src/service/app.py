import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pathlib import Path
import joblib


app = FastAPI()

ARTIFACTS = Path("artifacts")

movies = pd.read_csv(ARTIFACTS /'movies.csv')
similarity_matrix = joblib.load(ARTIFACTS / "similarity_matrix.joblib")
movieid_to_index = joblib.load(ARTIFACTS / "movieid_to_index.joblib")


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
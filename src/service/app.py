import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse


app = FastAPI()

movies = pd.read_csv('data/movies.csv')
movies["movieId"] = movies["movieId"].astype(int)



# text prep
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)
movies["text"] = (movies["title"] + " " + movies["genres"]).str.lower()


#tf-idf build
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies["text"])

similarity_matrix = cosine_similarity(tfidf_matrix)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/similar/{movie_id}")
def get_similar(movie_id: int, k:int = 5):

    matches = movies.index[movies["movieId"] == movie_id]
    if len(matches) == 0:
        raise HTTPException(status_code=404, detail=f"movie_id {movie_id} not found")
    

    idx = matches[0]

    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_k = scores[1:k+1]

    query_movie = {
        "movieId": int(movies.loc[idx, "movieId"]),
        "title": movies.loc[idx, "title"],
        "genres": movies.loc[idx, "genres"],
    }

    results = []
    for i, score in top_k:
        results.append({
            "movieId": int(movies.loc[i, "movieId"]),
            "title": movies.loc[i, "title"],
            "genres": movies.loc[i, "genres"],
            "similarity_score": round(float(score), 3)
        })

    return {
        "query_movie": query_movie,
        "top_k": k,
        "recommendations": results
    }


@app.get("/movies")
def search_movies(name:str):
    name = name.lower()
    matches = movies[movies["text"].str.contains(name)]
    if len(matches) == 0:
        raise HTTPException(status_code=404, detail=f"No movies found matching '{name}'")
    
    return matches[["movieId", "title"]].to_dict(orient="records")
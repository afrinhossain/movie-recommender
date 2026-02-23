from pathlib import Path    
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity  

DATA_PATH = Path("data/movies.csv")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def main():
    movies = pd.read_csv('data/movies.csv')
    
    movies["title"] = movies["title"].fillna("")
    movies["genres"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    movies["movieId"] = movies["movieId"].astype(int)

    movies["text"] = (movies["title"] + " " + movies["genres"]).str.lower()
    

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies["text"])

    similarity_matrix = cosine_similarity(tfidf_matrix) 

    movieid_to_index = pd.Series(movies.index, index=movies["movieId"]).to_dict()

    movies[["movieId", "title", "genres"]].to_csv(ARTIFACTS / "movies.csv", index=False)
    joblib.dump(similarity_matrix, ARTIFACTS / "similarity_matrix.joblib")
    joblib.dump(movieid_to_index, ARTIFACTS / "movieid_to_index.joblib")

    print("Saved artifacts to:", ARTIFACTS)                




if __name__ == "__main__":    main()



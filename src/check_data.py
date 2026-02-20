import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



movies = pd.read_csv('data/movies.csv')

# text prep
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)
movies["text"] = (movies["title"] + " " + movies["genres"]).str.lower()


#Checking the structure of the dataset

print("Number of movies:", len(movies))
print("Columns in the dataset:", list(movies.columns))
print("\nFirst 5 rows of the dataset:")
print(movies.head())



#tf-idf build
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies["text"])

similarity_matrix = cosine_similarity(tfidf_matrix)

#sanity check
print("TF-IDF shape:", tfidf_matrix.shape)
print("Similarity matrix shape:", similarity_matrix.shape)


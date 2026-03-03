import pandas as pd



movies = pd.read_csv('data/movies.csv')

# text prep
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)
movies["text"] = (movies["title"] + " " + movies["genres"]).str.lower()


#Checking the structure of the movies dataset

print("Number of movies:", len(movies))
print("Columns in the dataset:", list(movies.columns))
print("\nFirst 5 rows of the dataset:")
print(movies.head())


#Checking the structure of the ratings dataset
ratings = pd.read_csv("data/ratings.csv")
print("Rows:", len(ratings))
print("Columns:", ratings.columns.tolist())
print("Unique users:", ratings["userId"].nunique())
print("Unique movies:", ratings["movieId"].nunique())



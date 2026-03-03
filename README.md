# Movie Recommendation System

A production-style movie recommendation API built with **FastAPI**, featuring:

- **Content-Based Filtering** (TF-IDF + Cosine Similarity)
- **Item-Item Collaborative Filtering**
- Sparse matrix optimization (SciPy)
- Structured JSON API responses

---

## Features

### 1. Content-Based Recommendations

Recommend movies similar to a given movie using metadata features.

**Endpoint**
```bash
GET /similar/{movie_id}?k=5
```

**Example**
```bash
/similar/1?k=5
```

Returns:
- Queried movie (movieId, title, genres)
- Top-k similar movies
- Cosine similarity scores

---

###  2. Search Movies

Search movies by name (case-insensitive substring match).

**Endpoint**
```bash
GET /movies?name=star
```

**Example**
```bash
/movies?name=toy
```

Returns:
- List of matching movies
- Each containing `movieId` and `title`

---

### 3. Collaborative Filtering (Personalized)

Recommend movies for a specific user based on rating behavior.

**Endpoint**
```bash
GET /recommend/collab/{user_id}?n_recs=10&reverse=true
```

**Query Parameters**
- `n_recs` → Number of recommendations (default: 10)
- `reverse`
  - `true` → Best recommendations (highest score first)
  - `false` → Worst recommendations (lowest score first)

**Examples**
```bash
/recommend/collab/8?n_recs=5
/recommend/collab/8?n_recs=5&reverse=false
```

---

##  How It Works

### Content-Based Filtering

- TF-IDF vectorization of movie metadata
- Cosine similarity between movie vectors
- Returns movies with similar textual features

**Pros**
- No user ratings required
- Works well for cold start

**Cons**
- Not personalized

---

### Collaborative Filtering (Item-Item)

For a given user:

```
score(candidate_movie) += similarity(rated_movie, candidate_movie) 
                           × centered_rating(user, rated_movie)
```

Where:

- `similarity(i, j)` = cosine similarity between movies  
- `centered_rating = rating − user_mean`

The result is a **ranking signal** (`predicted_score`) used to sort movies.

> Higher score = stronger recommendation  
> Lower score = weaker recommendation  

---

##  Project Structure

```
movie-recommender/
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── artifacts/
│   ├── movies.csv
│   ├── similarity_matrix.joblib
│   └── collab/
│       ├── item_similarity.joblib
│       ├── R_centered.joblib
│       ├── user_means.joblib
│       ├── userid_to_index.joblib
│       ├── movieid_to_index.joblib
│       └── index_to_movieid.joblib
│
├── src/
│   └── service/
│       ├── app.py
│       └── collab_recommender.py
│
└── README.md
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
```

Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Build Artifacts

Build content-based artifacts:

```bash
python src/build_content.py
```

Build collaborative artifacts:

```bash
python src/build_collab_artifacts.py
```

---

##  Run the API

```bash
python -m uvicorn src.service.app:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

##  Example Response (Collaborative)

```json
{
  "user_id": 8,
  "top_k": 5,
  "recommendations": [
    {
      "movieId": 1196,
      "title": "Star Wars (1977)",
      "predicted_score": 3.8421
    }
  ]
}
```

---


🎬 Movie Recommendation System

A production-style movie recommendation system built with:

Content-Based Filtering (TF-IDF + cosine similarity)

Item-Item Collaborative Filtering

FastAPI REST API

Sparse matrix optimization using SciPy

🚀 Features
Content-Based Recommendations

Find movies similar to a given movie based on metadata.

Endpoint

GET /similar/{movie_id}?k=5

Example:

/similar/1

Returns:

Queried movie (movieId, title, genres)

Top-k similar movies

Cosine similarity scores

Collaborative Filtering (Personalized)

Recommend movies for a specific user based on rating behavior.

Endpoint

GET /recommend/collab/{user_id}?n_recs=10&mode=best|worst

Examples:

/recommend/collab/8?n_recs=5
/recommend/collab/8?n_recs=5&mode=worst

Returns:

Query user

Recommendation mode (best/worst)

Structured movie objects

recommendation_score (ranking signal)

🧠 How It Works
Content-Based Filtering

TF-IDF vectorization on movie metadata

Cosine similarity between movie vectors

Returns movies with similar textual features

Strength:

Works without user ratings

Good cold-start performance

Limitation:

Not personalized

Collaborative Filtering (Item-Item)

For a given user:

score(j) = sum over i ( similarity(i, j) * centered_rating_ui )

Where:

similarity(i,j) = cosine similarity between movies

centered_rating_ui = rating - user_mean

The score is used as a ranking signal.

Strength:

Personalized recommendations

Learns user taste patterns

Limitation:

Requires rating history

🏗 Project Structure
movie-recsys/
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── artifacts/
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
│   ├── service/
│   │   ├── app.py
│   │   └── collab_recommender.py
│   ├── build_content.py
│   └── build_collab_artifacts.py
│
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/movie-recsys.git
cd movie-recsys

Create a virtual environment:

Windows:

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
🔨 Build Artifacts

Build content-based artifacts:

python src/build_content.py

Build collaborative artifacts:

python src/build_collab_artifacts.py
▶️ Run the API
python -m uvicorn src.service.app:app --reload

Open in browser:

http://127.0.0.1:8000/docs

Swagger UI will appear.

📦 Example Response (Collaborative)
{
  "query_user": { "userId": 8 },
  "mode": "best",
  "top_n": 5,
  "recommendations": [
    {
      "movieId": 1196,
      "title": "Star Wars (1977)",
      "recommendation_score": 3.8421
    }
  ]
}
📈 Future Improvements

Hybrid recommender (content + collaborative blend)

Evaluation metrics (Precision@K)

Similarity shrinkage

Docker deployment

Frontend UI

👨‍💻 Author

Portfolio project demonstrating:

Recommender systems

Machine learning fundamentals

Sparse matrix optimization

FastAPI backend development

Clean software architecture

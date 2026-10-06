import pandas as pd
import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


# Load the movie dataset
movies = pd.read_csv("data/tmdb_5000_movies.csv")


# Convert genres and keywords into simple names
def get_names(text):
    items = json.loads(text)
    names = []

    for item in items:
        names.append(item["name"])

    return " ".join(names)


# Prepare genres
movies["genres"] = movies["genres"].apply(get_names)

# Prepare keywords
movies["keywords"] = movies["keywords"].apply(get_names)

# Fill missing overview
movies["overview"] = movies["overview"].fillna("")


# Create movie tags
movies["tags"] = (
    movies["genres"]
    + " "
    + movies["keywords"]
    + " "
    + movies["overview"]
)


# Get release year
movies["release_year"] = pd.to_datetime(
    movies["release_date"],
    errors="coerce"
).dt.year

movies["release_year"] = movies["release_year"].fillna(0).astype(int)


# Keep useful movie information
movies = movies[
    [
        "title",
        "genres",
        "vote_average",
        "release_year",
        "popularity",
        "tags"
    ]
]


# Save cleaned movie data
movies.to_csv("data/cleaned_movies.csv", index=False)


# Create TF-IDF model
vectorizer = TfidfVectorizer(stop_words="english")

movie_vectors = vectorizer.fit_transform(movies["tags"])


# Save model files
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(movie_vectors, "models/movie_vectors.pkl")


print("Model training completed!")
print("Number of movies:", len(movies))
print("Number of features:", movie_vectors.shape[1])
print("Cleaned movie data and model files saved successfully!")
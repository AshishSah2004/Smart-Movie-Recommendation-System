import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# Page settings
st.set_page_config(
    page_title="Smart Movie Recommendation System",
    page_icon="🎬"
)


# Load movie data
movies = pd.read_csv("data/cleaned_movies.csv")

# Load TF-IDF model data
movie_vectors = joblib.load("models/movie_vectors.pkl")


# Recommendation function
def recommend_movies(movie_name, number_of_movies):

    # Find selected movie
    movie_index = movies[movies["title"] == movie_name].index[0]

    # Calculate similarity
    similarity_scores = cosine_similarity(
        movie_vectors[movie_index],
        movie_vectors
    )[0]

    # Sort by similarity
    similar_movies = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    # Get recommended movies
    for index, score in similar_movies[1:number_of_movies + 1]:

        movie = movies.iloc[index]

        recommendations.append({
            "title": movie["title"],
            "genres": movie["genres"],
            "rating": movie["vote_average"],
            "year": movie["release_year"],
            "popularity": movie["popularity"],
            "similarity": score
        })

    return recommendations


# App heading
st.title("🎬 Smart Movie Recommendation System")

st.write(
    "Select a movie and discover similar movies using Machine Learning."
)


# Search for a movie
search_text = st.text_input(
    "🔎 Search for a movie:",
    placeholder="Type a movie name..."
)


# Filter movies based on search
if search_text:

    matching_movies = movies[
        movies["title"].str.contains(
            search_text,
            case=False,
            na=False
        )
    ]["title"].tolist()

else:

    matching_movies = movies["title"].tolist()


# Movie selection
movie_name = st.selectbox(
    "🎬 Choose a movie:",
    matching_movies
)


# Number of recommendations
number_of_movies = st.slider(
    "Number of recommendations:",
    3,
    10,
    5
)


if st.button("🎯 Recommend Movies"):

    recommendations = recommend_movies(
        movie_name,
        number_of_movies
    )

    st.subheader("🎥 Recommended Movies")

    for movie in recommendations:

        st.markdown("---")

        st.markdown(
            f"### 🎬 {movie['title']}"
        )

        st.write(
            "🎭 Genre:",
            movie["genres"]
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "⭐ Rating",
                round(movie["rating"], 1)
            )

        with col2:
            st.metric(
                "📅 Year",
                movie["year"]
            )

        with col3:
            st.metric(
                "🔥 Popularity",
                round(movie["popularity"], 1)
            )

        similarity = round(
            movie["similarity"] * 100,
            2
        )

        st.write(
            "🎯 Similarity Score:",
            str(similarity) + "%"
        )

        st.progress(
            float(movie["similarity"])
        )

# Footer
st.markdown("---")

st.write(
    "Developed by Ashish Sah | B.Tech CSE (AI & ML)"
)
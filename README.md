# 🎬 Smart Movie Recommendation System

A content-based movie recommendation system built using Python, Machine Learning and Streamlit. The application recommends similar movies based on genres, keywords and movie overview.

## 🌐 Live Demo

🔗 https://smart-movie-recommendation-system-by-ashish.streamlit.app

## 📌 Overview

The Smart Movie Recommendation System is a machine learning project that helps users discover movies similar to a movie they already like.

The system analyzes movie content such as genres, keywords and overview, converts this information into numerical features using TF-IDF, and uses Cosine Similarity to find the most similar movies.

## 🚀 Features

- 🔎 Search movies by name
- 🎬 Select a movie from the available dataset
- 🎯 Get similar movie recommendations
- 🔢 Choose the number of recommendations
- 🎭 View movie genres
- ⭐ View movie ratings
- 📅 View release year
- 🔥 View movie popularity
- 📊 View similarity scores
- 🌐 Interactive Streamlit web application

## 🧠 How It Works

The recommendation system follows a content-based approach.

Movie information from different columns is combined into a single text feature called `tags`.

The main information used is:

- Genres
- Keywords
- Overview

The combined movie information is then processed using TF-IDF.

### TF-IDF

TF-IDF converts the text information of movies into numerical vectors.

This allows the machine learning system to compare movies mathematically.

### Cosine Similarity

Cosine Similarity is used to compare the selected movie with other movies.

Movies with higher similarity scores are considered more similar and are shown as recommendations.

## 🔄 Recommendation Flow

```text
Movie Selection
       ↓
Movie Information
       ↓
Genres + Keywords + Overview
       ↓
Combined Tags
       ↓
TF-IDF Vectorization
       ↓
Cosine Similarity
       ↓
Similarity Ranking
       ↓
Top Similar Movies
       ↓
Recommendations
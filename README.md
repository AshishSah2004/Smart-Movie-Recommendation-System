# 🎬 Smart Movie Recommendation System

A content-based movie recommendation system built using Python, Machine Learning and Streamlit.

## 🚀 Features

- Search movies by name
- Get similar movie recommendations
- Select number of recommendations
- View movie genres
- View ratings and release year
- View popularity
- View similarity score
- Interactive Streamlit web interface

## 🧠 How It Works

The system uses a content-based recommendation approach.

Movie information such as:

- Genres
- Keywords
- Overview

is combined into a single text feature.

TF-IDF converts the movie text into numerical vectors.

Cosine Similarity compares the selected movie with other movies and finds the most similar ones.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-Learn
- TF-IDF
- Cosine Similarity
- Streamlit
- Joblib

## 📂 Project Structure

```text
Smart-Movie-Recommendation-System/
│
├── data/
│   └── cleaned_movies.csv
│
├── models/
│   ├── movie_vectors.pkl
│   └── vectorizer.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
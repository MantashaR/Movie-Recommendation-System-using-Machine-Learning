# Movie-Recommendation-System-using-Machine-Learning
A content-based Movie Recommendation System using ML and NLP that suggests similar movies based on cosine similarity of movie features, built in Google Colab using Kaggle dataset.
<br>

A Machine Learning-based Movie Recommendation System built using Python in Google Colab.  
It suggests movies similar to a user's selected movie using content-based filtering and NLP techniques.



## 📌 Project Overview

This project recommends movies based on their similarity in features like:
- Genre
- Cast
- Keywords
- Overview (description)


## 🧠 How It Works

1. Data is collected from the TMDB dataset (Kaggle)
2. Relevant features are combined into a single text field
3. Text is converted into vectors using **TF-IDF**
4. Cosine similarity is calculated between all movies
5. Top similar movies are recommended

## 📊 Dataset

- Source: Kaggle TMDB Movie Dataset  
- Contains movie details like title, overview, genre, cast, etc.

## 🛠️ Tech Stack

- Python 🐍  
- Google Colab  
- Pandas  
- NumPy  
- Scikit-learn  
- NLP (TF-IDF Vectorizer)

## 🚀 Features

- Recommend similar movies instantly
- Content-based filtering system
- Simple and fast predictions
- Easy to extend with UI (Streamlit/Flask)

## 📂 Project Structure

```text
movie-recommendation-system/
│
├── movie_recommender.ipynb   # Main Colab notebook
├── dataset/                  # Kaggle dataset files
├── README.md                 # Project documentation

# Movie Recommendation System

A content-based movie recommender built with **TF-IDF + cosine similarity** and a **Streamlit** web UI.

Pick a movie you like, get a ranked list of similar movies based on genres, cast, keywords, and plot.

## Demo

```
streamlit run app.py
```

Opens at http://localhost:8501

## Project structure

```
.
├── app.py                       # Streamlit UI
├── recommender.py               # TF-IDF + cosine similarity model
├── requirements.txt
├── dataset/
│   └── sample_movies.csv        # Bundled 35-movie sample (works out of the box)
└── README.md
```


The app runs immediately using the bundled `sample_movies.csv` (35 popular movies).

## Using the full TMDB 5000 dataset

The bundled sample is for demo purposes. For a more realistic experience, swap in the full Kaggle dataset:

1. Download from [Kaggle: TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
2. Run a small preprocessing script to merge `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` into the schema the app expects:

   ```python
   import ast, pandas as pd
   movies = pd.read_csv("tmdb_5000_movies.csv")
   credits = pd.read_csv("tmdb_5000_credits.csv")
   df = movies.merge(credits, on="title")

   def names(s, top=3):
       try:
           return "|".join(d["name"] for d in ast.literal_eval(s)[:top])
       except Exception:
           return ""

   out = pd.DataFrame({
       "movie_id": df["id"],
       "title": df["title"],
       "genres": df["genres"].apply(lambda s: names(s, 5)),
       "cast": df["cast"].apply(lambda s: names(s, 3)),
       "keywords": df["keywords"].apply(lambda s: names(s, 8)),
       "overview": df["overview"].fillna(""),
   })
   out.to_csv("dataset/sample_movies.csv", index=False)
   ```

3. Restart the Streamlit app — it will pick up the new dataset automatically.

## How it works

1. **Feature combination** — Genres, cast, keywords, and overview are concatenated into a single text field per movie.
2. **TF-IDF vectorization** — `TfidfVectorizer` converts the text into a sparse matrix, weighting rare informative tokens higher than common ones.
3. **Cosine similarity** — Pairwise similarity is computed between all movie vectors.
4. **Ranking** — For a chosen movie, the top-N most similar are returned (excluding the movie itself).

## Tech stack

- **Python 3.10+**
- **scikit-learn** — TF-IDF + cosine similarity
- **pandas** — data handling
- **Streamlit** — web UI

## Deploy free on Streamlit Cloud

1. Push this repo to GitHub (already done).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, file `app.py`.
4. Click **Deploy** — your app is live in ~2 minutes.



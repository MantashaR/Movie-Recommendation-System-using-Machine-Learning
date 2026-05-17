"""Content-based movie recommender using TF-IDF and cosine similarity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, csv_path: str | Path):
        self.df = pd.read_csv(csv_path)
        required = {"title", "genres", "cast", "keywords", "overview"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        self.df = self.df.drop_duplicates(subset="title").reset_index(drop=True)
        self.df["tags"] = (
            self.df["genres"].fillna("").str.replace("|", " ", regex=False)
            + " "
            + self.df["cast"].fillna("").str.replace("|", " ", regex=False)
            + " "
            + self.df["keywords"].fillna("").str.replace("|", " ", regex=False)
            + " "
            + self.df["overview"].fillna("")
        ).str.lower()

        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(self.df["tags"])
        self.similarity = cosine_similarity(tfidf_matrix)

    def titles(self) -> list[str]:
        return sorted(self.df["title"].tolist())

    def recommend(self, title: str, top_n: int = 5) -> list[dict]:
        matches = self.df.index[self.df["title"] == title]
        if len(matches) == 0:
            return []
        idx = matches[0]
        scores = sorted(
            enumerate(self.similarity[idx]),
            key=lambda x: x[1],
            reverse=True,
        )[1 : top_n + 1]
        return [
            {
                "title": self.df.iloc[i]["title"],
                "genres": self.df.iloc[i]["genres"],
                "overview": self.df.iloc[i]["overview"],
                "score": round(float(s), 3),
            }
            for i, s in scores
        ]

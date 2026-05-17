"""Streamlit UI for the content-based movie recommender."""

from pathlib import Path

import streamlit as st

from recommender import MovieRecommender

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

DATA_PATH = Path(__file__).parent / "dataset" / "sample_movies.csv"


@st.cache_resource
def load_model() -> MovieRecommender:
    return MovieRecommender(DATA_PATH)


st.title("🎬 Movie Recommendation System")
st.caption("Content-based recommendations using TF-IDF + cosine similarity")

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found at `{DATA_PATH}`. "
        "Place a CSV with columns: title, genres, cast, keywords, overview."
    )
    st.stop()

model = load_model()
titles = model.titles()

col1, col2 = st.columns([3, 1])
with col1:
    selected = st.selectbox("Pick a movie you like:", titles)
with col2:
    top_n = st.number_input(
        "How many?", min_value=1, max_value=15, value=5, step=1
    )

if st.button("Recommend", type="primary", use_container_width=True):
    results = model.recommend(selected, top_n=int(top_n))
    if not results:
        st.warning("No recommendations found.")
    else:
        st.subheader(f"Movies similar to *{selected}*")
        for r in results:
            with st.container(border=True):
                st.markdown(
                    f"### {r['title']}  \n"
                    f"`similarity: {r['score']}`"
                )
                st.caption(r["genres"].replace("|", " · "))
                st.write(r["overview"])

st.divider()
st.caption(
    "Built with Streamlit, scikit-learn, and pandas. "
    "Dataset is a small bundled sample — see README for using the full TMDB 5000."
)

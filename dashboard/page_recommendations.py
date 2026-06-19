"""
Recommendation Engine — Select a movie and method, get personalised
similar-movie recommendations displayed as styled cards.
"""

import re
import streamlit as st

from dashboard.styles import (
    inject_css, get_plotly_layout, get_theme, CHART_COLORS,
    recommendation_card_html, insight_html,
)
from dashboard.data_loader import load_data, get_movie_titles
from dashboard.models import (
    build_pivot_table, build_similarity_matrix, build_knn_model,
)
from dashboard.recommenders import (
    recommend_pearson, recommend_cosine, recommend_knn, recommend_hybrid,
)


def render():
    dark_mode = st.session_state.get("dark_mode", False)
    theme = get_theme(dark_mode)

    _, _, movies_df, data = load_data()

    # ── Controls ──────────────────────────────────────────────────────────
    st.markdown("### 🎯 Recommendation Engine")
    ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 1])
    with ctrl1:
        selected_movie = st.selectbox(
            "Search Movie",
            get_movie_titles(data),
            index=None,
            placeholder="Type to search...",
        )
    with ctrl2:
        method = st.selectbox(
            "Method",
            ["Pearson Correlation", "Cosine Similarity + KNN", "Hybrid"],
        )
    with ctrl3:
        n_results = st.slider("Results", 5, 20, 10)

    if selected_movie is None:
        st.info(
            "🎬 **Select a movie above** to get personalised recommendations. "
            "You can type part of a title to filter the list, then choose a method "
            "and how many results you'd like."
        )
        return

    # ── Generate recommendations ──────────────────────────────────────────
    try:
        with st.spinner("Building models & computing recommendations…"):
            pivot_raw, pivot_filled = build_pivot_table()
            similarity_df = build_similarity_matrix()
            knn_model, csr_data, movie_index = build_knn_model()

            if method == "Pearson Correlation":
                results = recommend_pearson(
                    selected_movie, pivot_raw, n=n_results, min_ratings=100
                )
                score_col = "Correlation"
            elif method == "Cosine Similarity + KNN":
                results = recommend_knn(
                    selected_movie, knn_model, csr_data, movie_index, n=n_results
                )
                score_col = "Similarity"
            else:  # Hybrid
                results = recommend_hybrid(
                    selected_movie, pivot_raw, similarity_df,
                    knn_model, csr_data, movie_index, n=n_results,
                )
                score_col = "HybridScore"

        if results is None or len(results) == 0:
            st.warning(
                "No recommendations found for this movie. "
                "It may have too few ratings to produce reliable results."
            )
            return

        # ── Display results ───────────────────────────────────────────────
        st.markdown(f"#### Recommendations for *{selected_movie}*")
        st.markdown(
            insight_html(
                f"Showing <b>{len(results)}</b> results using <b>{method}</b>. "
                f"Higher scores indicate stronger similarity to the selected movie."
            ),
            unsafe_allow_html=True,
        )

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            title = row["Title"]

            # Look up genre
            movie_info = movies_df[movies_df["Title"] == title]
            genre = (
                movie_info["Genres"].values[0] if len(movie_info) > 0 else "N/A"
            )

            # Extract year from title
            year_match = re.search(r"\((\d{4})\)", title)
            year = year_match.group(1) if year_match else "N/A"

            score = round(float(row[score_col]), 4)

            st.markdown(
                recommendation_card_html(
                    title=title,
                    genre=genre,
                    year=year,
                    score=score,
                    rank=rank,
                ),
                unsafe_allow_html=True,
            )

    except KeyError as e:
        st.warning(
            f"⚠️ Could not find **{selected_movie}** in the model data. "
            f"The movie may not have enough ratings. (Detail: {e})"
        )
    except Exception as e:
        st.warning(
            f"⚠️ An error occurred while generating recommendations: {e}"
        )

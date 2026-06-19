"""
User-Based Recommendation page module.
Allows users to rate movies and receive collaborative-filtering recommendations.
"""

import re
import streamlit as st
import plotly.graph_objects as go

from dashboard.styles import (
    inject_css, get_plotly_layout, get_theme, CHART_COLORS,
    metric_card_html, recommendation_card_html, insight_html, business_card_html,
)
from dashboard.data_loader import load_data, get_movie_stats, get_movie_titles
from dashboard.recommenders import recommend_user_based


def _extract_year(title: str) -> str:
    """Extract release year from a movie title like 'Toy Story (1995)'."""
    match = re.search(r"\((\d{4})\)", title)
    return match.group(1) if match else "N/A"


def _genre_for_title(movies_df, title: str) -> str:
    """Look up the genre string for a given movie title."""
    row = movies_df.loc[movies_df["Title"] == title, "Genres"]
    if not row.empty:
        return row.values[0]
    return "Unknown"


def render():
    """Main entry-point called by the dashboard router."""
    dark_mode = st.session_state.get("dark_mode", False)
    theme = get_theme(dark_mode)
    inject_css(dark_mode)

    # ── Load data ──────────────────────────────────────────────────────
    ratings_df, users_df, movies_df, data = load_data()
    stats = get_movie_stats(data)

    st.markdown("## 🎯 User-Based Recommendations")
    st.markdown(
        insight_html(
            "Rate at least <strong>3 movies</strong> you've watched, then click "
            "<em>Get Recommendations</em> to discover new films based on similar users."
        ),
        unsafe_allow_html=True,
    )

    # ── Movie selection ────────────────────────────────────────────────
    popular_titles = stats.head(30)["Title"].tolist()

    selected = st.multiselect(
        "🎬 Select movies you've watched",
        options=popular_titles,
        default=st.session_state.get("selected_movies", []),
        help="Pick from the top-30 most-rated movies.",
    )
    st.session_state["selected_movies"] = selected

    # ── Rating sliders ─────────────────────────────────────────────────
    if "user_ratings" not in st.session_state:
        st.session_state["user_ratings"] = {}

    if selected:
        st.markdown("### ⭐ Rate your selected movies")
        cols_per_row = 3
        for idx in range(0, len(selected), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if idx + j < len(selected):
                    title = selected[idx + j]
                    with col:
                        rating = st.slider(
                            title,
                            min_value=1,
                            max_value=5,
                            value=st.session_state["user_ratings"].get(title, 3),
                            key=f"slider_{title}",
                        )
                        st.session_state["user_ratings"][title] = rating

    # Remove ratings for deselected movies
    st.session_state["user_ratings"] = {
        t: r for t, r in st.session_state["user_ratings"].items() if t in selected
    }

    user_ratings = st.session_state["user_ratings"]

    # ── Summary metrics ────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        metric_card_html("Movies Rated", str(len(user_ratings)), icon="🎬"),
        unsafe_allow_html=True,
    )
    avg_r = (
        f"{sum(user_ratings.values()) / len(user_ratings):.1f}"
        if user_ratings
        else "—"
    )
    c2.markdown(
        metric_card_html("Avg Rating", avg_r, icon="⭐"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        metric_card_html("Min Required", "3", icon="🔒", sub="movies"),
        unsafe_allow_html=True,
    )

    # ── Generate button ────────────────────────────────────────────────
    st.markdown("---")

    if len(user_ratings) < 3:
        st.warning("⚠️ Please rate at least **3 movies** to generate recommendations.")
        return

    if st.button("🚀 Get Recommendations", use_container_width=True):
        with st.spinner("Finding similar users and generating recommendations…"):
            try:
                similar_users_df, recommendations_df = recommend_user_based(
                    user_ratings, data, n=10
                )
            except Exception as exc:
                st.error(f"An error occurred: {exc}")
                return

        # ── Similar Users ──────────────────────────────────────────────
        st.markdown("### 👥 Similar Users")

        if similar_users_df is None or similar_users_df.empty:
            st.info("No sufficiently similar users were found. Try rating more movies.")
        else:
            layout = get_plotly_layout(dark_mode)
            fig_sim = go.Figure(
                go.Bar(
                    x=similar_users_df["UserID"].astype(str),
                    y=similar_users_df["Similarity"],
                    marker_color=CHART_COLORS[: len(similar_users_df)],
                    text=similar_users_df["Similarity"].round(3),
                    textposition="outside",
                )
            )
            fig_sim.update_layout(
                **layout,
                title="User Similarity Scores",
                xaxis_title="User ID",
                yaxis_title="Cosine Similarity",
                height=380,
            )
            st.plotly_chart(fig_sim, use_container_width=True)
            st.markdown(
                insight_html(
                    f"Found <strong>{len(similar_users_df)}</strong> similar users. "
                    f"Highest similarity: <strong>{similar_users_df['Similarity'].max():.3f}</strong>."
                ),
                unsafe_allow_html=True,
            )

        # ── Recommendations ────────────────────────────────────────────
        st.markdown("### 🎬 Recommended Movies")

        if recommendations_df is None or recommendations_df.empty:
            st.info("No recommendations could be generated. Try changing your ratings.")
        else:
            cols_per_row = 2
            recs = recommendations_df.head(10)
            for idx in range(0, len(recs), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if idx + j < len(recs):
                        row = recs.iloc[idx + j]
                        title = row["Title"]
                        genre = _genre_for_title(movies_df, title)
                        year = _extract_year(title)
                        score = row["AvgWeightedScore"]
                        rank = idx + j + 1
                        with col:
                            st.markdown(
                                recommendation_card_html(
                                    title=title,
                                    genre=genre,
                                    year=year,
                                    score=round(score, 2),
                                    rank=rank,
                                ),
                                unsafe_allow_html=True,
                            )

            st.markdown(
                insight_html(
                    "Recommendations are ranked by <strong>weighted average score</strong> "
                    "from the most similar users in the dataset."
                ),
                unsafe_allow_html=True,
            )

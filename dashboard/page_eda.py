"""
Exploratory Data Analysis — Six-tab deep-dive into ratings, genres,
popularity, user behaviour, decade trends, and correlations.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from dashboard.styles import (
    inject_css, get_plotly_layout, get_theme, CHART_COLORS,
    insight_html,
)
from dashboard.data_loader import load_data, get_movie_stats, get_genre_counts


def render():
    dark_mode = st.session_state.get("dark_mode", False)
    theme = get_theme(dark_mode)
    layout_kw = get_plotly_layout(dark_mode)

    # ── data ──────────────────────────────────────────────────────────────
    ratings, users, movies, data = load_data()
    stats = get_movie_stats(data)
    genre_counts = get_genre_counts(data)

    # ── tabs ──────────────────────────────────────────────────────────────
    tab_ratings, tab_genres, tab_popular, tab_users, tab_decades, tab_corr = st.tabs(
        ["⭐ Ratings", "🎭 Genres", "🏆 Popular Movies", "👥 User Activity", "📅 Decades", "🔗 Correlations"]
    )

    # ═══════════════════════  TAB 1 — Ratings  ════════════════════════════
    with tab_ratings:
        st.markdown("#### Rating Distribution")
        rating_vals = data["Rating"]
        rating_counts = rating_vals.value_counts().sort_index()

        fig_hist = go.Figure(
            go.Bar(
                x=[str(r) for r in rating_counts.index],
                y=rating_counts.values,
                marker_color=CHART_COLORS[: len(rating_counts)],
                text=rating_counts.values,
                textposition="outside",
                texttemplate="%{text:,}",
            )
        )
        fig_hist.update_layout(
            **layout_kw,
            xaxis_title="Rating",
            yaxis_title="Frequency",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Rating Box Plot")
        fig_box = go.Figure(
            go.Box(
                y=rating_vals,
                marker_color=CHART_COLORS[2],
                boxmean="sd",
                name="Rating",
            )
        )
        fig_box.update_layout(**layout_kw, showlegend=False, height=350)
        st.plotly_chart(fig_box, use_container_width=True)

        median_r = rating_vals.median()
        mean_r = round(rating_vals.mean(), 2)
        st.markdown(
            insight_html(
                f"The rating distribution is <b>left-skewed</b> — the median ({median_r}) exceeds "
                f"the midpoint of the scale, and the mean is <b>{mean_r}</b>. Users tend to rate "
                f"movies they enjoy, creating a positivity bias."
            ),
            unsafe_allow_html=True,
        )

    # ═══════════════════════  TAB 2 — Genres  ═════════════════════════════
    with tab_genres:
        st.markdown("#### Genre Frequency")
        gc_sorted = genre_counts.sort_values(ascending=True)
        fig_gf = go.Figure(
            go.Bar(
                y=gc_sorted.index,
                x=gc_sorted.values,
                orientation="h",
                marker_color=CHART_COLORS[: len(gc_sorted)],
                text=gc_sorted.values,
                textposition="outside",
                texttemplate="%{text:,}",
            )
        )
        fig_gf.update_layout(
            **layout_kw,
            xaxis_title="Number of Ratings",
            yaxis_title="Genre",
            showlegend=False,
            height=550,
        )
        st.plotly_chart(fig_gf, use_container_width=True)

        st.markdown("#### Average Rating by Genre")
        # explode genres
        genre_exploded = data.assign(Genre=data["Genres"].str.split("|")).explode("Genre")
        avg_by_genre = (
            genre_exploded.groupby("Genre")["Rating"]
            .mean()
            .sort_values(ascending=True)
            .round(2)
        )
        fig_ag = go.Figure(
            go.Bar(
                y=avg_by_genre.index,
                x=avg_by_genre.values,
                orientation="h",
                marker_color=CHART_COLORS[4 : 4 + len(avg_by_genre)],
                text=avg_by_genre.values,
                textposition="outside",
            )
        )
        fig_ag.update_layout(
            **layout_kw,
            xaxis_title="Average Rating",
            yaxis_title="Genre",
            showlegend=False,
            height=550,
        )
        st.plotly_chart(fig_ag, use_container_width=True)

        top_pop = gc_sorted.index[-1]
        top_rated_genre = avg_by_genre.index[-1]
        st.markdown(
            insight_html(
                f"<b>{top_pop}</b> is the most frequently rated genre, but <b>{top_rated_genre}</b> "
                f"has the highest average rating. Popularity ≠ quality — niche genres often receive "
                f"higher ratings from their dedicated audiences."
            ),
            unsafe_allow_html=True,
        )

    # ═══════════════════════  TAB 3 — Popular Movies  ═════════════════════
    with tab_popular:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Top 20 Most Rated")
            top20_rated = stats.head(20).sort_values("NumRatings", ascending=True)
            fig_mr = go.Figure(
                go.Bar(
                    y=top20_rated["Title"].str[:35],
                    x=top20_rated["NumRatings"],
                    orientation="h",
                    marker_color=CHART_COLORS[0],
                    text=top20_rated["NumRatings"],
                    textposition="outside",
                    texttemplate="%{text:,}",
                )
            )
            fig_mr.update_layout(
                **layout_kw,
                xaxis_title="Number of Ratings",
                showlegend=False,
                height=600,
                margin=dict(l=200),
            )
            st.plotly_chart(fig_mr, use_container_width=True)

        with col_right:
            st.markdown("#### Top 20 Highest Rated (min 50 ratings)")
            qualified = stats[stats["NumRatings"] >= 50].sort_values(
                "AvgRating", ascending=False
            ).head(20)
            qualified_sorted = qualified.sort_values("AvgRating", ascending=True)
            fig_hr = go.Figure(
                go.Bar(
                    y=qualified_sorted["Title"].str[:35],
                    x=qualified_sorted["AvgRating"],
                    orientation="h",
                    marker_color=CHART_COLORS[3],
                    text=qualified_sorted["AvgRating"].round(2),
                    textposition="outside",
                )
            )
            fig_hr.update_layout(
                **layout_kw,
                xaxis_title="Average Rating",
                showlegend=False,
                height=600,
                margin=dict(l=200),
            )
            st.plotly_chart(fig_hr, use_container_width=True)

        st.markdown(
            insight_html(
                "Blockbusters dominate the 'most rated' list, while critically acclaimed films "
                "lead in average rating. A minimum threshold of 50 ratings filters out flukes "
                "with very few reviews."
            ),
            unsafe_allow_html=True,
        )

    # ═══════════════════════  TAB 4 — User Activity  ══════════════════════
    with tab_users:
        st.markdown("#### Ratings per User Distribution")
        user_rating_counts = data.groupby("UserID").size()

        fig_ua = go.Figure(
            go.Histogram(
                x=user_rating_counts,
                nbinsx=60,
                marker_color=CHART_COLORS[1],
            )
        )
        fig_ua.update_layout(
            **layout_kw,
            xaxis_title="Number of Ratings by a User",
            yaxis_title="Number of Users",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_ua, use_container_width=True)

        st.markdown("#### Top 10 Power Users")
        power = (
            user_rating_counts.sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        power.columns = ["UserID", "TotalRatings"]
        power.index = range(1, len(power) + 1)
        power.index.name = "Rank"
        st.dataframe(power, use_container_width=True)

        median_per_user = int(user_rating_counts.median())
        st.markdown(
            insight_html(
                f"User activity follows a <b>long-tail</b> distribution — the median user has rated "
                f"only <b>{median_per_user}</b> movies, while power users have rated thousands. "
                f"This sparse-data challenge is a key reason collaborative filtering can struggle."
            ),
            unsafe_allow_html=True,
        )

    # ═══════════════════════  TAB 5 — Decades  ════════════════════════════
    with tab_decades:
        if "Decade" in data.columns:
            st.markdown("#### Average Rating by Decade")
            decade_avg = (
                data.groupby("Decade")["Rating"]
                .mean()
                .sort_index()
                .round(2)
            )
            fig_da = go.Figure()
            fig_da.add_trace(
                go.Bar(
                    x=[str(d) + "s" for d in decade_avg.index],
                    y=decade_avg.values,
                    marker_color=CHART_COLORS[2],
                    name="Avg Rating",
                    opacity=0.6,
                )
            )
            fig_da.add_trace(
                go.Scatter(
                    x=[str(d) + "s" for d in decade_avg.index],
                    y=decade_avg.values,
                    mode="lines+markers+text",
                    text=decade_avg.values,
                    textposition="top center",
                    marker=dict(color=CHART_COLORS[0], size=8),
                    line=dict(color=CHART_COLORS[0], width=2),
                    name="Trend",
                )
            )
            fig_da.update_layout(
                **layout_kw,
                xaxis_title="Decade",
                yaxis_title="Average Rating",
                showlegend=True,
                height=420,
            )
            st.plotly_chart(fig_da, use_container_width=True)

            st.markdown("#### Number of Movies per Decade")
            movies_per_decade = (
                data.drop_duplicates("Title")
                .groupby("Decade")
                .size()
                .sort_index()
            )
            fig_mpd = go.Figure(
                go.Bar(
                    x=[str(d) + "s" for d in movies_per_decade.index],
                    y=movies_per_decade.values,
                    marker_color=CHART_COLORS[5],
                    text=movies_per_decade.values,
                    textposition="outside",
                    texttemplate="%{text:,}",
                )
            )
            fig_mpd.update_layout(
                **layout_kw,
                xaxis_title="Decade",
                yaxis_title="Movie Count",
                showlegend=False,
                height=380,
            )
            st.plotly_chart(fig_mpd, use_container_width=True)

            best_decade = decade_avg.idxmax()
            st.markdown(
                insight_html(
                    f"Older movies from the <b>{int(best_decade)}s</b> tend to have higher average "
                    f"ratings — likely a survivorship bias where only the best films from earlier "
                    f"decades remain well-known and rated."
                ),
                unsafe_allow_html=True,
            )

    # ═══════════════════════  TAB 6 — Correlations  ═══════════════════════
    with tab_corr:
        st.markdown("#### Average Rating vs Number of Ratings")
        scatter_data = stats.copy()
        fig_sc = go.Figure(
            go.Scatter(
                x=scatter_data["NumRatings"],
                y=scatter_data["AvgRating"],
                mode="markers",
                marker=dict(
                    size=scatter_data["NumRatings"].clip(upper=2000) / 80 + 4,
                    color=scatter_data["AvgRating"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Avg Rating"),
                    opacity=0.6,
                ),
                text=scatter_data["Title"],
                hovertemplate="<b>%{text}</b><br>Ratings: %{x:,}<br>Avg: %{y:.2f}<extra></extra>",
            )
        )
        # annotate top 5
        top5 = scatter_data.head(5)
        for _, row in top5.iterrows():
            fig_sc.add_annotation(
                x=row["NumRatings"],
                y=row["AvgRating"],
                text=row["Title"][:25],
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                ax=30,
                ay=-25,
                font=dict(size=10, color=theme["text"]),
            )

        fig_sc.update_layout(
            **layout_kw,
            xaxis_title="Number of Ratings",
            yaxis_title="Average Rating",
            showlegend=False,
            height=550,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown(
            insight_html(
                "There is a noticeable <b>popularity bias</b>: heavily-rated movies cluster around "
                "a 3.5–4.2 average, while low-rating-count films show extreme variance. "
                "The top 5 most-rated movies are annotated above — they tend to be mainstream "
                "blockbusters rather than the highest-quality films."
            ),
            unsafe_allow_html=True,
        )

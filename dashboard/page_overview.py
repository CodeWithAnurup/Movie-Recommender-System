"""
Overview Dashboard — KPI cards, distribution charts, and demographic breakdowns.
"""

import streamlit as st
import plotly.graph_objects as go

from dashboard.styles import (
    inject_css, get_plotly_layout, get_theme, CHART_COLORS,
    metric_card_html, insight_html,
)
from dashboard.data_loader import load_data, get_movie_stats, get_genre_counts


def render():
    dark_mode = st.session_state.get("dark_mode", False)
    theme = get_theme(dark_mode)
    layout_kw = get_plotly_layout(dark_mode)

    # ── data ──────────────────────────────────────────────────────────────
    ratings, users, movies, data = load_data()

    total_ratings = len(data)
    total_users = data["UserID"].nunique()
    total_movies = data["Title"].nunique()
    avg_rating = round(data["Rating"].mean(), 2)

    stats = get_movie_stats(data)
    most_rated = stats.iloc[0]["Title"] if len(stats) > 0 else "N/A"
    most_rated_count = int(stats.iloc[0]["NumRatings"]) if len(stats) > 0 else 0

    most_active_age = (
        data["AgeGroup"].value_counts().idxmax()
        if "AgeGroup" in data.columns
        else "N/A"
    )
    age_count = int(data["AgeGroup"].value_counts().max()) if "AgeGroup" in data.columns else 0

    most_active_occ = (
        data["OccupationLabel"].value_counts().idxmax()
        if "OccupationLabel" in data.columns
        else "N/A"
    )
    occ_count = int(data["OccupationLabel"].value_counts().max()) if "OccupationLabel" in data.columns else 0

    # ── KPI row ───────────────────────────────────────────────────────────
    st.markdown("### 📊 Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card_html("Total Ratings", f"{total_ratings:,}", icon="⭐"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card_html("Total Users", f"{total_users:,}", icon="👥"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card_html("Total Movies", f"{total_movies:,}", icon="🎬"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card_html("Average Rating", f"{avg_rating}", icon="📈", sub="out of 5"),
            unsafe_allow_html=True,
        )

    # ── Highlight row ─────────────────────────────────────────────────────
    st.markdown("### 🏆 Highlights")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(
            metric_card_html(
                "Most Rated Movie",
                most_rated[:40] + ("…" if len(most_rated) > 40 else ""),
                icon="🎥",
                sub=f"{most_rated_count:,} ratings",
            ),
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            metric_card_html(
                "Most Active Age Group",
                str(most_active_age),
                icon="🧑‍🤝‍🧑",
                sub=f"{age_count:,} ratings",
            ),
            unsafe_allow_html=True,
        )
    with h3:
        st.markdown(
            metric_card_html(
                "Most Active Occupation",
                str(most_active_occ),
                icon="💼",
                sub=f"{occ_count:,} ratings",
            ),
            unsafe_allow_html=True,
        )

    # ── Rating Distribution ───────────────────────────────────────────────
    st.markdown("### ⭐ Rating Distribution")
    rating_counts = data["Rating"].value_counts().sort_index()
    rating_colors = ["#ea4335", "#ff6d01", "#fbbc04", "#34a853", "#0d904f"]
    fig_rating = go.Figure(
        go.Bar(
            x=[str(r) for r in rating_counts.index],
            y=rating_counts.values,
            marker_color=rating_colors[: len(rating_counts)],
            text=rating_counts.values,
            textposition="outside",
            texttemplate="%{text:,}",
        )
    )
    fig_rating.update_layout(
        **layout_kw,
        xaxis_title="Rating",
        yaxis_title="Count",
        showlegend=False,
        height=380,
    )
    st.plotly_chart(fig_rating, use_container_width=True)
    st.markdown(
        insight_html(
            f"Rating 4 is the most common choice, suggesting a positivity bias in user ratings. "
            f"The average rating across all entries is <b>{avg_rating}</b>."
        ),
        unsafe_allow_html=True,
    )

    # ── Gender Distribution ───────────────────────────────────────────────
    st.markdown("### 👤 Gender Distribution")
    g1, g2 = st.columns(2)
    gender_counts = data["Gender"].value_counts()
    gender_labels = [("Male" if g == "M" else "Female") for g in gender_counts.index]

    with g1:
        fig_gbar = go.Figure(
            go.Bar(
                x=gender_labels,
                y=gender_counts.values,
                marker_color=[CHART_COLORS[0], CHART_COLORS[3]],
                text=gender_counts.values,
                textposition="outside",
                texttemplate="%{text:,}",
            )
        )
        fig_gbar.update_layout(**layout_kw, showlegend=False, height=350, yaxis_title="Count")
        st.plotly_chart(fig_gbar, use_container_width=True)

    with g2:
        fig_gpie = go.Figure(
            go.Pie(
                labels=gender_labels,
                values=gender_counts.values,
                marker=dict(colors=[CHART_COLORS[0], CHART_COLORS[3]]),
                hole=0.45,
                textinfo="label+percent",
            )
        )
        fig_gpie.update_layout(**layout_kw, showlegend=False, height=350)
        st.plotly_chart(fig_gpie, use_container_width=True)

    male_pct = round(gender_counts.get("M", 0) / gender_counts.sum() * 100, 1)
    st.markdown(
        insight_html(
            f"Male users dominate the platform at <b>{male_pct}%</b> of all ratings, "
            f"indicating a significant gender imbalance in the dataset."
        ),
        unsafe_allow_html=True,
    )

    # ── Age Group Distribution ────────────────────────────────────────────
    st.markdown("### 🎂 Age Group Distribution")
    if "AgeGroup" in data.columns:
        age_counts = data["AgeGroup"].value_counts().sort_index()
        fig_age = go.Figure(
            go.Bar(
                y=[str(a) for a in age_counts.index],
                x=age_counts.values,
                orientation="h",
                marker_color=CHART_COLORS[: len(age_counts)],
                text=age_counts.values,
                textposition="outside",
                texttemplate="%{text:,}",
            )
        )
        fig_age.update_layout(
            **layout_kw,
            xaxis_title="Number of Ratings",
            yaxis_title="Age Group",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown(
            insight_html(
                f"The <b>{most_active_age}</b> age group contributes the most ratings, "
                f"reflecting the core user demographic of the platform."
            ),
            unsafe_allow_html=True,
        )

    # ── Occupation Distribution ───────────────────────────────────────────
    st.markdown("### 💼 Occupation Distribution")
    if "OccupationLabel" in data.columns:
        occ_counts = data["OccupationLabel"].value_counts().head(15)
        fig_occ = go.Figure(
            go.Bar(
                y=occ_counts.index[::-1],
                x=occ_counts.values[::-1],
                orientation="h",
                marker_color=CHART_COLORS[: len(occ_counts)],
                text=occ_counts.values[::-1],
                textposition="outside",
                texttemplate="%{text:,}",
            )
        )
        fig_occ.update_layout(
            **layout_kw,
            xaxis_title="Number of Ratings",
            yaxis_title="Occupation",
            showlegend=False,
            height=500,
        )
        st.plotly_chart(fig_occ, use_container_width=True)
        st.markdown(
            insight_html(
                f"<b>{most_active_occ}</b> is the most active occupation group with "
                f"<b>{occ_count:,}</b> ratings, followed by other white-collar professions."
            ),
            unsafe_allow_html=True,
        )

    # ── Movies Released per Decade ────────────────────────────────────────
    st.markdown("### 📅 Movies Released per Decade")
    if "Decade" in data.columns:
        decade_counts = (
            data.drop_duplicates("Title")
            .groupby("Decade")
            .size()
            .sort_index()
        )
        fig_dec = go.Figure(
            go.Bar(
                x=[str(d) + "s" for d in decade_counts.index],
                y=decade_counts.values,
                marker_color=CHART_COLORS[: len(decade_counts)],
                text=decade_counts.values,
                textposition="outside",
                texttemplate="%{text:,}",
            )
        )
        fig_dec.update_layout(
            **layout_kw,
            xaxis_title="Decade",
            yaxis_title="Number of Movies",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_dec, use_container_width=True)
        peak_decade = decade_counts.idxmax()
        st.markdown(
            insight_html(
                f"The <b>{int(peak_decade)}s</b> saw the highest number of movie releases in the dataset, "
                f"reflecting the explosion of film production in that era."
            ),
            unsafe_allow_html=True,
        )

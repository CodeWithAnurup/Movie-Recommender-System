"""
SVD Analytics page module.
Visualises SVD model performance, prediction distributions,
2-D item / user embeddings, and latent-factor heatmaps.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from dashboard.styles import (
    inject_css, get_plotly_layout, get_theme, CHART_COLORS,
    metric_card_html, insight_html,
)
from dashboard.data_loader import load_data, get_movie_stats
from dashboard.models import train_svd_model, train_svd_2d


def render():
    """Main entry-point called by the dashboard router."""
    dark_mode = st.session_state.get("dark_mode", False)
    theme = get_theme(dark_mode)
    inject_css(dark_mode)

    st.markdown("## 🔬 SVD Model Analytics")

    # ── Train / load SVD model ─────────────────────────────────────────
    with st.spinner("Training SVD model (this may take a moment on first load)…"):
        model, trainset, predictions, rmse, mape, movies_df, actuals, preds = (
            train_svd_model(n_factors=4)
        )

    layout = get_plotly_layout(dark_mode)

    # ── Metric cards ───────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        metric_card_html("RMSE", f"{rmse:.4f}", icon="📉", sub="Root Mean Squared Error"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        metric_card_html("MAPE", f"{mape:.2f}%", icon="📊", sub="Mean Abs % Error"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        metric_card_html("Latent Factors", "4", icon="🧬", sub="d = 4"),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Actual vs Predicted & Error Distribution ───────────────────────
    st.markdown("### 📈 Prediction Quality")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scattergl(
                x=actuals,
                y=preds,
                mode="markers",
                marker=dict(color="#1a73e8", opacity=0.1, size=3),
                name="Predictions",
            )
        )
        # y = x reference line
        axis_min = min(float(np.min(actuals)), float(np.min(preds))) - 0.5
        axis_max = max(float(np.max(actuals)), float(np.max(preds))) + 0.5
        fig_scatter.add_trace(
            go.Scatter(
                x=[axis_min, axis_max],
                y=[axis_min, axis_max],
                mode="lines",
                line=dict(dash="dash", color="#e8710a", width=2),
                name="Perfect (y = x)",
            )
        )
        fig_scatter.update_layout(
            **layout,
            title="Actual vs Predicted Ratings",
            xaxis_title="Actual Rating",
            yaxis_title="Predicted Rating",
            height=420,
            showlegend=True,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        errors = actuals - preds
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=errors,
                nbinsx=50,
                marker_color="#ea4335",
                opacity=0.85,
                name="Error",
            )
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#34a853", line_width=2)
        fig_hist.update_layout(
            **layout,
            title="Prediction Error Distribution",
            xaxis_title="Error (Actual − Predicted)",
            yaxis_title="Count",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(
        insight_html(
            f"The model achieves an RMSE of <strong>{rmse:.4f}</strong>. "
            f"Errors are centred near zero with a slight positive skew, indicating "
            f"the model rarely over-predicts by large margins."
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── 2-D Embeddings ─────────────────────────────────────────────────
    st.markdown("### 🗺️ 2-D Embedding Space")

    with st.spinner("Computing 2-D projections…"):
        item_emb_2d, user_emb_2d, item_inner_ids, emb_movies_df, movie_stats = (
            train_svd_2d()
        )

    col_m, col_u = st.columns(2)

    # — Movie embeddings —
    with col_m:
        fig_items = go.Figure()
        fig_items.add_trace(
            go.Scattergl(
                x=item_emb_2d[:, 0],
                y=item_emb_2d[:, 1],
                mode="markers",
                marker=dict(color="#1a73e8", opacity=0.25, size=4),
                name="Movies",
                hoverinfo="skip",
            )
        )
        # Annotate top-10 popular movies
        top10 = movie_stats.head(10)
        for _, row in top10.iterrows():
            mid = row.name if "MovieID" not in row else row.get("MovieID", row.name)
            # Try to find inner id
            inner = item_inner_ids.get(mid)
            if inner is not None and inner < len(item_emb_2d):
                fig_items.add_annotation(
                    x=float(item_emb_2d[inner, 0]),
                    y=float(item_emb_2d[inner, 1]),
                    text=row["Title"][:25],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=0.8,
                    font=dict(size=9, color=theme["text"]),
                    bgcolor=theme["card_bg"],
                    opacity=0.85,
                )
        fig_items.update_layout(
            **layout,
            title="Movie Embeddings (PCA 2-D)",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            height=460,
            showlegend=False,
        )
        st.plotly_chart(fig_items, use_container_width=True)

    # — User embeddings (sample 1 000) —
    with col_u:
        n_users = user_emb_2d.shape[0]
        sample_size = min(1000, n_users)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_users, size=sample_size, replace=False)

        fig_users = go.Figure()
        fig_users.add_trace(
            go.Scattergl(
                x=user_emb_2d[idx, 0],
                y=user_emb_2d[idx, 1],
                mode="markers",
                marker=dict(
                    color="#34a853",
                    opacity=0.3,
                    size=4,
                ),
                name="Users",
            )
        )
        fig_users.update_layout(
            **layout,
            title="User Embeddings (PCA 2-D, sample 1 000)",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            height=460,
            showlegend=False,
        )
        st.plotly_chart(fig_users, use_container_width=True)

    st.markdown(
        insight_html(
            "Movies that cluster together in embedding space share latent features "
            "(e.g. genre, tone, era). Users near those clusters tend to prefer those films."
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Latent Factor Heatmap ──────────────────────────────────────────
    st.markdown("### 🧩 Latent Factor Profiles — Top 20 Movies")

    _, _, _, data = load_data()
    stats = get_movie_stats(data)
    top20 = stats.head(20)

    # Build mapping: raw MovieID → inner item id
    raw_to_inner = {trainset.to_raw_iid(i): i for i in trainset.all_items()}

    # Resolve titles → MovieIDs via movies_df
    title_to_mid = dict(zip(movies_df["Title"], movies_df["MovieID"]))

    labels = []
    vectors = []
    for _, row in top20.iterrows():
        title = row["Title"]
        mid = title_to_mid.get(title)
        if mid is None:
            continue
        inner = raw_to_inner.get(mid)
        if inner is None:
            continue
        labels.append(title[:35])
        vectors.append(model.qi[inner])

    if vectors:
        mat = np.array(vectors)
        fig_heat = px.imshow(
            mat,
            x=[f"Factor {i+1}" for i in range(mat.shape[1])],
            y=labels,
            color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        fig_heat.update_layout(
            **layout,
            title="Item Latent Factors (qi) — Top 20 Movies",
            height=560,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown(
            insight_html(
                "Each row is a movie's latent-factor vector. Similar colour patterns across "
                "rows indicate movies the model considers alike — even if they span different genres."
            ),
            unsafe_allow_html=True,
        )
    else:
        st.warning("Could not map top movies to latent factors.")

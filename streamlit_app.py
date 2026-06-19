"""Movie Recommender System — Streamlit Dashboard (entry point)."""

import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ───────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# ── Inject CSS ───────────────────────────────────────────────────
from dashboard.styles import inject_css  # noqa: E402

inject_css(st.session_state.dark_mode)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 Movie Recommender")
    st.caption("MovieLens 1M · Collaborative Filtering")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🔍 Exploratory Analysis",
            "🎯 Recommendations",
            "👤 User-Based",
            "🧮 SVD Analytics",
            "💼 Business Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.divider()
    st.caption("Built with Streamlit · Plotly")
    st.caption("Author: Anurup Samanta")

# ── Page routing ─────────────────────────────────────────────────
import re

# Clean page name (remove leading emojis/symbols) for robust routing comparison
clean_page = re.sub(r"^[^\w\s]+", "", page).strip()

if clean_page == "Overview":
    from dashboard.page_overview import render
    render()
elif clean_page == "Exploratory Analysis":
    from dashboard.page_eda import render
    render()
elif clean_page == "Recommendations":
    from dashboard.page_recommendations import render
    render()
elif clean_page == "User-Based":
    from dashboard.page_user_based import render
    render()
elif clean_page == "SVD Analytics":
    from dashboard.page_svd import render
    render()
elif clean_page == "Business Insights":
    from dashboard.page_business import render
    render()


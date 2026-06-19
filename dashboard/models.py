"""Cached model building: pivot table, similarity matrices, KNN, SVD."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from dashboard.data_loader import load_data


# ── Pivot Table ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building movie-user matrix …")
def build_pivot_table():
    """Return (pivot_raw, pivot_filled_0).  ~3706×6040."""
    _, _, _, data = load_data()
    pivot = data.pivot_table(index="Title", columns="UserID", values="Rating")
    filled = pivot.fillna(0)
    return pivot, filled


# ── Item Similarity (Cosine) ────────────────────────────────────
@st.cache_resource(show_spinner="Computing similarity matrix …")
def build_similarity_matrix():
    """Return item–item cosine-similarity DataFrame."""
    _, filled = build_pivot_table()
    sim = cosine_similarity(filled)
    return pd.DataFrame(sim, index=filled.index, columns=filled.index)


# ── KNN Model ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Fitting KNN model …")
def build_knn_model():
    """Return (knn_model, csr_data, movie_index)."""
    _, filled = build_pivot_table()
    csr = csr_matrix(filled.values)
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20)
    knn.fit(csr)
    return knn, csr, filled.index


# ── SVD (d = 4) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training SVD model (d=4) …")
def train_svd_model(n_factors: int = 4):
    """Return (model, trainset, testset, predictions, movies_df)."""
    _, _, movies, data = load_data()
    from surprise import Dataset, Reader, SVD, accuracy
    from surprise.model_selection import train_test_split

    ratings_mf = data[["UserID", "MovieID", "Rating"]].copy()
    reader = Reader(rating_scale=(1, 5))
    sd = Dataset.load_from_df(ratings_mf, reader)
    trainset, testset = train_test_split(sd, test_size=0.2, random_state=42)

    model = SVD(n_factors=n_factors, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)

    # Pre-compute RMSE / MAPE
    actuals = np.array([p.r_ui for p in predictions])
    preds = np.array([p.est for p in predictions])
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mape = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)

    return model, trainset, predictions, rmse, mape, movies, actuals, preds


# ── SVD (d = 2) for visualisation ───────────────────────────────
@st.cache_resource(show_spinner="Training SVD model (d=2) …")
def train_svd_2d():
    """Return (item_emb_2d, user_emb_2d, item_inner_ids, movies_df, movie_stats)."""
    _, _, movies, data = load_data()
    from dashboard.data_loader import get_movie_stats
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split

    ratings_mf = data[["UserID", "MovieID", "Rating"]].copy()
    reader = Reader(rating_scale=(1, 5))
    sd = Dataset.load_from_df(ratings_mf, reader)
    trainset, _ = train_test_split(sd, test_size=0.2, random_state=42)

    model = SVD(n_factors=2, random_state=42)
    model.fit(trainset)

    item_emb = model.qi
    user_emb = model.pu
    item_inner_ids = {trainset.to_raw_iid(i): i for i in trainset.all_items()}

    stats = get_movie_stats(data)
    return item_emb, user_emb, item_inner_ids, movies, stats

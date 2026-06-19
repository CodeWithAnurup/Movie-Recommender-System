"""Recommendation functions: Pearson, Cosine, KNN, Hybrid, User-Based, SVD."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


# ── 1. Pearson Correlation (Item-Based) ──────────────────────────
def recommend_pearson(
    movie_name: str,
    pivot_raw: pd.DataFrame,
    n: int = 5,
    min_ratings: int = 100,
) -> pd.DataFrame:
    """Recommend movies using Pearson correlation on the raw pivot table."""
    movie_ratings = pivot_raw.loc[movie_name]
    correlations = pivot_raw.corrwith(movie_ratings)

    corr_df = pd.DataFrame({"Correlation": correlations, "Title": correlations.index})
    num_ratings = pivot_raw.notna().sum(axis=1)
    corr_df["NumRatings"] = num_ratings.values

    corr_df = corr_df.dropna()
    corr_df = corr_df[corr_df["Title"] != movie_name]
    corr_df = corr_df[corr_df["NumRatings"] >= min_ratings]
    corr_df = corr_df.sort_values("Correlation", ascending=False)
    return corr_df.head(n).reset_index(drop=True)


# ── 2. Cosine Similarity (pre-computed matrix) ──────────────────
def recommend_cosine(
    movie_name: str,
    similarity_df: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """Recommend movies from the pre-computed item-similarity matrix."""
    scores = similarity_df[movie_name].drop(movie_name)
    top = scores.sort_values(ascending=False).head(n)
    return pd.DataFrame(
        {"Title": top.index, "Similarity": top.values}
    ).reset_index(drop=True)


# ── 3. KNN (Cosine distance) ────────────────────────────────────
def recommend_knn(
    movie_name: str,
    knn_model,
    csr_data,
    movie_index,
    n: int = 5,
) -> pd.DataFrame:
    """Recommend movies using a pre-fitted KNN model."""
    idx = list(movie_index).index(movie_name)
    distances, indices = knn_model.kneighbors(
        csr_data[idx], n_neighbors=n + 1
    )
    rows = []
    for i in range(1, len(indices[0])):
        rows.append(
            {
                "Title": movie_index[indices[0][i]],
                "Distance": distances[0][i],
                "Similarity": 1 - distances[0][i],
            }
        )
    return pd.DataFrame(rows)


# ── 4. Hybrid (Pearson + Cosine + KNN weighted) ─────────────────
def recommend_hybrid(
    movie_name: str,
    pivot_raw: pd.DataFrame,
    similarity_df: pd.DataFrame,
    knn_model,
    csr_data,
    movie_index,
    n: int = 5,
    weights: tuple = (0.35, 0.35, 0.30),
) -> pd.DataFrame:
    """Blend three methods' normalised scores into a single ranking."""
    w_p, w_c, w_k = weights

    try:
        pearson = recommend_pearson(movie_name, pivot_raw, n=50, min_ratings=50)
        pearson = pearson.rename(columns={"Correlation": "score_p"})
        pearson["score_p"] = (pearson["score_p"] - pearson["score_p"].min()) / (
            pearson["score_p"].max() - pearson["score_p"].min() + 1e-9
        )
    except Exception:
        pearson = pd.DataFrame(columns=["Title", "score_p"])

    cosine = recommend_cosine(movie_name, similarity_df, n=50)
    cosine = cosine.rename(columns={"Similarity": "score_c"})
    cosine["score_c"] = (cosine["score_c"] - cosine["score_c"].min()) / (
        cosine["score_c"].max() - cosine["score_c"].min() + 1e-9
    )

    knn = recommend_knn(movie_name, knn_model, csr_data, movie_index, n=50)
    knn = knn.rename(columns={"Similarity": "score_k"})
    knn["score_k"] = (knn["score_k"] - knn["score_k"].min()) / (
        knn["score_k"].max() - knn["score_k"].min() + 1e-9
    )

    merged = (
        cosine[["Title", "score_c"]]
        .merge(knn[["Title", "score_k"]], on="Title", how="outer")
        .merge(pearson[["Title", "score_p"]], on="Title", how="outer")
        .fillna(0)
    )
    merged["HybridScore"] = (
        w_p * merged["score_p"] + w_c * merged["score_c"] + w_k * merged["score_k"]
    )
    merged = merged.sort_values("HybridScore", ascending=False).head(n)
    return merged[["Title", "HybridScore"]].reset_index(drop=True)


# ── 5. User-Based Collaborative Filtering ───────────────────────
def recommend_user_based(
    new_user_ratings: dict,
    data: pd.DataFrame,
    n: int = 10,
) -> tuple:
    """
    Given {title: rating, ...}, find similar users and recommend.
    Returns (similar_users_df, recommendations_df).
    """
    new_user_movies = set(new_user_ratings.keys())
    users_who_watched = (
        data[data["Title"].isin(new_user_movies)]
        .groupby("UserID")["Title"]
        .apply(set)
    )
    common_counts = users_who_watched.apply(
        lambda x: len(x.intersection(new_user_movies))
    ).sort_values(ascending=False)

    top_users = common_counts.head(100).index.tolist()

    similarity_scores = {}
    for uid in top_users:
        user_ratings = data[data["UserID"] == uid].set_index("Title")["Rating"]
        common = new_user_movies.intersection(set(user_ratings.index))
        if len(common) >= 3:
            new_vals = [new_user_ratings[m] for m in common]
            old_vals = [user_ratings[m] for m in common]
            if np.std(new_vals) > 0 and np.std(old_vals) > 0:
                corr, _ = stats.pearsonr(new_vals, old_vals)
                similarity_scores[uid] = corr

    sim_df = (
        pd.DataFrame(
            list(similarity_scores.items()), columns=["UserID", "Similarity"]
        )
        .sort_values("Similarity", ascending=False)
    )

    top_10 = sim_df.head(10)
    if top_10.empty:
        return top_10, pd.DataFrame(columns=["Title", "AvgWeightedScore", "NumRaters"])

    top_ids = top_10["UserID"].tolist()
    top_sims = dict(zip(top_10["UserID"], top_10["Similarity"]))

    sur = data[data["UserID"].isin(top_ids)][["UserID", "Title", "Rating"]].copy()
    sur["Similarity"] = sur["UserID"].map(top_sims)
    sur["WeightedRating"] = sur["Rating"] * sur["Similarity"]

    wrecs = (
        sur.groupby("Title")
        .agg(
            TotalWeightedRating=("WeightedRating", "sum"),
            TotalSimilarity=("Similarity", "sum"),
            NumRaters=("UserID", "nunique"),
        )
        .reset_index()
    )
    wrecs["AvgWeightedScore"] = wrecs["TotalWeightedRating"] / wrecs["TotalSimilarity"]
    wrecs = wrecs[~wrecs["Title"].isin(new_user_ratings.keys())]
    recs = wrecs.sort_values("AvgWeightedScore", ascending=False).head(n)

    return top_10.reset_index(drop=True), recs.reset_index(drop=True)

"""Data loading, merging, and feature engineering (all cached)."""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

AGE_LABELS = {
    1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
    45: "45-49", 50: "50-55", 56: "56+",
}

OCCUPATION_LABELS = {
    0: "Other", 1: "Academic/Educator", 2: "Artist", 3: "Clerical/Admin",
    4: "College/Grad Student", 5: "Customer Service", 6: "Doctor/Healthcare",
    7: "Executive/Managerial", 8: "Farmer", 9: "Homemaker", 10: "K-12 Student",
    11: "Lawyer", 12: "Programmer", 13: "Retired", 14: "Sales/Marketing",
    15: "Scientist", 16: "Self-Employed", 17: "Technician/Engineer",
    18: "Tradesman/Craftsman", 19: "Unemployed", 20: "Writer",
}


@st.cache_data(show_spinner="Loading MovieLens dataset …")
def load_data():
    """Load & merge ratings, users, movies; returns (ratings, users, movies, merged)."""
    ratings = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep="::", engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="ISO-8859-1",
    )
    users = pd.read_csv(
        DATA_DIR / "users.dat",
        sep="::", engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        encoding="ISO-8859-1",
    )
    movies = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::", engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="ISO-8859-1",
    )

    data = pd.merge(ratings, users, on="UserID")
    data = pd.merge(data, movies, on="MovieID")

    # Feature engineering
    data["AgeGroup"] = data["Age"].map(AGE_LABELS)
    data["OccupationLabel"] = data["Occupation"].map(OCCUPATION_LABELS)
    data["ReleaseYear"] = (
        data["Title"].str.extract(r"\((\d{4})\)")[0].astype(float)
    )
    data["Decade"] = (data["ReleaseYear"] // 10 * 10).astype("Int64")
    data["RatingDate"] = pd.to_datetime(data["Timestamp"], unit="s")
    data["RatingYear"] = data["RatingDate"].dt.year
    data["RatingMonth"] = data["RatingDate"].dt.month

    return ratings, users, movies, data


@st.cache_data(show_spinner=False)
def get_movie_stats(_data: pd.DataFrame) -> pd.DataFrame:
    """Per-movie aggregates: AvgRating, NumRatings."""
    stats = (
        _data.groupby("Title")
        .agg(AvgRating=("Rating", "mean"), NumRatings=("Rating", "count"))
        .reset_index()
        .sort_values("NumRatings", ascending=False)
    )
    return stats


@st.cache_data(show_spinner=False)
def get_genre_counts(_data: pd.DataFrame) -> pd.Series:
    """Explode pipe-separated genres and return value-counts."""
    return _data["Genres"].str.split("|").explode().value_counts()


@st.cache_data(show_spinner=False)
def get_movie_titles(_data: pd.DataFrame) -> list:
    """Return a sorted list of unique movie titles for search widgets."""
    return sorted(_data["Title"].unique().tolist())

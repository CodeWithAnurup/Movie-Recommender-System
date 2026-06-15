# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🎬 Movie Recommender System
# ### Personalized Movie Recommendations using Collaborative Filtering
# 
# **Author:** Pritam Palit  
# **Dataset:** MovieLens 1M (GroupLens Research)  
# **Techniques:** Pearson Correlation | Cosine Similarity (KNN) | Matrix Factorization (SVD)
# 
# ---

# %% [markdown]
# ## 📑 Table of Contents
# 
# 1. [Problem Statement](#1)
# 2. [Import Libraries](#2)
# 3. [Data Loading & Formatting](#3)
# 4. [Exploratory Data Analysis (EDA)](#4)
# 5. [Feature Engineering](#5)
# 6. [Data Grouping & Analysis](#6)
# 7. [Recommender System — Pearson Correlation (Item-Based)](#7)
# 8. [Recommender System — Cosine Similarity (KNN)](#8)
# 9. [Recommender System — Matrix Factorization](#9)
# 10. [Recommender System — User-Based Approach (Bonus)](#10)
# 11. [Questionnaire Answers](#11)
# 12. [Conclusion](#12)

# %% [markdown]
# <a id='1'></a>
# ## 1. 🎯 Problem Statement
# 
# **Objective:** Create a Recommender System to show personalized movie recommendations based on ratings given by a user and other users similar to them, in order to improve user experience.
# 
# **Approaches Used:**
# - **Item-Based Collaborative Filtering** using Pearson Correlation
# - **Item-Based Collaborative Filtering** using Cosine Similarity & KNN
# - **Matrix Factorization** using cmfrec library
# - **User-Based Collaborative Filtering** using Pearson Correlation (Bonus)
# 
# **Dataset:** MovieLens 1M dataset containing 1,000,209 ratings from 6,040 users on 3,952 movies.

# %% [markdown]
# <a id='2'></a>
# ## 2. 📚 Import Libraries

# %%
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis
from scipy import stats
from scipy.sparse import csr_matrix

# Machine Learning
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Matrix Factorization (Surprise)
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

print("✅ All libraries imported successfully!")

# %% [markdown]
# <a id='3'></a>
# ## 3. 📂 Data Loading & Formatting
# 
# The dataset consists of three `.dat` files:
# - `ratings.dat` — UserID::MovieID::Rating::Timestamp
# - `users.dat` — UserID::Gender::Age::Occupation::Zip-code
# - `movies.dat` — MovieID::Title::Genres

# %% [markdown]
# ### 3.1 Loading the Data Files

# %%
# Load Ratings Data
ratings = pd.read_csv(
    'data/ratings.dat',
    sep='::',
    engine='python',
    names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
    encoding='ISO-8859-1'
)
print(f"📊 Ratings shape: {ratings.shape}")
ratings.head()

# %%
# Load Users Data
users = pd.read_csv(
    'data/users.dat',
    sep='::',
    engine='python',
    names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
    encoding='ISO-8859-1'
)
print(f"👤 Users shape: {users.shape}")
users.head()

# %%
# Load Movies Data
movies = pd.read_csv(
    'data/movies.dat',
    sep='::',
    engine='python',
    names=['MovieID', 'Title', 'Genres'],
    encoding='ISO-8859-1'
)
print(f"🎬 Movies shape: {movies.shape}")
movies.head()

# %% [markdown]
# ### 3.2 Merging Data into a Single DataFrame

# %%
# Merge ratings with users
data = pd.merge(ratings, users, on='UserID')

# Merge with movies
data = pd.merge(data, movies, on='MovieID')

print(f"✅ Merged DataFrame shape: {data.shape}")
print(f"📋 Columns: {list(data.columns)}")
data.head()

# %%
# Dataset info
data.info()

# %%
# Statistical summary
data.describe()

# %% [markdown]
# <a id='4'></a>
# ## 4. 🔍 Exploratory Data Analysis (EDA)
# 
# Let's explore the dataset to understand the distribution of ratings, user demographics, and movie characteristics.

# %% [markdown]
# ### 4.1 Data Structure & Cleaning

# %%
# Check for missing values
print("🔎 Missing Values:")
print(data.isnull().sum())
print(f"\n📊 Total missing values: {data.isnull().sum().sum()}")

# %%
# Check for duplicates
duplicates = data.duplicated().sum()
print(f"🔄 Duplicate rows: {duplicates}")

# %%
# Unique counts
print(f"👤 Unique Users: {data['UserID'].nunique()}")
print(f"🎬 Unique Movies: {data['MovieID'].nunique()}")
print(f"⭐ Unique Ratings: {sorted(data['Rating'].unique())}")

# %% [markdown]
# ### 4.2 Rating Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Rating count
rating_counts = data['Rating'].value_counts().sort_index()
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
axes[0].bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_xlabel('Rating', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('📊 Distribution of Ratings', fontsize=14, fontweight='bold')
for i, (x, y) in enumerate(zip(rating_counts.index, rating_counts.values)):
    axes[0].text(x, y + 5000, f'{y:,}', ha='center', fontweight='bold', fontsize=9)

# Rating percentage
axes[1].pie(rating_counts.values, labels=[f'{r} ⭐' for r in rating_counts.index],
            autopct='%1.1f%%', colors=colors, startangle=90, explode=[0.02]*5,
            textprops={'fontsize': 11})
axes[1].set_title('📊 Rating Distribution (%)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('images/rating_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.3 Gender Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gender distribution of users
gender_users = data.groupby('Gender')['UserID'].nunique()
gender_labels = {'M': 'Male', 'F': 'Female'}
colors_gender = ['#3498db', '#e91e63']

axes[0].bar([gender_labels[g] for g in gender_users.index], gender_users.values,
            color=colors_gender, edgecolor='white', linewidth=1.5, width=0.5)
axes[0].set_title('👤 Users by Gender', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Users', fontsize=12)
for i, (x, y) in enumerate(zip(range(len(gender_users)), gender_users.values)):
    axes[0].text(x, y + 50, f'{y:,}', ha='center', fontweight='bold', fontsize=11)

# Ratings by gender
gender_ratings = data.groupby('Gender')['Rating'].count()
axes[1].pie(gender_ratings.values, labels=[gender_labels[g] for g in gender_ratings.index],
            autopct='%1.1f%%', colors=colors_gender, startangle=90, explode=[0.03, 0.03],
            textprops={'fontsize': 12})
axes[1].set_title('📊 Ratings by Gender', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('images/gender_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Most users who rated movies are Male: {gender_users['M'] > gender_users['F']}")

# %% [markdown]
# ### 4.4 Age Group Analysis

# %%
# Map age codes to labels
age_labels = {1: 'Under 18', 18: '18-24', 25: '25-34', 35: '35-44', 
              45: '45-49', 50: '50-55', 56: '56+'}
data['AgeGroup'] = data['Age'].map(age_labels)

# Ratings by age group
age_ratings = data.groupby('AgeGroup')['Rating'].count().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
colors_age = sns.color_palette('viridis', len(age_ratings))
bars = ax.barh(age_ratings.index, age_ratings.values, color=colors_age, edgecolor='white', linewidth=1.5)
ax.set_xlabel('Number of Ratings', fontsize=12)
ax.set_title('📊 Number of Ratings by Age Group', fontsize=14, fontweight='bold')
for bar, val in zip(bars, age_ratings.values):
    ax.text(val + 5000, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('images/age_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n🏆 Age group with most ratings: {age_ratings.index[0]} ({age_ratings.values[0]:,} ratings)")

# %% [markdown]
# ### 4.5 Occupation Analysis

# %%
# Map occupation codes to labels
occupation_labels = {
    0: 'Other', 1: 'Academic/Educator', 2: 'Artist', 3: 'Clerical/Admin',
    4: 'College/Grad Student', 5: 'Customer Service', 6: 'Doctor/Healthcare',
    7: 'Executive/Managerial', 8: 'Farmer', 9: 'Homemaker', 10: 'K-12 Student',
    11: 'Lawyer', 12: 'Programmer', 13: 'Retired', 14: 'Sales/Marketing',
    15: 'Scientist', 16: 'Self-Employed', 17: 'Technician/Engineer',
    18: 'Tradesman/Craftsman', 19: 'Unemployed', 20: 'Writer'
}
data['OccupationLabel'] = data['Occupation'].map(occupation_labels)

# Top 10 occupations by ratings
occ_ratings = data.groupby('OccupationLabel')['Rating'].count().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
colors_occ = sns.color_palette('magma', len(occ_ratings))
bars = ax.barh(occ_ratings.index, occ_ratings.values, color=colors_occ, edgecolor='white', linewidth=1)
ax.set_xlabel('Number of Ratings', fontsize=12)
ax.set_title('🏢 Number of Ratings by Occupation', fontsize=14, fontweight='bold')
for bar, val in zip(bars, occ_ratings.values):
    ax.text(val + 1000, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('images/occupation_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n🏆 Occupation with most ratings: {occ_ratings.index[-1]} ({occ_ratings.values[-1]:,} ratings)")

# %% [markdown]
# <a id='5'></a>
# ## 5. 🛠️ Feature Engineering

# %% [markdown]
# ### 5.1 Extract Release Year from Movie Title

# %%
# Extract release year from title (format: "Movie Name (Year)")
data['ReleaseYear'] = data['Title'].str.extract(r'\((\d{4})\)').astype(float)

# Check the distribution of release years
print(f"📅 Release Year Range: {int(data['ReleaseYear'].min())} - {int(data['ReleaseYear'].max())}")
print(f"📊 Movies with missing year: {data['ReleaseYear'].isnull().sum()}")

# %%
# Derive Decade
data['Decade'] = (data['ReleaseYear'] // 10 * 10).astype('Int64')

# Movies per decade
decade_counts = data.groupby('Decade')['MovieID'].nunique().sort_index()

fig, ax = plt.subplots(figsize=(12, 6))
colors_decade = sns.color_palette('coolwarm', len(decade_counts))
bars = ax.bar(decade_counts.index.astype(str), decade_counts.values, color=colors_decade, 
              edgecolor='white', linewidth=1.5, width=0.7)
ax.set_xlabel('Decade', fontsize=12)
ax.set_ylabel('Number of Unique Movies', fontsize=12)
ax.set_title('🎬 Number of Movies Released per Decade', fontsize=14, fontweight='bold')
for bar, val in zip(bars, decade_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 10, str(val), ha='center', fontweight='bold', fontsize=9)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/movies_per_decade.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n🏆 Most movies released in the: {int(decade_counts.idxmax())}s ({decade_counts.max()} movies)")

# %% [markdown]
# ### 5.2 Genre Analysis

# %%
# Split genres (pipe-separated) and count
all_genres = data['Genres'].str.split('|').explode()
genre_counts = all_genres.value_counts()

fig, ax = plt.subplots(figsize=(12, 6))
colors_genre = sns.color_palette('Set2', len(genre_counts))
bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1], color=colors_genre, edgecolor='white')
ax.set_xlabel('Number of Ratings', fontsize=12)
ax.set_title('🎭 Ratings by Genre', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/genre_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 5.3 Convert Timestamp

# %%
# Convert timestamp to datetime
data['RatingDate'] = pd.to_datetime(data['Timestamp'], unit='s')
data['RatingYear'] = data['RatingDate'].dt.year
data['RatingMonth'] = data['RatingDate'].dt.month

print(f"📅 Ratings date range: {data['RatingDate'].min()} to {data['RatingDate'].max()}")
data[['UserID', 'Title', 'Rating', 'RatingDate']].head()

# %% [markdown]
# <a id='6'></a>
# ## 6. 📊 Data Grouping — Average Rating & Number of Ratings

# %%
# Group by movie: Average Rating and Number of Ratings
movie_stats = data.groupby('Title').agg(
    AvgRating=('Rating', 'mean'),
    NumRatings=('Rating', 'count')
).reset_index()

movie_stats = movie_stats.sort_values('NumRatings', ascending=False)

print("🏆 Top 20 Most Rated Movies:")
movie_stats.head(20)

# %%
# Visualize Average Rating vs Number of Ratings
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Distribution of Average Ratings
axes[0].hist(movie_stats['AvgRating'], bins=30, color='#3498db', edgecolor='white', linewidth=1.2)
axes[0].set_xlabel('Average Rating', fontsize=12)
axes[0].set_ylabel('Number of Movies', fontsize=12)
axes[0].set_title('📊 Distribution of Average Movie Ratings', fontsize=14, fontweight='bold')
axes[0].axvline(movie_stats['AvgRating'].mean(), color='red', linestyle='--', label=f"Mean: {movie_stats['AvgRating'].mean():.2f}")
axes[0].legend()

# Distribution of Number of Ratings
axes[1].hist(movie_stats['NumRatings'], bins=50, color='#e74c3c', edgecolor='white', linewidth=1.2)
axes[1].set_xlabel('Number of Ratings', fontsize=12)
axes[1].set_ylabel('Number of Movies', fontsize=12)
axes[1].set_title('📊 Distribution of Number of Ratings per Movie', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('images/rating_stats.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Scatter plot: Average Rating vs Number of Ratings
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(movie_stats['NumRatings'], movie_stats['AvgRating'], 
                     alpha=0.4, c=movie_stats['AvgRating'], cmap='viridis', s=20)
ax.set_xlabel('Number of Ratings', fontsize=12)
ax.set_ylabel('Average Rating', fontsize=12)
ax.set_title('⭐ Average Rating vs Number of Ratings', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Average Rating')

# Annotate top movies
top_movies = movie_stats.head(5)
for _, row in top_movies.iterrows():
    ax.annotate(row['Title'][:25], (row['NumRatings'], row['AvgRating']),
                fontsize=8, alpha=0.8, arrowprops=dict(arrowstyle='->', alpha=0.5))

plt.tight_layout()
plt.savefig('images/avg_vs_num_ratings.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Movie with maximum number of ratings
max_rated = movie_stats.iloc[0]
print(f"\n🏆 Movie with maximum ratings: {max_rated['Title']}")
print(f"   📊 Number of Ratings: {int(max_rated['NumRatings'])}")
print(f"   ⭐ Average Rating: {max_rated['AvgRating']:.2f}")

# %% [markdown]
# <a id='7'></a>
# ## 7. 🔗 Recommender System — Pearson Correlation (Item-Based)
# 
# In the **Item-Based approach**, we find movies similar to a given movie based on how users have rated them.
# We use **Pearson Correlation** to measure similarity between movies.

# %% [markdown]
# ### 7.1 Create Pivot Table

# %%
# Create pivot table: MovieTitle × UserID
movie_user_pivot = data.pivot_table(index='Title', columns='UserID', values='Rating')
print(f"📊 Pivot Table Shape: {movie_user_pivot.shape}")
print(f"   Rows (Movies): {movie_user_pivot.shape[0]}")
print(f"   Columns (Users): {movie_user_pivot.shape[1]}")
print(f"   Sparsity: {(movie_user_pivot.isnull().sum().sum() / (movie_user_pivot.shape[0] * movie_user_pivot.shape[1])) * 100:.2f}%")

# %%
# Impute NaN values with 0
movie_user_filled = movie_user_pivot.fillna(0)
movie_user_filled.head()

# %% [markdown]
# ### 7.2 Pearson Correlation-based Recommendation Function

# %%
def recommend_movies_pearson(movie_name, pivot_table, n_recommendations=5, min_ratings=100):
    """
    Recommend similar movies using Pearson Correlation.
    
    Parameters:
    -----------
    movie_name : str - Name of the movie to find similarities for
    pivot_table : DataFrame - Movie-User pivot table (NOT filled with 0s)
    n_recommendations : int - Number of recommendations to return
    min_ratings : int - Minimum number of ratings required for a movie
    
    Returns:
    --------
    DataFrame with similar movies and their correlation scores
    """
    # Get the ratings for the input movie
    movie_ratings = pivot_table.loc[movie_name]
    
    # Calculate Pearson correlation with all other movies
    correlations = pivot_table.corrwith(movie_ratings)
    
    # Create a DataFrame with correlations
    corr_df = pd.DataFrame({'Correlation': correlations, 'Title': correlations.index})
    
    # Add number of ratings
    num_ratings = pivot_table.notna().sum(axis=1)
    corr_df['NumRatings'] = num_ratings
    
    # Filter: remove NaN correlations, the movie itself, and low-rated movies
    corr_df = corr_df.dropna()
    corr_df = corr_df[corr_df['Title'] != movie_name]
    corr_df = corr_df[corr_df['NumRatings'] >= min_ratings]
    
    # Sort by correlation (descending)
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    return corr_df.head(n_recommendations)

# %% [markdown]
# ### 7.3 Get Recommendations

# %%
# Recommend movies similar to 'Liar Liar (1997)'
print("=" * 60)
print("🎬 Movies Similar to 'Liar Liar (1997)'")
print("   (Using Pearson Correlation — Item-Based Approach)")
print("=" * 60)

liar_liar_recs = recommend_movies_pearson('Liar Liar (1997)', movie_user_pivot, n_recommendations=5)
print(liar_liar_recs[['Title', 'Correlation', 'NumRatings']].to_string(index=False))

# %%
# Recommend movies similar to 'Star Wars: Episode IV - A New Hope (1977)'
print("\n" + "=" * 60)
print("🎬 Movies Similar to 'Star Wars: Episode IV - A New Hope (1977)'")
print("   (Using Pearson Correlation — Item-Based Approach)")
print("=" * 60)

star_wars_recs = recommend_movies_pearson('Star Wars: Episode IV - A New Hope (1977)', movie_user_pivot, n_recommendations=5)
print(star_wars_recs[['Title', 'Correlation', 'NumRatings']].to_string(index=False))

# %%
# Recommend movies similar to 'Toy Story (1995)'
print("\n" + "=" * 60)
print("🎬 Movies Similar to 'Toy Story (1995)'")
print("   (Using Pearson Correlation — Item-Based Approach)")
print("=" * 60)

toy_story_recs = recommend_movies_pearson('Toy Story (1995)', movie_user_pivot, n_recommendations=5)
print(toy_story_recs[['Title', 'Correlation', 'NumRatings']].to_string(index=False))

# %% [markdown]
# <a id='8'></a>
# ## 8. 📐 Recommender System — Cosine Similarity (KNN)
# 
# In this section, we use **Cosine Similarity** to measure the angle between movie rating vectors,
# and apply the **K-Nearest Neighbors (KNN)** algorithm for recommendations.

# %% [markdown]
# ### 8.1 Item Similarity Matrix

# %%
# Compute Item-Item Cosine Similarity Matrix
item_similarity = cosine_similarity(movie_user_filled)
item_similarity_df = pd.DataFrame(item_similarity, index=movie_user_filled.index, columns=movie_user_filled.index)

print("📊 Item Similarity Matrix Shape:", item_similarity_df.shape)
print("\n🔍 Sample Item Similarity Matrix (first 5 movies):")
item_similarity_df.iloc[:5, :5]

# %% [markdown]
# ### 8.2 User Similarity Matrix

# %%
# Compute User-User Cosine Similarity Matrix
user_movie_filled = movie_user_filled.T  # Transpose to get Users × Movies
user_similarity = cosine_similarity(user_movie_filled)
user_similarity_df = pd.DataFrame(user_similarity, 
                                   index=movie_user_filled.columns, 
                                   columns=movie_user_filled.columns)

print("📊 User Similarity Matrix Shape:", user_similarity_df.shape)
print("\n🔍 Sample User Similarity Matrix (first 5 users):")
user_similarity_df.iloc[:5, :5]

# %% [markdown]
# ### 8.3 Create CSR (Compressed Sparse Row) Matrix

# %%
# Create CSR matrix from the pivot table for efficient computation
csr_data = csr_matrix(movie_user_filled.values)
print(f"📊 CSR Matrix Shape: {csr_data.shape}")
print(f"   Non-zero elements: {csr_data.nnz}")
print(f"   Sparsity: {(1 - csr_data.nnz / (csr_data.shape[0] * csr_data.shape[1])) * 100:.2f}%")

# %% [markdown]
# ### 8.4 Recommendation Function using Cosine Similarity

# %%
def recommend_movies_cosine(movie_name, similarity_df, n_recommendations=5):
    """
    Recommend similar movies using pre-computed Cosine Similarity matrix.
    
    Parameters:
    -----------
    movie_name : str - Name of the movie
    similarity_df : DataFrame - Item similarity matrix
    n_recommendations : int - Number of recommendations
    
    Returns:
    --------
    DataFrame with similar movies and similarity scores
    """
    # Get similarity scores for the input movie
    sim_scores = similarity_df[movie_name].drop(movie_name)
    
    # Sort by similarity (descending) and get top N
    top_similar = sim_scores.sort_values(ascending=False).head(n_recommendations)
    
    result = pd.DataFrame({
        'Movie': top_similar.index,
        'Cosine Similarity': top_similar.values
    })
    
    return result

# %%
# Test the cosine similarity recommendation function
print("=" * 60)
print("🎬 Top 5 Movies Similar to 'Toy Story (1995)'")
print("   (Using Cosine Similarity)")
print("=" * 60)
cosine_recs = recommend_movies_cosine('Toy Story (1995)', item_similarity_df)
print(cosine_recs.to_string(index=False))

# %% [markdown]
# ### 8.5 KNN-based Recommender using sklearn

# %%
# Fit KNN model on the CSR matrix
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
knn_model.fit(csr_data)

print("✅ KNN Model fitted successfully!")

# %%
def recommend_movies_knn(movie_name, model, data_matrix, movie_index, n_recommendations=5):
    """
    Recommend movies using KNN algorithm with Cosine Similarity.
    
    Parameters:
    -----------
    movie_name : str - Name of the movie
    model : NearestNeighbors - Fitted KNN model
    data_matrix : csr_matrix - Sparse matrix of movie-user ratings
    movie_index : Index - Movie title index
    n_recommendations : int - Number of recommendations
    
    Returns:
    --------
    DataFrame with similar movies and distance scores
    """
    # Find the index of the movie
    idx = list(movie_index).index(movie_name)
    
    # Get K nearest neighbors
    distances, indices = model.kneighbors(data_matrix[idx], n_neighbors=n_recommendations + 1)
    
    # Create results DataFrame (skip the first one as it's the movie itself)
    results = []
    for i in range(1, len(indices[0])):
        results.append({
            'Movie': movie_index[indices[0][i]],
            'Distance': distances[0][i],
            'Similarity': 1 - distances[0][i]
        })
    
    return pd.DataFrame(results)

# %%
# Recommend movies similar to 'Liar Liar (1997)' using KNN
print("=" * 60)
print("🎬 Top 5 Movies Similar to 'Liar Liar (1997)'")
print("   (Using KNN with Cosine Similarity)")
print("=" * 60)

knn_recs = recommend_movies_knn('Liar Liar (1997)', knn_model, csr_data, movie_user_filled.index)
print(knn_recs.to_string(index=False))

# %%
# More examples
print("\n" + "=" * 60)
print("🎬 Top 5 Movies Similar to 'Toy Story (1995)'")
print("   (Using KNN with Cosine Similarity)")
print("=" * 60)

knn_recs_ts = recommend_movies_knn('Toy Story (1995)', knn_model, csr_data, movie_user_filled.index)
print(knn_recs_ts.to_string(index=False))

print("\n" + "=" * 60)
print("🎬 Top 5 Movies Similar to 'Godfather, The (1972)'")
print("   (Using KNN with Cosine Similarity)")
print("=" * 60)

knn_recs_gf = recommend_movies_knn('Godfather, The (1972)', knn_model, csr_data, movie_user_filled.index)
print(knn_recs_gf.to_string(index=False))

# %% [markdown]
# <a id='9'></a>
# ## 9. 🧮 Recommender System — Matrix Factorization
# 
# Matrix Factorization decomposes the user-item interaction matrix into lower-dimensional latent factor matrices.
# We use the **cmfrec** library for this purpose.

# %% [markdown]
# ### 9.1 Prepare Data for Matrix Factorization

# %%
# Prepare ratings data for Surprise SVD
ratings_mf = data[['UserID', 'MovieID', 'Rating']].copy()

# Create Surprise dataset
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(ratings_mf, reader)

print(f"📊 Total ratings: {len(ratings_mf)}")
print(f"👤 Unique users: {ratings_mf['UserID'].nunique()}")
print(f"🎬 Unique movies: {ratings_mf['MovieID'].nunique()}")

# %% [markdown]
# ### 9.2 Train-Test Split

# %%
# Train-Test split (80-20)
trainset, testset = surprise_train_test_split(surprise_data, test_size=0.2, random_state=42)
print(f"📊 Training set size: {trainset.n_ratings}")
print(f"📊 Test set size: {len(testset)}")

# %% [markdown]
# ### 9.3 Build Matrix Factorization Model (d=4)
# 
# **R ≈ P × Qᵀ**  
# Where:  
# - P = User Embeddings  
# - Q = Movie Embeddings  
# - d = 4 latent factors

# %%
# Build SVD model with n_factors=4 (d=4 latent factors)
model_mf = SVD(n_factors=4, random_state=42)

# Fit on training data
model_mf.fit(trainset)

print("✅ Matrix Factorization (SVD) model trained with d=4 latent factors!")

# %% [markdown]
# ### 9.4 Evaluate Model — RMSE & MAPE

# %%
# Predict on test data
predictions_mf = model_mf.test(testset)

# Calculate RMSE using Surprise built-in
rmse = accuracy.rmse(predictions_mf, verbose=False)

# Calculate MAPE manually
actuals = np.array([pred.r_ui for pred in predictions_mf])
preds = np.array([pred.est for pred in predictions_mf])
mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

print("=" * 50)
print("📊 Matrix Factorization Model Evaluation")
print("=" * 50)
print(f"   🎯 RMSE: {rmse:.4f}")
print(f"   📏 MAPE: {mape:.2f}%")
print("=" * 50)

# %%
# Visualize prediction vs actual
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
axes[0].scatter(actuals, preds, alpha=0.1, s=10, color='#3498db')
axes[0].plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Rating', fontsize=12)
axes[0].set_ylabel('Predicted Rating', fontsize=12)
axes[0].set_title('🎯 Actual vs Predicted Ratings', fontsize=14, fontweight='bold')
axes[0].legend()

# Error distribution
errors = actuals - preds
axes[1].hist(errors, bins=50, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Prediction Error', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('📊 Distribution of Prediction Errors', fontsize=14, fontweight='bold')
axes[1].axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
axes[1].legend()

plt.tight_layout()
plt.savefig('images/mf_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 9.5 Embeddings — Item-Item & User-User Similarity

# %%
# Extract embeddings from the SVD model (d=4)
# model_mf.qi = Item factors, model_mf.pu = User factors
item_embeddings = model_mf.qi  # Item latent factors
user_embeddings = model_mf.pu  # User latent factors

print(f"📊 Item Embeddings shape: {item_embeddings.shape}")
print(f"📊 User Embeddings shape: {user_embeddings.shape}")

# Create mappings for item inner IDs
item_inner_ids = {trainset.to_raw_iid(i): i for i in trainset.all_items()}
user_inner_ids = {trainset.to_raw_uid(u): u for u in trainset.all_users()}

# %%
# Re-design Item-Item Similarity using MF Embeddings
item_sim_mf = cosine_similarity(item_embeddings)
item_sim_mf_df = pd.DataFrame(item_sim_mf)

print("📊 Item-Item Similarity Matrix (using MF embeddings d=4):")
print(f"   Shape: {item_sim_mf_df.shape}")
item_sim_mf_df.iloc[:5, :5]

# %%
def recommend_movies_mf_embeddings(movie_id, item_embeddings, item_inner_ids, movies_df, n=5):
    """
    Recommend movies using MF embeddings-based item similarity.
    """
    if movie_id not in item_inner_ids:
        return "Movie not found in the training set."
    
    idx = item_inner_ids[movie_id]
    
    # Calculate cosine similarity with all items
    sim_scores = cosine_similarity([item_embeddings[idx]], item_embeddings)[0]
    
    # Get top N similar items (excluding itself)
    similar_indices = np.argsort(sim_scores)[::-1][1:n+1]
    
    # Map inner IDs back to raw IDs
    reverse_inner = {v: k for k, v in item_inner_ids.items()}
    
    results = []
    for i in similar_indices:
        if i in reverse_inner:
            item_id = reverse_inner[i]
            title = movies_df[movies_df['MovieID'] == item_id]['Title'].values
            if len(title) > 0:
                results.append({
                    'MovieID': item_id,
                    'Title': title[0],
                    'Similarity': sim_scores[i]
                })
    
    return pd.DataFrame(results)

# %%
# Get MovieID for 'Toy Story (1995)'
toy_story_id = movies[movies['Title'] == 'Toy Story (1995)']['MovieID'].values[0]

print("=" * 60)
print("🎬 Top 5 Movies Similar to 'Toy Story (1995)'")
print("   (Using MF Embeddings — d=4)")
print("=" * 60)
mf_recs = recommend_movies_mf_embeddings(toy_story_id, item_embeddings, item_inner_ids, movies)
print(mf_recs.to_string(index=False))

# %%
# User-User Similarity using MF Embeddings
user_sim_mf = cosine_similarity(user_embeddings)
user_sim_mf_df = pd.DataFrame(user_sim_mf)

print("\n📊 User-User Similarity Matrix (using MF embeddings d=4):")
print(f"   Shape: {user_sim_mf_df.shape}")
user_sim_mf_df.iloc[:5, :5]

# %% [markdown]
# ### 9.6 Bonus — 2D Embeddings Visualization

# %%
# Train a new model with d=2 for visualization
model_mf_2d = SVD(n_factors=2, random_state=42)
model_mf_2d.fit(trainset)

item_emb_2d = model_mf_2d.qi  # Item embeddings (d=2)
user_emb_2d = model_mf_2d.pu  # User embeddings (d=2)

print(f"📊 2D Item Embeddings: {item_emb_2d.shape}")
print(f"📊 2D User Embeddings: {user_emb_2d.shape}")

# %%
# Visualize Item Embeddings (2D)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Item embeddings
axes[0].scatter(item_emb_2d[:, 0], item_emb_2d[:, 1], alpha=0.3, s=15, c='#3498db')
axes[0].set_xlabel('Dimension 1', fontsize=12)
axes[0].set_ylabel('Dimension 2', fontsize=12)
axes[0].set_title('🎬 Movie Embeddings (d=2)', fontsize=14, fontweight='bold')

# Annotate some popular movies
popular_movie_ids = movie_stats.head(10)['Title'].values
for title in popular_movie_ids:
    mid = movies[movies['Title'] == title]['MovieID'].values
    if len(mid) > 0 and mid[0] in item_inner_ids:
        idx = item_inner_ids[mid[0]]
        if idx < len(item_emb_2d):
            axes[0].annotate(title[:20], (item_emb_2d[idx, 0], item_emb_2d[idx, 1]),
                           fontsize=7, alpha=0.8, color='red',
                           arrowprops=dict(arrowstyle='->', alpha=0.5, color='red'))

# User embeddings (sample 1000 users)
sample_idx = np.random.choice(len(user_emb_2d), min(1000, len(user_emb_2d)), replace=False)
axes[1].scatter(user_emb_2d[sample_idx, 0], user_emb_2d[sample_idx, 1], 
                alpha=0.3, s=15, c='#e74c3c')
axes[1].set_xlabel('Dimension 1', fontsize=12)
axes[1].set_ylabel('Dimension 2', fontsize=12)
axes[1].set_title('👤 User Embeddings (d=2, sample=1000)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('images/embeddings_2d.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **📝 Analysis of 2D Embeddings:**
# 
# - **Movie Embeddings:** Movies cluster based on latent preferences. Popular blockbusters (e.g., Star Wars, Toy Story) tend to be near the center, while niche movies occupy the periphery. Nearby movies in this space share similar user rating patterns — validating that MF captures meaningful latent features.
# 
# - **User Embeddings:** Users form a continuous distribution rather than distinct clusters, suggesting diverse tastes. Users near each other have similar movie preferences. The spread indicates the dataset captures a wide range of viewing behaviors.
# 
# - **Comparison with raw Cosine Similarity:** The MF-based embeddings provide a more compact and denoised representation compared to raw ratings vectors. While cosine similarity on raw data suffers from sparsity, MF embeddings capture latent patterns more effectively.

# %% [markdown]
# <a id='10'></a>
# ## 10. 👤 Recommender System — User-Based Approach (Bonus)
# 
# In this approach, we simulate a **new user**, find similar existing users, and recommend movies based on their preferences.

# %%
# Step 1: Simulate a new user's ratings
new_user_ratings = {
    'Toy Story (1995)': 5,
    'Jurassic Park (1993)': 4,
    'Star Wars: Episode IV - A New Hope (1977)': 5,
    'Forrest Gump (1994)': 4,
    'Shawshank Redemption, The (1994)': 5,
    'Matrix, The (1999)': 5,
    'Silence of the Lambs, The (1991)': 3,
    'Titanic (1997)': 3,
    'Back to the Future (1985)': 4,
    'Terminator 2: Judgment Day (1991)': 4
}

new_user_df = pd.DataFrame(list(new_user_ratings.items()), columns=['Title', 'Rating'])
print("🆕 New User's Ratings:")
print(new_user_df.to_string(index=False))

# %%
# Step 2: Find users who watched the same movies
new_user_movies = set(new_user_ratings.keys())

# Get users who rated the same movies
users_who_watched = data[data['Title'].isin(new_user_movies)].groupby('UserID')['Title'].apply(set)

# Count common movies
common_counts = users_who_watched.apply(lambda x: len(x.intersection(new_user_movies)))
common_counts = common_counts.sort_values(ascending=False)

print(f"\n📊 Users who watched at least 1 common movie: {len(common_counts)}")
print(f"🏆 Max common movies with any user: {common_counts.max()}")

# %%
# Step 3: Take top 100 users with most common movies
top_100_users = common_counts.head(100).index.tolist()
print(f"📊 Selected top {len(top_100_users)} users with most common movies")

# Step 4: Calculate Pearson Correlation similarity for each user
similarity_scores = {}

for user_id in top_100_users:
    # Get the user's ratings for common movies
    user_ratings = data[data['UserID'] == user_id].set_index('Title')['Rating']
    
    # Get common movies between new user and this user
    common = set(new_user_ratings.keys()).intersection(set(user_ratings.index))
    
    if len(common) >= 3:  # Need at least 3 common movies for meaningful correlation
        new_user_common = [new_user_ratings[m] for m in common]
        old_user_common = [user_ratings[m] for m in common]
        
        if np.std(new_user_common) > 0 and np.std(old_user_common) > 0:
            corr, _ = stats.pearsonr(new_user_common, old_user_common)
            similarity_scores[user_id] = corr

# Sort by similarity
sim_df = pd.DataFrame(list(similarity_scores.items()), columns=['UserID', 'Similarity'])
sim_df = sim_df.sort_values('Similarity', ascending=False)
print(f"\n📊 Users with valid similarity scores: {len(sim_df)}")
sim_df.head(10)

# %%
# Step 5: Get top 10 most similar users
top_10_users = sim_df.head(10)
print("🏆 Top 10 Most Similar Users:")
print(top_10_users.to_string(index=False))

# Step 6: Get all movies rated by top 10 users and calculate weighted ratings
top_user_ids = top_10_users['UserID'].tolist()
top_user_sims = dict(zip(top_10_users['UserID'], top_10_users['Similarity']))

# Get all ratings from top 10 similar users
similar_users_ratings = data[data['UserID'].isin(top_user_ids)][['UserID', 'Title', 'Rating']]

# Add similarity weight
similar_users_ratings['Similarity'] = similar_users_ratings['UserID'].map(top_user_sims)
similar_users_ratings['WeightedRating'] = similar_users_ratings['Rating'] * similar_users_ratings['Similarity']

# Aggregate by movie
weighted_recs = similar_users_ratings.groupby('Title').agg(
    TotalWeightedRating=('WeightedRating', 'sum'),
    TotalSimilarity=('Similarity', 'sum'),
    NumRaters=('UserID', 'nunique')
).reset_index()

weighted_recs['AvgWeightedScore'] = weighted_recs['TotalWeightedRating'] / weighted_recs['TotalSimilarity']

# Remove movies already rated by new user
weighted_recs = weighted_recs[~weighted_recs['Title'].isin(new_user_ratings.keys())]

# Get top 10 recommendations
final_recs = weighted_recs.sort_values('AvgWeightedScore', ascending=False).head(10)

# %%
print("\n" + "=" * 60)
print("🎬 Top 10 Movie Recommendations for the New User")
print("   (Using User-Based Collaborative Filtering)")
print("=" * 60)
print(final_recs[['Title', 'AvgWeightedScore', 'NumRaters']].to_string(index=False))

# %% [markdown]
# <a id='11'></a>
# ## 11. 📝 Questionnaire Answers

# %% [markdown]
# ### Q1: Users of which age group have watched and rated the most number of movies?

# %%
age_group_ratings = data.groupby('AgeGroup')['Rating'].count().sort_values(ascending=False)
print(f"🏆 Answer: {age_group_ratings.index[0]}")
print(f"\nAll age groups ranked by number of ratings:")
for age, count in age_group_ratings.items():
    print(f"   {age}: {count:,}")

# %% [markdown]
# ### Q2: Users belonging to which profession have watched and rated the most movies?

# %%
occ_most_ratings = data.groupby('OccupationLabel')['Rating'].count().sort_values(ascending=False)
print(f"🏆 Answer: {occ_most_ratings.index[0]}")
print(f"\nTop 5 professions by number of ratings:")
for occ, count in occ_most_ratings.head(5).items():
    print(f"   {occ}: {count:,}")

# %% [markdown]
# ### Q3: Most of the users in our dataset who've rated the movies are Male. (T/F)

# %%
male_count = data[data['Gender'] == 'M']['UserID'].nunique()
female_count = data[data['Gender'] == 'F']['UserID'].nunique()
print(f"👨 Male users: {male_count}")
print(f"👩 Female users: {female_count}")
print(f"\n🏆 Answer: TRUE — {male_count / (male_count + female_count) * 100:.1f}% of users are Male")

# %% [markdown]
# ### Q4: Most of the movies present in our dataset were released in which decade?

# %%
decade_movie_counts = data.groupby('Decade')['MovieID'].nunique().sort_values(ascending=False)
print(f"🏆 Answer: b. 90s ({decade_movie_counts.get(1990, 0)} unique movies)")
print(f"\nMovies per decade:")
for decade, count in decade_movie_counts.items():
    print(f"   {decade}s: {count}")

# %% [markdown]
# ### Q5: The movie with maximum no. of ratings is ___.

# %%
print(f"🏆 Answer: {movie_stats.iloc[0]['Title']}")
print(f"   Number of Ratings: {int(movie_stats.iloc[0]['NumRatings'])}")

# %% [markdown]
# ### Q6: Name the top 3 movies similar to 'Liar Liar' on the item-based approach.

# %%
liar_liar_top3 = recommend_movies_pearson('Liar Liar (1997)', movie_user_pivot, n_recommendations=3)
print("🏆 Answer: Top 3 movies similar to 'Liar Liar (1997)' (Pearson Correlation):")
for i, (_, row) in enumerate(liar_liar_top3.iterrows(), 1):
    print(f"   {i}. {row['Title']} (Correlation: {row['Correlation']:.4f})")

# %% [markdown]
# ### Q7: On the basis of approach, Collaborative Filtering methods can be classified into ___-based and ___-based.

# %%
print("🏆 Answer: Item-based and User-based")
print("\n   • Item-based: Finds items similar to what a user has already liked")
print("   • User-based: Finds users similar to the target user and recommends what they liked")

# %% [markdown]
# ### Q8: Pearson Correlation ranges between ___ to ___ whereas, Cosine Similarity belongs to the interval between ___ to ___.

# %%
print("🏆 Answer:")
print("   • Pearson Correlation: -1 to +1")
print("   • Cosine Similarity: 0 to 1 (for non-negative data like ratings)")
print("\n   Note: In general, Cosine Similarity ranges from -1 to +1,")
print("   but since movie ratings are non-negative, it ranges from 0 to 1.")

# %% [markdown]
# ### Q9: Mention the RMSE and MAPE that you got while evaluating the Matrix Factorization model.

# %%
print(f"🏆 Answer:")
print(f"   • RMSE: {rmse:.4f}")
print(f"   • MAPE: {mape:.2f}%")

# %% [markdown]
# ### Q10: Give the sparse 'row' matrix representation for the following dense matrix -
# ```
# [[1 0]
#  [3 7]]
# ```

# %%
from scipy.sparse import csr_matrix as csr

dense_matrix = np.array([[1, 0], [3, 7]])
sparse_matrix = csr(dense_matrix)

print("🏆 Answer: CSR (Compressed Sparse Row) representation:")
print(f"\n   Dense Matrix:")
print(f"   [[1 0]")
print(f"    [3 7]]")
print(f"\n   Sparse Row Matrix Representation:")
print(f"   Row 0: (0, 0) → 1")
print(f"   Row 1: (1, 0) → 3, (1, 1) → 7")
print(f"\n   CSR Components:")
print(f"   data    = {sparse_matrix.data}")
print(f"   indices = {sparse_matrix.indices}")
print(f"   indptr  = {sparse_matrix.indptr}")

# %% [markdown]
# <a id='12'></a>
# ## 12. ✅ Conclusion
# 
# In this project, we built a comprehensive **Movie Recommender System** using three collaborative filtering approaches:
# 
# ### Key Findings:
# 
# | Approach | Method | Key Result |
# |----------|--------|------------|
# | **Item-Based (Pearson)** | Pearson Correlation | Successfully identifies movies with similar rating patterns |
# | **Item-Based (Cosine/KNN)** | Cosine Similarity + KNN | Efficient similarity search using sparse matrices |
# | **Matrix Factorization** | cmfrec (d=4) | Captures latent features; RMSE and MAPE demonstrate reasonable accuracy |
# | **User-Based (Bonus)** | Pearson Correlation | Recommends movies based on similar users' preferences |
# 
# ### Insights:
# - **Age group 25-34** is the most active demographic in rating movies
# - **Male users** dominate the dataset in terms of ratings
# - **The 1990s** saw the most movie releases in the dataset
# - Matrix Factorization with embeddings provides a compact representation that captures user-item interactions effectively
# - 2D embedding visualizations reveal natural clustering of movies and users based on latent preferences
# 
# ---
# 
# **Author:** Pritam Palit  
# **GitHub:** [@PritamPalit-official](https://github.com/PritamPalit-official)  
# **LinkedIn:** [Pritam Palit](https://www.linkedin.com/in/pritam-palit-77b2071b4/)

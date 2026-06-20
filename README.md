<div align="center">

# 🎬 Movie Recommender System

### Personalized Movie Recommendations using Collaborative Filtering

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Surprise](https://img.shields.io/badge/Surprise-SVD-FF6F61?style=for-the-badge)](https://surpriselib.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Building a recommendation engine that predicts user preferences using collaborative filtering techniques on the MovieLens 1M dataset.*

---

</div>

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Analysis & Methodology](#-analysis--methodology)
- [Recommender Approaches](#-recommender-approaches)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Key Results](#-key-results)
- [Questionnaire Answers](#-questionnaire-answers)
- [Business Recommendations](#-business-recommendations)
- [Tools & Technologies](#-tools--technologies)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Problem Statement

**Objective:** Create a Recommender System to show personalized movie recommendations based on ratings given by a user and other users similar to them, in order to **improve user experience and increase engagement**.

**Business Goal:** Increase user watch time by recommending movies that users are most likely to enjoy, using multiple collaborative filtering techniques.

---

## 📊 Dataset Description

**Source:** [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/) by GroupLens Research

| File | Records | Description |
|------|---------|-------------|
| `ratings.dat` | 1,000,209 | UserID :: MovieID :: Rating (1-5) :: Timestamp |
| `users.dat` | 6,040 | UserID :: Gender :: Age :: Occupation :: Zip-code |
| `movies.dat` | 3,883 | MovieID :: Title :: Genres |

| Metric | Value |
|--------|-------|
| **Total Ratings** | 1,000,209 |
| **Unique Users** | 6,040 |
| **Unique Movies** | 3,883 |
| **Rating Scale** | 1 ⭐ to 5 ⭐ |
| **Merged Dataset** | 1,000,209 rows × 10 columns |

---

## 🔬 Analysis & Methodology

### Exploratory Data Analysis (EDA)

| Analysis | Key Finding |
|----------|------------|
| 👤 **Gender Distribution** | 75.36% Male, 24.64% Female |
| 📅 **Most Active Age Group** | 25-34 (395,556 ratings) |
| 🏢 **Most Active Occupation** | College/Grad Student (131,032 ratings) |
| 🎬 **Most Rated Movie** | American Beauty (1999) — 3,428 ratings |
| 📆 **Most Movies Released In** | 1990s decade (2,283 movies) |

### Feature Engineering

- **Release Year** — Extracted from movie title using regex
- **Decade** — Derived from release year for trend analysis
- **Age Group Labels** — Mapped age codes to readable categories
- **Occupation Labels** — Mapped occupation codes to descriptions
- **Rating Timestamp** — Converted Unix timestamp to datetime

---

## 🧠 Recommender Approaches

### 1️⃣ Item-Based — Pearson Correlation

| Aspect | Details |
|--------|---------|
| **Method** | Pearson Correlation between movie rating vectors |
| **Range** | -1 to +1 |
| **Input** | Movie title from user |
| **Output** | Top 5 similar movies |

**Example — Movies similar to "Liar Liar (1997)":**

| Rank | Movie | Correlation |
|------|-------|-------------|
| 1 | Life (1999) | 0.576 |
| 2 | East-West (1999) | 0.562 |
| 3 | Oliver & Company (1988) | 0.551 |

---

### 2️⃣ Item-Based — Cosine Similarity + KNN

| Aspect | Details |
|--------|---------|
| **Method** | Cosine Similarity with K-Nearest Neighbors |
| **Range** | 0 to 1 |
| **Algorithm** | sklearn NearestNeighbors (brute force) |
| **Matrix** | CSR (Compressed Sparse Row) format |

**Features:**
- ✅ Item-Item Similarity Matrix
- ✅ User-User Similarity Matrix
- ✅ CSR Sparse Matrix for efficiency
- ✅ KNN-based top-5 recommendations

---

### 3️⃣ Matrix Factorization — SVD

| Aspect | Details |
|--------|---------|
| **Method** | Singular Value Decomposition (SVD) |
| **Library** | Surprise (scikit-surprise) |
| **Latent Factors** | d = 4 |
| **Formula** | R ≈ P × Qᵀ (User × Item embeddings) |

**Evaluation Results:**

| Metric | Value |
|--------|-------|
| **RMSE** | ~0.87 |
| **MAPE** | ~20% |

**Embeddings:**
- 📊 d=4 embeddings for item-item & user-user similarity
- 📊 d=2 embeddings for visualization
- 🎨 Movies of similar genres cluster together (Action near Action, Romance near Romance)

---

### 4️⃣ User-Based Collaborative Filtering (Bonus)

| Step | Description |
|------|-------------|
| 1 | New user rates a few movies |
| 2 | Find users who rated the same movies |
| 3 | Calculate Pearson Similarity with top 100 users |
| 4 | Select top 10 most similar users |
| 5 | Compute weighted ratings (Rating × Similarity) |
| 6 | Recommend top 10 movies with highest weighted average |

---

## 📈 Key Results

| Approach | Best For | Scalability |
|----------|----------|-------------|
| **Pearson Correlation** | Quick, interpretable recommendations | Medium |
| **Cosine Similarity + KNN** | Efficient similarity search | High (with CSR) |
| **Matrix Factorization (SVD)** | Latent feature discovery, best accuracy | High |
| **User-Based CF** | Cold-start scenarios with new users | Low-Medium |

---

## 📝 Questionnaire Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Most active age group? | **25-34** (395,556 ratings) |
| 2 | Most active profession? | **College/Grad Student** (131,032 ratings) |
| 3 | Most users are Male? (T/F) | **True** (75.36% Male) |
| 4 | Most movies released in which decade? | **b. 90s** (2,283 movies) |
| 5 | Movie with max ratings? | **American Beauty (1999)** (3,428 ratings) |
| 6 | Top 3 similar to Liar Liar? | 1. Life (1999), 2. East-West (1999), 3. Oliver & Company (1988) |
| 7 | CF classification? | **User-based** and **Item-based** |
| 8 | Pearson & Cosine ranges? | Pearson: **-1 to +1**, Cosine: **0 to 1** |
| 9 | RMSE & MAPE? | RMSE ≈ **0.87**, MAPE ≈ **20%** |
| 10 | Sparse row matrix for [[1,0],[3,7]]? | data=[1,3,7], indices=[0,0,1], indptr=[0,1,3] |

---

## 💼 Business Recommendations

| # | Recommendation | Rationale |
|---|---------------|-----------|
| 1 | **Use Item-Based CF in Production** | Scales well with growing movie catalog |
| 2 | **Build a Hybrid System** | Combine Pearson + Cosine + MF for better accuracy |
| 3 | **Target 25-34 Age Group** | Largest and most active user demographic |
| 4 | **Improve Female User Engagement** | Only ~25% ratings from females — opportunity gap |
| 5 | **Use Popular Movies for Cold Start** | American Beauty, Star Wars, Jurassic Park as defaults |
| 6 | **Deploy Matrix Factorization** | Best prediction accuracy via latent-feature learning |

---

## 💻 Streamlit Dashboard

A professional, production-ready interactive web application built with Streamlit and styled with a custom SaaS-like design (with native Light Mode default and a Dark Mode toggle).

### 🖥️ Dashboard features:
* **Overview page**: Displays key platform metrics (total ratings, users, movies, average rating) and user demographic highlights.
* **Exploratory Data Analysis (EDA) page**: Fully interactive Plotly charts showcasing rating distributions, genre performance, user behaviors, decade releases, and movie correlations.
* **Recommendation Engine page**: Live inputs allowing search-by-title and selection of recommendation algorithms (Pearson Correlation, Cosine Similarity + KNN, or Hybrid blend) to return recommended movie cards.
* **User-Based page**: Prompting new users to rate popular movies to instantly find overlapping neighbor profiles and generate collaborative recommendation scores.
* **SVD Analytics page**: Visualizations of latent-factor embeddings (d=2 PCA space, d=4 latent variables) and item profile heatmaps along with regression metrics (RMSE, MAPE).
* **Business Insights page**: Segment analysis (AgeGroup × Gender), cold start playbooks, and strategic SaaS-like business advice cards.

---

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | scikit-learn (KNN, Cosine Similarity) |
| **Matrix Factorization** | Surprise (SVD) |
| **Statistical Analysis** | SciPy |
| **Web Dashboard** | Streamlit |
| **Notebook** | Jupyter Notebook |

---

## 📁 Project Structure

```
Movie-Recommender-System/
│
├── 📓 Movie_Recommender_System.ipynb   # Main analysis notebook
├── 📄 Movie_Recommender_System.py      # Python script version
├── 💻 streamlit_app.py                 # Streamlit entry point
├── 📋 requirements.txt                 # Dependencies
├── 📝 README.md                        # This file
├── 🚫 .gitignore                       # Git ignore rules
│
├── 📂 .streamlit/
│   └── ⚙️ config.toml                 # Streamlit theme & layout configs
│
├── 📂 dashboard/                       # Dashboard page modules & helpers
│   ├── 📁 page_overview.py            # Overview charts & metrics
│   ├── 📁 page_eda.py                 # Multi-tab EDA Plotly dashboards
│   ├── 📁 page_recommendations.py     # Search & recommend live engines
│   ├── 📁 page_user_based.py          # User rating onboarding
│   ├── 📁 page_svd.py                 # Latent space & SVD heatmap
│   ├── 📁 page_business.py            # Strategic cards & segments
│   ├── 🛠️ data_loader.py              # Cached dataset pre-processing
│   ├── 🛠️ models.py                   # Cached Surprise models
│   ├── 🛠️ recommenders.py             # Collaborative filtering recommenders
│   └── 🎨 styles.py                   # SaaS style cards & CSS themes
│
├── 📂 data/
│   ├── movies.dat                      # Movie info (3,883 movies)
│   ├── users.dat                       # User demographics (6,040 users)
│   └── ratings.dat                     # Ratings (1,000,209 ratings)
│
└── 📂 images/
    ├── rating_distribution.png         # Rating distribution charts
    ├── gender_distribution.png         # Gender analysis
    ├── age_distribution.png            # Age group analysis
    ├── occupation_distribution.png     # Occupation analysis
    ├── movies_per_decade.png           # Decade-wise movie releases
    ├── genre_distribution.png          # Genre distribution
    ├── rating_stats.png                # Rating statistics
    ├── avg_vs_num_ratings.png          # Avg rating vs count
    ├── mf_evaluation.png               # MF model evaluation
    └── embeddings_2d.png               # 2D embedding visualization
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/PritamPalit-official/Movie-Recommender-System.git
cd Movie-Recommender-System

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Web Dashboard

```bash
python -m streamlit run streamlit_app.py
```

### Run the Notebook

```bash
jupyter notebook Movie_Recommender_System.ipynb
```

---

## 👤 Author

<div align="center">

**Pritam Palit**

*Electronics and Communication Engineering Graduate*

*Focus Areas: Data Analytics, Machine Learning, Business Intelligence*

[![GitHub](https://img.shields.io/badge/GitHub-PritamPalit--official-181717?style=for-the-badge&logo=github)](https://github.com/PritamPalit-official)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Pritam_Palit-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/pritam-palit-77b2071b4/)

</div>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

</div>

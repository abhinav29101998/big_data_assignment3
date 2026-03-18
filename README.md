#  CSL7110 Assignment 3 — Recommender Systems
### Content-Based Filtering, Collaborative Filtering, Matrix Factorization, Hybrid Models, RL & Explainability

---

##  Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Dependencies & Installation](#dependencies--installation)
5. [How to Run](#how-to-run)
6. [Notebook Walkthrough](#notebook-walkthrough)
7. [Architecture Diagrams](#architecture-diagrams)
8. [Results Summary](#results-summary)
9. [Troubleshooting](#troubleshooting)

---

##  Project Overview

This notebook implements a **full-stack recommender system pipeline** on the [MovieLens Small dataset](https://grouplens.org/datasets/movielens/latest/), covering six major paradigms:

| Part | Topic | Tasks | Marks |
|------|-------|-------|-------|
| 1 | Content-Based Filtering | 1–2 | 20 |
| 2 | Collaborative Filtering | 3–4 | 20 |
| 3 | Matrix Factorization | 5–6 | 20 |
| 4 | Hybrid Recommendation | 7 | 10 |
| 5 | Learning-Based (NN + RL) | 8–9 | 40 |
| 6 | Explainability | 10–13 | 10 |
| | **Total** | **13 tasks** | **120** |

---

##  Repository Structure

```
project/
│
├── CSL7110_Assignment3_Recommender_Systems.ipynb   ← Main notebook
├── README.md                                        ← This file
│
└── ml-latest-small/                                 ← Dataset folder (place here)
    ├── movies.csv
    ├── ratings.csv
    ├── tags.csv
    ├── links.csv
    └── README.txt
```

>  **Important:** The `ml-latest-small/` folder must be in the **same directory** as the notebook.

---

##  Dataset

**MovieLens Small** — a standard benchmark for recommender system research.

| File | Description | Size |
|------|-------------|------|
| `movies.csv` | 9,742 movies with title and genres | ~500 KB |
| `ratings.csv` | 100,836 ratings by 610 users (scale: 0.5–5.0) | ~2.4 MB |
| `tags.csv` | 3,683 user-applied tags | ~120 KB |
| `links.csv` | IMDb / TMDb cross-reference IDs | ~200 KB |

**Download:** [https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)
Direct link: `ml-latest-small.zip` → extract into project folder.

```
Dataset Statistics
──────────────────────────────────────
  Users         :   610
  Movies        : 9,742
  Ratings       : 100,836
  Rating Range  : 0.5 – 5.0
  Matrix Density: ~1.7% (very sparse)
──────────────────────────────────────
```

---

##  Dependencies & Installation

### Core Requirements (Standard Python scientific stack)

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.21 | Array math, SVD |
| `pandas` | ≥ 1.3 | Data loading and manipulation |
| `scikit-learn` | ≥ 0.24 | TF-IDF, MLP, Ridge, metrics |
| `scipy` | ≥ 1.7 | Sparse SVD (`svds`) |
| `matplotlib` | ≥ 3.4 | All plots and visualizations |
| `seaborn` | ≥ 0.11 | Heatmaps and styled charts |

### Optional (notebook interface)

```bash
pip install jupyterlab notebook
```

###  Notes on Unavailable Libraries

The following libraries are **implemented from scratch** inside the notebook, so you do **not** need to install them:

| Library | What We Implemented Instead |
|---------|----------------------------|
| `scikit-surprise` | Full Funk SVD with SGD, biases, regularization |
| `tensorflow` / `torch` | Used `sklearn.neural_network.MLPRegressor` |
| `shap` | Permutation-based feature importance |
| `lime` | Local perturbation + Ridge surrogate model |

---

##  How to Run

### Step 1 — Clone / Download Files

Place both files in the same directory:
- `CSL7110_Assignment3_Recommender_Systems.ipynb`
- `ml-latest-small/` folder (extracted from zip)

### Step 2 — Install Dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn jupyterlab
```

### Step 3 — Launch Jupyter

```bash
# Option A: JupyterLab (recommended)
jupyter lab

# Option B: Classic Jupyter Notebook
jupyter notebook
```

### Step 4 — Open and Run

1. Navigate to `CSL7110_Assignment3_Recommender_Systems.ipynb`
2. Click **Kernel → Restart & Run All**
3. Full execution takes approximately **10–20 minutes** (dominated by Item-CF and RL training)

### Step 5 — Alternative: Run via Script

```bash
jupyter nbconvert --to notebook --execute \
    CSL7110_Assignment3_Recommender_Systems.ipynb \
    --output executed_output.ipynb \
    --ExecutePreprocessor.timeout=1200
```

---

##  Notebook Walkthrough

###  Part 1 — Content-Based Filtering

#### Task 1: TF-IDF Recommender

```
Movie genres are treated as "documents"
       ↓
TfidfVectorizer (sklearn) builds a (9742 × 21) matrix
       ↓
Cosine similarity computed between all movie pairs
       ↓
Given a query movie → return top-N most similar by genre
```

**Example output:**
```
Input: Toy Story (1995)
──────────────────────────────────────────────────
Recommended Movie                  Cosine Similarity
Antz (1998)                               1.000
Toy Story 2 (1999)                        1.000
Monsters, Inc. (2001)                     1.000
Emperor's New Groove, The (2000)          1.000
Adventures of Rocky and Bullwinkle (2000) 1.000
```

#### Task 2: User-Profile-Based Recommender

The user profile is a **rating-weighted average** of TF-IDF vectors:

```
         Σ (rating_m × tfidf_vector_m)
P_user = ─────────────────────────────
                 Σ rating_m
```

- High-rated movies contribute **more** to the user's taste vector
- Profile is then compared to all unseen movies via cosine similarity
- Evaluated with **Precision@10** and **Recall@10**

---

###  Part 2 — Collaborative Filtering

#### Task 3: User-Based CF

```
User-Movie Rating Matrix (610 × 9724, 98.3% sparse)
           ↓
Mean-center ratings per user (corrects for lenient/harsh raters)
           ↓
Pearson correlation between all user pairs (610 × 610 matrix)
           ↓
For target user: find K most similar users
           ↓
Predict rating = user_mean + weighted average of neighbor deviations
           ↓
Rank unrated movies → recommend top-N
```

**Effect of K on RMSE:**
```
K= 5  → RMSE = 1.0412
K=10  → RMSE = 1.0301
K=20  → RMSE = 1.0243   ← good balance
K=30  → RMSE = 1.0198
K=50  → RMSE = 1.0175
```
>  More neighbors = smoother but slower predictions. Diminishing returns beyond K=30.

#### Task 4: Item-Based CF

```
Transpose matrix → items × users
           ↓
Pearson correlation between all item pairs (9724 × 9724)
           ↓
For target (user, movie): find K most similar items the user HAS rated
           ↓
Predict = weighted average of those item ratings
```

**Why Item-CF is more efficient in production:**
- Item space grows slowly vs. user base (which can be millions)
- Item-item similarity is computed **once offline** and cached
- New user joins → no recomputation needed
- Item-CF RMSE: **0.9841** vs User-CF RMSE: **1.0243** (better accuracy too)

---

###  Part 3 — Matrix Factorization

#### Task 5: NumPy/SciPy SVD

```
Fill missing ratings with per-user mean
           ↓
Mean-center entire matrix globally
           ↓
R ≈ U · Σ · Vᵀ    (scipy.sparse.linalg.svds, k=50 factors)
           ↓
Reconstruct R̂ = U · Σₖ · Vᵀ + global_mean
           ↓
Clip predictions to [0.5, 5.0]
           ↓
Recommend top-N unrated movies per user
```

**Latent factor sensitivity:**
```
k= 10  → RMSE = 0.9891
k= 20  → RMSE = 0.9634
k= 50  → RMSE = 0.9423   ← used in notebook
k=100  → RMSE = 0.9318
k=150  → RMSE = 0.9291
```

#### Task 6: Surprise-Style SVD (SGD / Funk SVD)

Implemented **from scratch** using Stochastic Gradient Descent:

```
r̂_ui = μ + b_u + b_i + p_u · q_i

Where:
  μ    = global mean rating
  b_u  = user bias (generous/harsh rater offset)
  b_i  = item bias (universally liked/disliked offset)
  p_u  = user latent factor vector  (1 × k)
  q_i  = item latent factor vector  (1 × k)
```

Update rules per observed rating:
```
err   = r_ui - r̂_ui
b_u  += lr × (err - reg × b_u)
b_i  += lr × (err - reg × b_i)
p_u  += lr × (err × q_i - reg × p_u)
q_i  += lr × (err × p_u - reg × q_i)
```

**Why SGD SVD beats NumPy SVD:**
- Only trains on **observed** ratings (no imputation noise)
- Bias terms explicitly model user/item tendencies
- RMSE: **0.9077** vs NumPy SVD **0.9423**

---

###  Part 4 — Hybrid Recommendation Model

#### Task 7: Meta-Learning Hybrid

A Ridge Regression **meta-model** learns to blend signals dynamically:

```
Features for each (user, movie) pair:
  [cbf_score, svd_cf_score, movie_avg_rating, user_avg_rating]
           ↓
Ridge Regression meta-model trained on historical ratings
           ↓
Final predicted rating = α·CBF + β·CF + γ·popularity + δ·user_bias
           ↓
Clip to [0.5, 5.0]
```

**Learned coefficients:**
```
CBF score       : 0.0412   (genre match matters, but less than CF)
SVD CF score    : 0.8873   (dominant signal)
Movie avg rating: 0.0621   (popularity correction)
User avg rating : 0.1204   (user bias adjustment)
```

**Cold-start analysis:**
```
Warm users  (>20 ratings): RMSE = 0.8671  ← excellent
Cold-start  (≤5 ratings) : RMSE = 1.1243  ← harder, CBF helps
```

---

###  Part 5 — Learning-Based Recommender Systems

#### Task 8: Neural Network CBF

Two-branch MLP architecture:

```
User Features (19-dim)         Movie Features (21-dim)
  avg rating per genre           one-hot genres
       ↓                         release year
  Dense(64) + ReLU               avg rating
       ↓                              ↓
  User Embedding (32-dim)      Dense(64) + ReLU
                    ↘               ↙
                  Concatenate (64-dim)
                         ↓
                    Dense(32) + ReLU
                         ↓
                    Dense(1) → predicted rating
```

- Loss: MSE | Optimizer: Adam (lr=0.001)
- Early stopping on validation loss to prevent overfitting
- Test RMSE: **0.9712**

#### Task 9: Reinforcement Learning

Three RL strategies implemented:

**ε-Greedy Bandit (ε=0.1):**
```
With probability ε  → explore random movie
With probability 1-ε → exploit highest estimated reward movie
Reward: +1 if rating ≥ 4, -1 if rating < 4, 0 if unrated
```

**UCB Bandit:**
```
Select arm = argmax [ Q(a) + c × √(ln(t) / n(a)) ]
                      ↑ exploit     ↑ explore bonus
Prioritises under-explored movies with high uncertainty
```

**Q-Learning Agent:**
```
State: last reward context (positive/negative/neutral)
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]
  α=0.1 (learning rate), γ=0.9 (discount factor)
```

**Cumulative reward comparison (1000 rounds):**
```
ε-Greedy MAB :  87  (simple, reliable)
Q-Learning   : 119  (state-aware, adaptive)
UCB          : 134  (best exploration-exploitation balance)
```

---

###  Part 6 — Explainability

#### Task 10: Feature-Based (SHAP Approximation)

Permutation importance: measure how much MSE increases when each feature is randomly shuffled.

```
Higher increase in MSE = Feature was more important

Example output (top features):
  movie_avg_rating    : +0.0412  ← most important
  user_Drama          : +0.0387
  movie_Drama         : +0.0341
  user_Comedy         : +0.0298
  movie_year          : +0.0187
```

Human-readable explanation generated:
```
"Movie 'Goodfellas (1990)' was recommended because:
  • Movie genres: Crime, Drama
  • Your top preferences: Drama, Thriller, Crime
  • Genre overlap: Crime, Drama ✓
  → Recommended because you enjoy Crime, Drama films."
```

#### Task 11: Neighborhood-Based (k-NN Explanation)

```
User-CF explanation for Movie 318 (Shawshank Redemption):
Similar User  | Similarity | Their Rating
           73 |     0.8421 |           5.0
          452 |     0.8213 |           5.0
          380 |     0.7994 |           4.5
          567 |     0.7841 |           5.0
           89 |     0.7612 |           4.0
→ Users most similar to you rated this film ~4.8/5.
```

#### Task 12: LIME Explainability (Neural Network)

Local Interpretable Model-agnostic Explanations — approximate the neural network's decision locally with a linear model:

```
1. Take one prediction instance
2. Perturb features with Gaussian noise (200 samples)
3. Weight perturbed samples by distance (Gaussian kernel)
4. Fit local Ridge Regression → get linear coefficients
5. Coefficients = "why did the NN predict this rating?"
```

#### Task 13: Explainability Evaluation

| Method | Clarity | Bias Detection | Cost |
|--------|---------|---------------|------|
| SHAP/Permutation | ★★★★☆ | Excellent (global) | Medium |
| k-NN Neighborhood | ★★★★★ | Good (social echo chamber) | Low |
| LIME | ★★★☆☆ | Good (local anomalies) | High |

---

##  Results Summary

```
╔══════════════════════════════════════════════════════════════╗
║           FINAL PERFORMANCE SUMMARY – ALL METHODS           ║
╠══════════════════════╦══════════╦══════════════╦════════════╣
║ Method               ║   RMSE   ║ Precision@10 ║  Recall@10 ║
╠══════════════════════╬══════════╬══════════════╬════════════╣
║ User-CF (K=20)       ║  1.0243  ║     0.1267   ║   0.0389   ║
║ Item-CF (K=20)       ║  0.9841  ║       —      ║     —      ║
║ SVD (numpy, k=50)    ║  0.9423  ║     0.1540   ║   0.0501   ║
║ Surprise SVD (SGD)   ║  0.9077  ║     0.1612   ║   0.0538   ║
║ Neural Network (MLP) ║  0.9712  ║       —      ║     —      ║
║ Hybrid (Meta-Ridge)  ║  0.8972  ║       —      ║     —      ║
╠══════════════════════╬══════════╬══════════════╬════════════╣
║  Best Overall        ║  Hybrid  ║  0.8972 RMSE ║            ║
╚══════════════════════╩══════════╩══════════════╩════════════╝
```

**Key takeaways:**
- **Hybrid model** achieves the best RMSE by blending CBF + CF signals
- **Surprise-style SVD** is the best single method (avoids imputation noise)
- **Item-CF** beats User-CF in both accuracy and real-world scalability
- **Neural Network** captures non-linear patterns but needs more data to beat CF
- **UCB Bandit** is the best RL strategy for long-term exploration-exploitation balance

---

##  Troubleshooting

###  `FileNotFoundError: ml-latest-small/movies.csv`
The dataset folder must be in the **same directory** as the notebook.
```bash
# Correct layout:
project/
├── notebook.ipynb
└── ml-latest-small/
    ├── movies.csv
    └── ratings.csv
```

###  `ModuleNotFoundError`
Install missing packages:
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

###  Kernel crashes on Item-CF or SVD
These cells are memory-intensive. Try:
- Closing other applications
- Reducing `user_movie_matrix` size by filtering to users with ≥ 20 ratings
- Running on a machine with ≥ 8 GB RAM

###  Slow execution (>30 min)
Speed up by reducing sample sizes:
```python
# In Task 3 — reduce test sample
test_sample = test_ratings.sample(200, random_state=42)  # was 500

# In Task 9 — reduce RL rounds
N_ROUNDS = 300  # was 1000
```

###  `scikit-surprise not found`
This is expected — Surprise SVD is fully implemented from scratch inside the notebook. No installation needed.

---


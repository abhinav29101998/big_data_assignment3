# CSL7110 Assignment 3 — Recommender Systems

Content-Based Filtering, Collaborative Filtering, Matrix Factorization, Hybrid Models, Reinforcement Learning, and Explainability on the MovieLens dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Dependencies and Installation](#dependencies-and-installation)
5. [How to Run](#how-to-run)
6. [Notebook Walkthrough](#notebook-walkthrough)
7. [Results Summary](#results-summary)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

This notebook implements a complete recommender system pipeline on the MovieLens Small dataset. It covers six parts and thirteen tasks, going from simple genre-based similarity all the way to reinforcement learning agents and model explainability.

| Part | Topic | Tasks | Marks |
|------|-------|-------|-------|
| 1 | Content-Based Filtering | 1-2 | 20 |
| 2 | Collaborative Filtering | 3-4 | 20 |
| 3 | Matrix Factorization | 5-6 | 20 |
| 4 | Hybrid Recommendation | 7 | 10 |
| 5 | Learning-Based (NN + RL) | 8-9 | 40 |
| 6 | Explainability | 10-13 | 10 |
| | Total | 13 tasks | 120 |

---

## Repository Structure

```
project/
|
|-- CSL7110_Assignment3_Recommender_Systems.ipynb
|-- README.md
|
|-- ml-latest-small/
    |-- movies.csv
    |-- ratings.csv
    |-- tags.csv
    |-- links.csv
    |-- README.txt
```

The `ml-latest-small/` folder must be placed in the same directory as the notebook before running.

---

## Dataset

The MovieLens Small dataset is a standard benchmark for recommender system research, maintained by GroupLens at the University of Minnesota.

| File | Description | Rows |
|------|-------------|------|
| movies.csv | Movie titles and genre labels | 9,742 |
| ratings.csv | User ratings on a 0.5 to 5.0 scale | 100,836 |
| tags.csv | Free-text tags applied by users | 3,683 |
| links.csv | IMDb and TMDb cross-reference IDs | 9,742 |

Download link: https://grouplens.org/datasets/movielens/latest/

Download `ml-latest-small.zip`, extract it, and place the resulting folder next to the notebook.

Quick stats:

```
Users         :   610
Movies        : 9,742
Ratings       : 100,836
Rating scale  : 0.5 to 5.0
Matrix density: ~1.7%  (very sparse)
```

---

## Dependencies and Installation

### Required libraries

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

| Library | Minimum version | Used for |
|---------|----------------|----------|
| numpy | 1.21 | Array operations, SVD |
| pandas | 1.3 | Data loading and manipulation |
| scikit-learn | 0.24 | TF-IDF, MLP, Ridge regression, metrics |
| scipy | 1.7 | Sparse truncated SVD (svds) |
| matplotlib | 3.4 | All plots |
| seaborn | 0.11 | Heatmaps and styled charts |

### To run the notebook interactively

```bash
pip install jupyterlab
```

### Note on libraries that are not installed

The following libraries are implemented from scratch inside the notebook, so you do not need to install them separately:

- `scikit-surprise` — Funk SVD with SGD, user and item biases, regularization
- `tensorflow` / `pytorch` — replaced with `sklearn.neural_network.MLPRegressor`
- `shap` — replaced with permutation-based feature importance
- `lime` — replaced with local perturbation and a Ridge surrogate model

---

## How to Run

### Step 1 — Set up the directory

Make sure the layout looks like this:

```
project/
|-- CSL7110_Assignment3_Recommender_Systems.ipynb
|-- README.md
|-- ml-latest-small/
    |-- movies.csv
    |-- ratings.csv
    ...
```

### Step 2 — Install dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn jupyterlab
```

### Step 3 — Launch Jupyter

```bash
# JupyterLab
jupyter lab

# or classic Jupyter Notebook
jupyter notebook
```

### Step 4 — Run the notebook

Open `CSL7110_Assignment3_Recommender_Systems.ipynb` and select Kernel > Restart & Run All.

Full execution takes roughly 10 to 20 minutes. The slower parts are Item-CF (large item-item similarity matrix) and the RL simulation.

### Running without a browser

```bash
jupyter nbconvert --to notebook --execute \
    CSL7110_Assignment3_Recommender_Systems.ipynb \
    --output executed_output.ipynb \
    --ExecutePreprocessor.timeout=1200
```

---

## Notebook Walkthrough

### Part 1 — Content-Based Filtering

#### Task 1: TF-IDF Recommender

Movie genres are treated as documents. The TfidfVectorizer from scikit-learn builds a (9742 x 21) matrix where each row is a movie and each column is a genre token. Cosine similarity is then computed between every pair of movies, and a lookup function returns the top-N most similar movies for any given title.

Sample output:

```
Input movie: Toy Story (1995)

Recommended Movie                       Cosine Similarity
Antz (1998)                                      1.000
Toy Story 2 (1999)                               1.000
Monsters, Inc. (2001)                            1.000
Emperor's New Groove, The (2000)                 1.000
Adventures of Rocky and Bullwinkle (2000)        1.000
```

The similarity scores are 1.0 here because these movies share the exact same set of genres (Adventure, Animation, Children, Comedy, Fantasy). Movies with partially overlapping genres will have similarity scores between 0 and 1.

#### Task 2: User Profile Recommender

Rather than comparing movies directly, this task builds a profile vector for each user. The profile is a weighted average of the TF-IDF vectors of all movies the user has rated, where the weights are the actual ratings given.

```
                sum over rated movies (rating_m x tfidf_vector_m)
user_profile =  ------------------------------------------------
                         sum of all ratings given
```

Movies the user rated highly contribute more to their profile. The profile is then compared against all unrated movies using cosine similarity to generate recommendations. Evaluation uses Precision@10 and Recall@10.

---

### Part 2 — Collaborative Filtering

#### Task 3: User-Based CF

The user-movie matrix is (610 x 9724) and roughly 98% sparse. Before computing similarity, each user's ratings are mean-centered to remove the effect of lenient or harsh raters. Pearson correlation is then computed between all user pairs.

For prediction, the K most similar users are selected and a weighted average of their ratings is used, adjusted by each user's mean:

```
predicted(u, m) = mean(u) + weighted_sum( sim(u,k) * (rating(k,m) - mean(k)) )
                             ----------------------------------------------------
                                          sum( |sim(u,k)| )
```

Effect of K on RMSE:

```
K =  5   RMSE = 1.0412
K = 10   RMSE = 1.0301
K = 20   RMSE = 1.0243
K = 30   RMSE = 1.0198
K = 50   RMSE = 1.0175
```

Accuracy improves with more neighbors but plateaus beyond K=30. K=20 is a reasonable default.

#### Task 4: Item-Based CF

The matrix is transposed so rows become movies and columns become users. Pearson correlation is computed between item pairs. For prediction, the K most similar items that the target user has already rated are found, and a weighted average of those ratings becomes the predicted score.

Item-CF RMSE (0.9841) is better than User-CF (1.0243) on this dataset.

On why item-CF is more practical at scale: the number of items grows slowly compared to users. Item-item similarity can be precomputed once and cached. When a new user signs up, no recomputation is needed. In systems with hundreds of millions of users and a fixed catalogue, item-CF is far more efficient than recomputing a user-user similarity matrix continuously.

---

### Part 3 — Matrix Factorization

#### Task 5: SVD with NumPy/SciPy

Missing ratings are filled with each user's mean before decomposition. The matrix is then globally mean-centered and passed to scipy's svds for truncated SVD with k latent factors.

```
R  (610 x 9724)  -->  U (610 x k)  *  diag(sigma)  *  Vt (k x 9724)

Reconstruction:
R_hat = U * diag(sigma_k) * Vt + global_mean
Clip predictions to [0.5, 5.0]
```

Sensitivity to k:

```
k =  10   RMSE = 0.9891
k =  20   RMSE = 0.9634
k =  50   RMSE = 0.9423
k = 100   RMSE = 0.9318
k = 150   RMSE = 0.9291
```

#### Task 6: Surprise-Style SVD (implemented from scratch)

This follows the Funk SVD formulation used by the scikit-surprise library, trained with stochastic gradient descent:

```
r_hat(u, i) = global_mean + bias_u + bias_i + dot(p_u, q_i)
```

SGD update per observed rating:

```
error   = actual_rating - r_hat
bias_u += lr * (error - reg * bias_u)
bias_i += lr * (error - reg * bias_i)
p_u    += lr * (error * q_i - reg * p_u)
q_i    += lr * (error * p_u_old - reg * q_i)
```

This approach only trains on observed ratings, which avoids the noise introduced by mean-imputation. It also models user and item biases explicitly. As a result it achieves RMSE 0.9077, compared to 0.9423 for the imputed NumPy SVD.

---

### Part 4 — Hybrid Recommendation Model

#### Task 7: Meta-Learning Hybrid

A Ridge Regression meta-model is trained to blend CBF and CF signals. For each (user, movie) pair in the training set, four features are computed:

- CBF score: cosine similarity between user profile and movie TF-IDF vector
- CF score: predicted rating from the Surprise SVD model
- Movie average rating: popularity signal
- User average rating: accounts for user-level rating bias

The meta-model learns weights for each signal from historical ratings.

Learned coefficients:

```
CBF score          0.0412
SVD CF score       0.8873
Movie avg rating   0.0621
User avg rating    0.1204
```

CF dominates, which makes sense on a dense dataset. CBF is more useful for cold-start users who have few or no ratings.

Cold-start analysis:

```
Warm users (more than 20 ratings):  RMSE = 0.8671
Cold-start users (5 or fewer):      RMSE = 1.1243
```

The hybrid model degrades gracefully for cold-start users because the CBF component can still generate recommendations from genre information alone, without needing any collaborative signal.

---

### Part 5 — Learning-Based Recommender Systems

#### Task 8: Neural Network CBF

Two feature branches are built separately then concatenated:

```
User features (19 values: avg rating per genre)
    --> Dense(64, ReLU) --> user embedding

Movie features (21 values: one-hot genres + release year + avg rating)
    --> Dense(64, ReLU) --> movie embedding

Concatenated (64-dim)
    --> Dense(32, ReLU)
    --> Dense(1)  --> predicted rating
```

Trained with MSE loss and Adam optimizer (lr=0.001). Early stopping monitors validation loss to prevent overfitting. Test RMSE: 0.9712.

The neural model captures non-linear interactions between genre preferences and movie attributes, but on this relatively small and sparse dataset it does not outperform CF-based methods, which benefit from rich collaborative signal.

#### Task 9: Reinforcement Learning

Three strategies are implemented.

Epsilon-Greedy Bandit (epsilon=0.1): with probability 0.1, recommend a random movie (explore). Otherwise recommend the movie with the highest estimated reward (exploit). Reward is +1 for rating >= 4, -1 for rating < 4, 0 for no rating.

UCB Bandit (c=2):

```
score(arm) = Q(arm) + c * sqrt( ln(t) / n(arm) )
```

Movies with fewer interactions get a bonus to encourage trying them. This naturally balances exploration and exploitation without a fixed epsilon.

Q-Learning Agent: state is defined as the user's last reward context (positive, negative, or neutral). The Q-table is updated after each recommendation:

```
Q(s, a) = Q(s, a) + alpha * [ reward + gamma * max Q(s', a') - Q(s, a) ]

alpha = 0.1  (learning rate)
gamma = 0.9  (discount factor)
```

Cumulative reward after 1000 rounds:

```
Epsilon-Greedy MAB :   87
Q-Learning         :  119
UCB                :  134
```

UCB performs best because it directs exploration toward movies with genuine uncertainty rather than purely random sampling.

---

### Part 6 — Explainability

#### Task 10: Feature-Based Explanations

For each feature, its column in the test set is randomly permuted and the increase in MSE is measured. A larger drop in performance means the feature was more important.

Example output:

```
movie_avg_rating    +0.0412   most important
user_Drama          +0.0387
movie_Drama         +0.0341
user_Comedy         +0.0298
movie_year          +0.0187
```

A textual explanation is also generated per recommendation, for example: "This movie was recommended because you consistently rate Drama and Crime films highly, and this movie belongs to both genres."

#### Task 11: Neighborhood-Based Explanations

For User-CF, the explanation shows which similar users influenced the prediction:

```
Explaining: User 1 --> The Shawshank Redemption (1994)

Similar User   Similarity   Their Rating
          73       0.8421            5.0
         452       0.8213            5.0
         380       0.7994            4.5
          89       0.7612            4.0

Users most similar to you gave this film an average of 4.8 out of 5.
```

For Item-CF, the explanation shows which previously rated movies are most similar to the recommended one.

#### Task 12: LIME Explainability

LIME fits a local linear model around a single prediction to approximate what the neural network is doing in that region of the input space. Steps:

1. Take one instance (user + movie feature vector)
2. Generate 200 perturbed versions by adding Gaussian noise
3. Predict ratings for all perturbed versions using the neural network
4. Weight each perturbed sample by its distance from the original (Gaussian kernel)
5. Fit a Ridge Regression on the weighted perturbed data
6. The Ridge coefficients are the local feature importances

The output shows which features pushed the prediction higher or lower for that specific user-movie pair.

#### Task 13: Explainability Evaluation

| Method | Clarity to user | Useful for auditing bias | Computational cost |
|--------|----------------|-------------------------|--------------------|
| Permutation importance | Good (global view) | Yes, reveals popularity bias | Medium |
| k-NN neighborhood | Very good (intuitive) | Yes, reveals echo chamber effect | Low |
| LIME | Moderate (local view) | Yes, catches local anomalies | High |

Neighborhood explanations are the most immediately understandable. Permutation importance is best for spotting systemic biases across the whole model. LIME is most useful for debugging individual surprising predictions.

---

## Results Summary

| Method | RMSE | Precision@10 | Recall@10 |
|--------|------|--------------|-----------|
| User-CF (K=20) | 1.0243 | 0.1267 | 0.0389 |
| Item-CF (K=20) | 0.9841 | — | — |
| SVD (numpy, k=50) | 0.9423 | 0.1540 | 0.0501 |
| Surprise SVD (SGD) | 0.9077 | 0.1612 | 0.0538 |
| Neural Network (MLP) | 0.9712 | — | — |
| Hybrid (Meta-Ridge) | 0.8972 | — | — |

The hybrid model achieves the lowest RMSE overall by blending content-based and collaborative signals. Among single methods, Surprise SVD performs best because it avoids imputation noise and explicitly models biases. Item-CF beats User-CF on both accuracy and practical scalability. The neural network is competitive but does not have enough signal on this small dataset to outperform CF methods.

---

## Troubleshooting

**FileNotFoundError on movies.csv or ratings.csv**

The dataset folder must sit next to the notebook. Check the layout with:

```bash
ls -la
# Should show both the notebook and ml-latest-small/ in the same folder
```

**ModuleNotFoundError**

Run the install command again:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

If you are inside a conda environment:

```bash
conda install numpy pandas scikit-learn scipy matplotlib seaborn
```

**Kernel crash or memory error on Item-CF or SVD**

The item-item similarity matrix for roughly 9700 movies is large (around 700 MB as float64). If the machine has less than 8 GB RAM, filter to movies with at least 50 ratings before running Task 4:

```python
popular_movies = ratings.groupby('movieId').size()
popular_movies = popular_movies[popular_movies >= 50].index
ratings = ratings[ratings['movieId'].isin(popular_movies)]
```

**Execution taking more than 30 minutes**

Reduce sample sizes in the slower cells:

```python
# Tasks 3 and 4 — smaller test sample
test_sample = test_ratings.sample(200, random_state=42)

# Task 9 — fewer RL rounds
N_ROUNDS = 300
```

**scikit-surprise import error**

This is expected and not a problem. The notebook does not import scikit-surprise. Task 6 implements Funk SVD from scratch using only numpy, so no additional installation is required.

---


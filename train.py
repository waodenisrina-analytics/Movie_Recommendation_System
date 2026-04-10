# ==============================================================================
# Final Project Nalar Bootcamp 
# Movie Recommendation System - Content-Based Filtering with TMDB Movie Dataset   
# -------------------------------------------------------------------------------
 # Author     : Wa Ode Nisrina Sayyidah Hidayat                 
 # Date       : 2026`
# ==============================================================================
# Train.py

import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAW_PATH   = "TMDB_movie_dataset_finall.csv"   # <-- sesuaikan path dataset kamu
ARTIFACTS  = "artifacts"
MIN_VOTES  = 100
MIN_RATING = 1.0

os.makedirs(ARTIFACTS, exist_ok=True)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("[1/5] Loading dataset …")
df = pd.read_csv(RAW_PATH)

# ==============================================================================
# 2. WEIGHTED SCORE (Bayesian Average)
# ==============================================================================
print("[3/5] Computing weighted score …")
C = df["vote_average"].mean()
m = df["vote_count"].quantile(0.60)

df["weighted_score"] = df.apply(
    lambda row: (row["vote_count"] / (row["vote_count"] + m)) * row["vote_average"]
              + (m                 / (row["vote_count"] + m)) * C,
    axis=1
)

# Normalize weighted_score → [0, 1] supaya siap di-blend dengan cosine
scaler = MinMaxScaler()
df["ws_norm"] = scaler.fit_transform(df[["weighted_score"]])

# ==============================================================================
# 3. TF-IDF VECTORIZATION
# ==============================================================================
print("[4/5] Fitting TF-IDF …")
tfidf = TfidfVectorizer(
    max_features = 15_000,
    ngram_range  = (1, 2),
    min_df       = 2,
    sublinear_tf = True,
    stop_words   = "english"
)
tfidf_matrix = tfidf.fit_transform(df["combined_features"])
print(f"      TF-IDF matrix: {tfidf_matrix.shape}  "
      f"sparsity={1 - tfidf_matrix.nnz / np.prod(tfidf_matrix.shape):.3%}")

# ==============================================================================
# 4. SAVE ARTIFACTS
# ==============================================================================
print("[5/5] Saving artifacts …")

# Simpan sparse matrix
save_npz(os.path.join(ARTIFACTS, "tfidf_matrix.npz"), tfidf_matrix)

# Simpan vectorizer
joblib.dump(tfidf, os.path.join(ARTIFACTS, "tfidf_vectorizer.pkl"))

# Simpan dataframe (hanya kolom yang dibutuhkan app)
keep_cols = ["title", "overview", "genres", "keywords",
             "vote_average", "vote_count", "weighted_score", "ws_norm"]
df[keep_cols].to_pickle(os.path.join(ARTIFACTS, "movies_df.pkl"))

print("\n✅  Training selesai. Artifacts tersimpan di folder 'artifacts/'")
print(f"    • artifacts/tfidf_matrix.npz")
print(f"    • artifacts/tfidf_vectorizer.pkl")
print(f"    • artifacts/movies_df.pkl")
print(f"\n    Total film: {len(df):,}  |  C={C:.3f}  |  m={m:.0f}")

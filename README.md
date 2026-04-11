# 🎬 CineMatch: Movie Recommendation System

Aplikasi rekomendasi film berbasis **Content-Based Filtering** (TF-IDF + Cosine Similarity + Bayesian Average) menggunakan Streamlit.

Link aplikasi (deployment dengan streamlit) : https://movierecommendationsystem-kaphgyrpffauypm9qfqsxx.streamlit.app/

---

## 📁 Struktur Folder

```text
movie-recommender/
│
├── TMDB_movie_dataset_finall.csv      
│
├── train.py                   
├── app.py                      
├── requirements.txt
├── README.md
│
└── artifacts/                  
    ├── tfidf_matrix.npz
    ├── tfidf_vectorizer.pkl
    └── movies_df.pkl

---
```

## 🚀 Cara Menjalankan Aplikasi

### 1. Unduh Dataset
Dataset raw (untuk menjalankan notebook) : https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies/data                                                                                                                  
Dataset clean : https://github.com/waodenisrina-analytics/Movie_Recommendation_System/blob/main/TMDB_movie_dataset_finall.csv

### 2. Masukkan Dataset, train.py, serta app.py dalam 1 folder

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan training 
```bash
python train.py
```
Proses ini akan:
- Membersihkan dan memfilter data
- Membuat TF-IDF matrix
- Menghitung Weighted Score
- Menyimpan semua artifacts ke folder `artifacts/`

### 5. Jalankan aplikasi Streamlit
```bash
streamlit run app.py
```
Buka browser di `http://localhost:8501`

---

## 🎛️ Fitur Aplikasi

| Fitur | Keterangan |
|---|---|
| 🔍 Search film | Dropdown dengan search/autocomplete |
| 🎬 Top-N rekomendasi | Slider 5–20 film |
| ⚖️ Bobot similarity vs kualitas | Slider 0–1 (default 70:30) |
| 🎭 Filter genre | Saring rekomendasi per genre |
| 📊 Similarity bar | Visualisasi skor kemiripan |
| 📝 Cuplikan sinopsis | Toggle on/off |
| ⬇️ Download CSV | Export hasil rekomendasi |

---

## ⚙️ Konfigurasi Model

| Parameter | Nilai | Keterangan |
|---|---|---|
| `max_features` | 15,000 | Term TF-IDF terbaik |
| `ngram_range` | (1, 2) | Unigram + bigram |
| `min_df` | 2 | Hapus term sangat jarang |
| `sublinear_tf` | True | Log normalization TF |
| `MIN_VOTES` | 100 | Filter noise film langka |
| `ws_quantile` | 0.60 | Threshold min votes Bayesian |
| `weight_sim` | 0.70 | 70% similarity + 30% quality |

---

## 🧠 Arsitektur Pipeline

```
CSV Dataset
    │
    ▼
[ train.py ]
    ├─ Filter (vote_count ≥ 100, rating ≥ 1.0)
    ├─ Text Cleaning (lowercase, remove punctuation)
    ├─ Feature Engineering:
    │   ├─ combined_features = overview + genres×2 + keywords×2
    │   └─ weighted_score (Bayesian Average)
    ├─ TF-IDF Vectorization (15k terms, bigrams)
    └─ Simpan artifacts/
    
[ app.py ]
    ├─ Load artifacts (cached)
    ├─ User pilih film → cari idx
    ├─ Cosine Similarity (on-demand, 1 baris)
    ├─ Combined Score = 0.7×sim + 0.3×ws_norm
    └─ Tampilkan Top-N rekomendasi
```

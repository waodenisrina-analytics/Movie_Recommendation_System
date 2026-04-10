# ==============================================================================
# Final Project Nalar Bootcamp 
# Movie Recommendation System - Content-Based Filtering with TMDB Movie Dataset   
# -------------------------------------------------------------------------------
 # Author     : Wa Ode Nisrina Sayyidah Hidayat                 
 # Date       : 2026`
# ==============================================================================
# App.py

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "🎬 CineMatch — Movie Recommender",
    page_icon   = "🎬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Font & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0D0F14;
    color: #E8E9F0;
}

/* ── Header ── */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E8E9F0 0%, #A78BFA 50%, #60A5FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
}
.main-header p {
    color: #6B7280;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ── Query card ── */
.query-card {
    background: #161820;
    border: 1px solid #252836;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
}
.query-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #A78BFA;
    margin: 0 0 0.3rem;
}
.query-meta {
    font-size: 0.82rem;
    color: #6B7280;
}

/* ── Movie card ── */
.movie-card {
    background: #161820;
    border: 1px solid #252836;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.65rem;
    transition: border-color .2s, transform .2s;
    position: relative;
}
.movie-card:hover {
    border-color: #7C3AED44;
    transform: translateX(4px);
}
.movie-rank {
    position: absolute;
    top: 1rem;
    right: 1.2rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #252836;
}
.movie-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #E8E9F0;
    margin-bottom: 0.25rem;
}
.movie-genres {
    font-size: 0.75rem;
    color: #A78BFA;
    margin-bottom: 0.5rem;
    letter-spacing: 0.4px;
}
.movie-meta {
    display: flex;
    gap: 1.2rem;
    align-items: center;
    flex-wrap: wrap;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 20px;
}
.badge-rating {
    background: #1C2233;
    color: #FBBF24;
    border: 1px solid #FBBF2430;
}
.badge-votes {
    background: #1C2233;
    color: #6B7280;
}
.sim-bar-wrap {
    margin-top: 0.55rem;
}
.sim-label {
    font-size: 0.72rem;
    color: #A78BFA;
    margin-bottom: 3px;
    display: flex;
    justify-content: space-between;
}
.sim-bar-bg {
    height: 5px;
    border-radius: 999px;
    background: #9aa2ca;
    overflow: hidden;
}
.sim-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7C3AED, #60A5FA);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0D0F14;
    border-right: 1px solid #1C1E26;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── Misc ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #9CA3AF !important;
    font-size: 0.85rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #3B82F6);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    width: 100%;
    cursor: pointer;
    letter-spacing: 0.5px;
    transition: opacity .2s;
}
.stButton > button:hover {
    opacity: 0.88;
}
.divider {
    height: 1px;
    background: #252836;
    margin: 1.5rem 0;
}
.stat-box {
    background: #161820;
    border: 1px solid #252836;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-box .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #A78BFA;
}
.stat-box .lbl {
    font-size: 0.72rem;
    color: #4B5563;
    margin-top: 2px;
}
.not-found {
    background: #1C1620;
    border: 1px solid #7C3AED33;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #6B7280;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD ARTIFACTS  (cached agar hanya load sekali)
# ══════════════════════════════════════════════════════════════════════════════
ARTIFACTS = "artifacts"

@st.cache_resource(show_spinner="⏳ Memuat model …")
def load_artifacts():
    """Load TF-IDF matrix, vectorizer, dan DataFrame dari disk."""
    matrix_path = os.path.join(ARTIFACTS, "tfidf_matrix.npz")
    vect_path   = os.path.join(ARTIFACTS, "tfidf_vectorizer.pkl")
    df_path     = os.path.join(ARTIFACTS, "movies_df.pkl")

    if not all(os.path.exists(p) for p in [matrix_path, vect_path, df_path]):
        return None, None, None

    tfidf_matrix = load_npz(matrix_path)
    tfidf        = joblib.load(vect_path)
    df           = pd.read_pickle(df_path)

    # Buat index title → row idx (lowercase)
    title_to_idx = pd.Series(df.index, index=df["title"].str.lower())

    return df, tfidf_matrix, title_to_idx


# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def get_recommendations(
    title_lower   : str,
    df            : pd.DataFrame,
    tfidf_matrix,
    title_to_idx  : pd.Series,
    top_n         : int = 10,
    weight_sim    : float = 0.70,
    genre_filter  : str  = "All",
) -> tuple[pd.DataFrame | None, int | None, str]:
    """
    Returns (result_df, query_idx, error_message).
    error_message kosong jika sukses.
    """
    # ── Cari idx ──────────────────────────────────────────────────────────────
    if title_lower not in title_to_idx.index:
        # Fuzzy: cari judul yang mengandung query
        matches = [t for t in title_to_idx.index if title_lower in t]
        if not matches:
            return None, None, f"Film **'{title_lower}'** tidak ditemukan."
        title_lower = matches[0]

    idx_val   = title_to_idx[title_lower]
    query_idx = int(idx_val) if isinstance(idx_val, (int, np.integer)) \
                else int(idx_val.iloc[0])

    # ── Cosine similarity (on-demand, 1 baris) ────────────────────────────────
    query_vec  = tfidf_matrix[query_idx]
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    candidates = df.copy()
    candidates["similarity_score"] = sim_scores
    candidates = candidates[candidates.index != query_idx]

    # ── Genre filter ──────────────────────────────────────────────────────────
    if genre_filter != "All":
        mask = candidates["genres"].str.contains(genre_filter, case=False, na=False)
        candidates = candidates[mask]

    if candidates.empty:
        return None, query_idx, "Tidak ada film dengan genre tersebut di dataset."

    # ── Combined Score: α × sim + (1-α) × ws_norm ────────────────────────────
    weight_ws = 1.0 - weight_sim
    candidates["final_score"] = (
        weight_sim * candidates["similarity_score"] +
        weight_ws  * candidates["ws_norm"]
    )
    candidates = candidates.sort_values("final_score", ascending=False)

    result = candidates.head(top_n)[
        ["title", "genres", "vote_average", "vote_count",
         "similarity_score", "weighted_score", "overview"]
    ].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    result = result.reset_index(drop=True)

    return result, query_idx, ""


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def render_movie_card(row, show_sim: bool, show_overview: bool):
    # 1. Persiapan Data
    sim_pct = int(row["similarity_score"] * 100)
    fill_w  = f"{sim_pct}%"
    genres  = row["genres"][:80] + ("…" if len(row["genres"]) > 80 else "")
    
    # 2. Sinopsis (Hanya satu blok IF)
    overview_html = ""
    if show_overview and row.get("overview"):
        full_overview = str(row["overview"]) 
        overview_html = (
            f'<p style="font-size:0.85rem; color:#9CA3AF; margin-top:0.8rem; '
            f'line-height:1.6; text-align:justify; max-height:150px; '
            f'overflow-y:auto; padding-right:5px;">'
            f'{full_overview}</p>'
        )

    # 3. Similarity Bar
    sim_section = ""
    if show_sim:
        sim_section = (
            f'<div class="sim-bar-wrap">'
            f'<div class="sim-label"><span>Similarity</span><span>{sim_pct}%</span></div>'
            f'<div class="sim-bar-bg"><div class="sim-bar-fill" style="width:{fill_w}"></div></div>'
            f'</div>'
        )

    # 4. Render to Streamlit
    st.markdown(f"""
    <div class="movie-card">
      <div class="movie-rank">#{int(row['rank'])}</div>
      <div class="movie-title">{row['title']}</div>
      <div class="movie-genres">{genres}</div>
      <div class="movie-meta">
        <span class="badge badge-rating">⭐ {row['vote_average']:.1f}</span>
        <span class="badge badge-votes">👥 {int(row['vote_count']):,} votes</span>
      </div>
      {sim_section}
      {overview_html}
    </div>
    """, unsafe_allow_html=True)

def render_query_card(query_film: pd.Series):
    overview = str(query_film.get("overview", "Sinopsis tidak tersedia."))
    
    st.markdown(f"""
    <div class="query-card">
      <h4>🎯 {query_film['title']}</h4>
      <div class="movie-genres" style="margin-bottom:0.4rem">{query_film['genres']}</div>
      <div class="query-meta" style="margin-bottom: 0.8rem;">
        ⭐ {query_film['vote_average']:.1f} &nbsp;|&nbsp;
        👥 {int(query_film['vote_count']):,} votes
      </div>
      
      <div style="font-size:0.85rem; color:#9CA3AF; line-height:1.6; text-align:justify; border-top:1px solid #252836; padding-top:0.8rem;">
        {overview}
      </div>
      
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
      <h1>🎬 CineMatch</h1>
      <p>Content-Based Movie Recommendation · TF-IDF + Cosine Similarity</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df, tfidf_matrix, title_to_idx = load_artifacts()

    if df is None:
        st.error(
            "⚠️ **Artifacts belum ditemukan.**\n\n"
            "Jalankan terlebih dahulu:\n```bash\npython train.py\n```\n"
            "untuk membuat folder `artifacts/` berisi model yang diperlukan."
        )
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:1.1rem;"
            "font-weight:800;color:#A78BFA;margin-bottom:1.2rem'>⚙️ Settings</div>",
            unsafe_allow_html=True
        )

        top_n = st.slider("Jumlah rekomendasi", 5, 20, 10, step=5)

        weight_sim = st.slider(
            "Bobot Similarity (vs Quality)",
            min_value=0.0, max_value=1.0, value=0.70, step=0.05,
            help="0.7 = 70% cosine similarity + 30% weighted score"
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Genre filter
        all_genres_list = sorted(set(
            g.strip()
            for genres_str in df["genres"].dropna()
            for g in genres_str.split(",")
            if g.strip()
        ))
        genre_filter = st.selectbox(
            "Filter Genre Rekomendasi",
            options=["All"] + all_genres_list
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        show_sim      = st.toggle("Tampilkan similarity bar", value=True)
        show_overview = st.toggle("Tampilkan cuplikan sinopsis", value=False)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
              <div class="val">{len(df):,}</div>
              <div class="lbl">Total Film</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
              <div class="val">{tfidf_matrix.shape[1]:,}</div>
              <div class="lbl">TF-IDF Terms</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(
            "<div style='margin-top:1.5rem;font-size:0.72rem;color:#374151;"
            "line-height:1.6'>Model: TF-IDF (15k terms, bigrams) + "
            "Cosine Similarity + Bayesian Weighted Score</div>",
            unsafe_allow_html=True
        )

    # ── Search Box ───────────────────────────────────────────────────────────
    titles_sorted = sorted(df["title"].dropna().unique().tolist())

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        selected_title = st.selectbox(
            "Cari & pilih film",
            options=[""] + titles_sorted,
            index=0,
            placeholder="Ketik nama film …",
            label_visibility="collapsed",
        )
    with col_btn:
        recommend_btn = st.button("🎬 Recommend", use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Output ────────────────────────────────────────────────────────────────
    if recommend_btn:
        if not selected_title:
            st.warning("⚠️ Pilih atau ketik judul film terlebih dahulu.")
            st.stop()

        with st.spinner("Menghitung rekomendasi …"):
            result, query_idx, err = get_recommendations(
                title_lower  = selected_title.lower().strip(),
                df           = df,
                tfidf_matrix = tfidf_matrix,
                title_to_idx = title_to_idx,
                top_n        = top_n,
                weight_sim   = weight_sim,
                genre_filter = genre_filter,
            )

        if err:
            st.markdown(f'<div class="not-found">🔍 {err}</div>',
                        unsafe_allow_html=True)
            st.stop()

        # Query film info
        render_query_card(df.iloc[query_idx])

        # Rekomendasi
        st.markdown(
            f"<div style='font-family:Syne,sans-serif;font-size:0.85rem;"
            f"font-weight:700;color:#4B5563;letter-spacing:1.5px;"
            f"text-transform:uppercase;margin-bottom:0.8rem'>"
            f"Top {len(result)} Rekomendasi</div>",
            unsafe_allow_html=True
        )

        for _, row in result.iterrows():
            render_movie_card(row, show_sim=show_sim, show_overview=show_overview)

        # Download CSV
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        csv = result[["rank","title","genres","vote_average",
                      "vote_count","similarity_score"]].to_csv(index=False)
        st.download_button(
            label    = "⬇️ Download hasil sebagai CSV",
            data     = csv,
            file_name= f"rekomendasi_{selected_title.replace(' ','_')}.csv",
            mime     = "text/csv",
        )

    else:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:#374151">
          <div style="font-size:3rem;margin-bottom:1rem">🎞️</div>
          <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;color:#4B5563">
            Pilih film dari dropdown di atas,<br>lalu klik <span style="color:#A78BFA">Recommend</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

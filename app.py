"""
DANA Review Intelligence System — Streamlit App
Production-ready sentiment analysis & semantic retrieval interface
"""

import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="DANA Sentiment Intelligence",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main { background-color: #0D1117; }
.stApp { background-color: #0D1117; }

.title-block {
    text-align: center;
    padding: 2rem 0 1rem;
}
.title-block h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #E6EDF3;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.title-block p {
    color: #8B949E;
    font-size: 0.9rem;
}

.metric-positive {
    background: linear-gradient(135deg, #0D2818, #1A4731);
    border: 1px solid #238636;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    text-align: center;
}
.metric-negative {
    background: linear-gradient(135deg, #2D0D0D, #4D1A1A);
    border: 1px solid #E74C3C;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    text-align: center;
}
.metric-neutral {
    background: linear-gradient(135deg, #1A1A2E, #2A2A4A);
    border: 1px solid #9B59B6;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    text-align: center;
}
.metric-label {
    font-size: 0.75rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #E6EDF3;
}
.metric-sub {
    font-size: 0.8rem;
    color: #8B949E;
    margin-top: 0.2rem;
}
.similar-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-left: 3px solid #1F6FEB;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #E6EDF3;
    line-height: 1.5;
}
.similar-score {
    font-size: 0.75rem;
    color: #8B949E;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
.stTextArea textarea {
    background-color: #161B22 !important;
    color: #E6EDF3 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
.stButton > button {
    background: #1F6FEB;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.5rem 1.5rem;
    transition: background 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: #388BFD;
}
.divider {
    border: none;
    border-top: 1px solid #30363D;
    margin: 1.5rem 0;
}
.warning-box {
    background: #2D1B00;
    border: 1px solid #F39C12;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #F39C12;
    margin: 0.5rem 0;
}
.info-pill {
    display: inline-block;
    background: #1F2937;
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: #9CA3AF;
    margin: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


ARTIFACTS = {
    "model":   "sentiment_model.keras",
    "encoder": "label_encoder.pkl",
    "knn":     "nearest_neighbors.pkl",
    "reviews": "reviews.pkl",
}

def check_artifacts() -> tuple[bool, list[str]]:
    missing = [k for k, v in ARTIFACTS.items() if not os.path.exists(v)]
    return len(missing) == 0, missing


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("intfloat/multilingual-e5-large")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    import tensorflow as tf
    model   = tf.keras.models.load_model(ARTIFACTS["model"])
    encoder = joblib.load(ARTIFACTS["encoder"])
    knn     = joblib.load(ARTIFACTS["knn"])
    reviews = joblib.load(ARTIFACTS["reviews"])
    return model, encoder, knn, reviews


def encode(text: str, embedding_model, prefix: str = "query") -> np.ndarray:
    return embedding_model.encode([f"{prefix}: {text}"], normalize_embeddings=True)


def predict_sentiment(text: str, embedding_model, model, encoder) -> dict:
    emb   = encode(text, embedding_model, prefix="query")
    probs = model.predict(emb, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = encoder.inverse_transform([idx])[0]

    all_labels = encoder.classes_
    prob_map   = {lbl: float(probs[encoder.transform([lbl])[0]])
                  for lbl in all_labels}

    return {"label": label, "confidence": float(probs[idx]), "probs": prob_map}


def find_similar(text: str, embedding_model, knn, reviews, k: int = 5) -> list[dict]:
    emb = encode(text, embedding_model, prefix="query")
    distances, indices = knn.kneighbors(emb, n_neighbors=k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        cosine_similarity = 1.0 - float(dist)  
        results.append({
            "text":       str(reviews.iloc[idx]),
            "similarity": cosine_similarity,
            "rank":       len(results) + 1,
        })
    return results


st.markdown("""
<div class='title-block'>
    <h1>🔍 DANA Review Intelligence</h1>
    <p>Multilingual Sentiment Classification & Semantic Retrieval Engine</p>
</div>
""", unsafe_allow_html=True)

# Pills
st.markdown("""
<div style='text-align:center; margin-bottom:1.5rem'>
    <span class='info-pill'>multilingual-e5-large</span>
    <span class='info-pill'>50K reviews · 3 classes</span>
    <span class='info-pill'>cosine KNN · k=5</span>
    <span class='info-pill'>Bahasa Indonesia</span>
</div>
""", unsafe_allow_html=True)


artifacts_ok, missing = check_artifacts()

if not artifacts_ok:
    st.markdown(f"""
    <div class='warning-box'>
    ⚠️ <b>Model artifacts not found:</b> {', '.join(missing)}<br>
    Run the training notebook first to generate all required files.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


with st.spinner("Loading models…"):
    emb_model = load_embedding_model()
    clf_model, label_enc, knn_index, review_store = load_artifacts()


st.markdown("<hr class='divider'>", unsafe_allow_html=True)
review_input = st.text_area(
    "Enter a review in Bahasa Indonesia",
    placeholder="contoh: aplikasi sangat mudah digunakan dan transfer cepat...",
    height=120,
    key="review_text",
)

col1, col2 = st.columns(2)
with col1:
    run_classify = st.button("⚡ Classify Sentiment", use_container_width=True)
with col2:
    run_retrieve = st.button("🔎 Find Similar Reviews", use_container_width=True)


if run_classify:
    text = review_input.strip()
    if not text:
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Encoding & predicting…"):
            result = predict_sentiment(text, emb_model, clf_model, label_enc)

        label      = result["label"]
        confidence = result["confidence"]
        probs      = result["probs"]

        css_class = {
            "POSITIVE": "metric-positive",
            "NEGATIVE": "metric-negative",
            "NEUTRAL":  "metric-neutral",
        }.get(label, "metric-neutral")

        emoji_map = {"POSITIVE": "✅", "NEGATIVE": "❌", "NEUTRAL": "➖"}

        st.markdown(f"""
        <div class='{css_class}'>
            <div class='metric-label'>Predicted Sentiment</div>
            <div class='metric-value'>{emoji_map.get(label, "")} {label}</div>
            <div class='metric-sub'>Confidence: {confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Class probability distribution**")
        for lbl, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar_color = {"POSITIVE": "#238636", "NEGATIVE": "#E74C3C", "NEUTRAL": "#9B59B6"}
            st.markdown(f"""
            <div style='margin-bottom:6px'>
                <span style='color:#8B949E; font-size:0.8rem; width:80px; display:inline-block'>{lbl}</span>
                <div style='display:inline-block; background:{bar_color[lbl]}; 
                     width:{prob*200:.0f}px; height:12px; border-radius:3px; vertical-align:middle'></div>
                <span style='color:#E6EDF3; font-size:0.8rem; margin-left:8px; font-family:monospace'>{prob:.3f}</span>
            </div>
            """, unsafe_allow_html=True)


if run_retrieve:
    text = review_input.strip()
    if not text:
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Computing embeddings & searching…"):
            results = find_similar(text, emb_model, knn_index, review_store, k=5)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Top 5 Semantically Similar Reviews**")
        st.markdown(
            f"<p style='color:#8B949E;font-size:0.82rem'>Query: <i>{text[:100]}{'…' if len(text)>100 else ''}</i></p>",
            unsafe_allow_html=True
        )

        for r in results:
            sim_pct = r['similarity'] * 100
            bar_w   = max(10, int(sim_pct * 2.5))
            st.markdown(f"""
            <div class='similar-card'>
                <b style='color:#8B949E;font-size:0.75rem'>#{r['rank']}</b>
                {r['text']}
                <div class='similar-score'>
                    similarity: {r['similarity']:.4f}
                    <div style='display:inline-block; background:#1F6FEB55; 
                         width:{bar_w}px; height:6px; border-radius:3px; 
                         vertical-align:middle; margin-left:8px'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#6E7781; font-size:0.78rem'>
    DANA Review Intelligence System · NoLimit Indonesia Assessment ·
    Model: intfloat/multilingual-e5-large · Yogi Dharma 2025
</p>
""", unsafe_allow_html=True)

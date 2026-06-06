import streamlit as st
import numpy as np
import joblib
import os
import gdown

st.set_page_config(
    page_title="DANA Review Intelligence",
    layout="centered"
)

ARTIFACTS = {
    "label_encoder": "label_encoder.pkl",
    "reviews": "reviews.pkl",
    "knn": "nearest_neighbors.pkl",
    "sentiment_model": "sentiment_model.keras"
}

KNn_FILE = "nearest_neighbors.pkl"
KNN_DRIVE_ID = "1S0aAwqsFzawpaMHZG8XG4Nt7JNSlfsaK"

def download_knn():
    if not os.path.exists(KNn_FILE):
        url = f"https://drive.google.com/uc?id={KNN_DRIVE_ID}"
        gdown.download(url, KNn_FILE, quiet=False)


def check_file(path):
    return os.path.exists(path)


@st.cache_resource
def load_embedding():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("intfloat/multilingual-e5-large")


@st.cache_resource
def load_knn():
    download_knn()  # 👈 auto download sebelum load
    return joblib.load(KNn_FILE)


@st.cache_resource
def load_reviews():
    return joblib.load("reviews.pkl")


def encode(text, model):
    return model.encode([f"query: {text}"], normalize_embeddings=True)


def sentiment_fallback(text):
    positive_words = ["bagus", "cepat", "mudah", "mantap", "baik"]
    negative_words = ["jelek", "lambat", "error", "buruk", "lemot"]

    text_low = text.lower()

    score = 0
    for w in positive_words:
        if w in text_low:
            score += 1
    for w in negative_words:
        if w in text_low:
            score -= 1

    if score > 0:
        return "POSITIVE"
    elif score < 0:
        return "NEGATIVE"
    return "NEUTRAL"


st.title("DANA Review Intelligence System")

text = st.text_area("Input review")

emb_model = load_embedding()
knn = load_knn()
reviews = load_reviews()

col1, col2 = st.columns(2)


def search_similar(text):
    emb = encode(text, emb_model)
    dist, idx = knn.kneighbors(emb, n_neighbors=5)

    results = []
    for i, d in zip(idx[0], dist[0]):
        results.append({
            "text": str(reviews.iloc[i]),
            "score": float(1 - d)
        })
    return results


if col1.button("Sentiment Analysis"):
    if text:
        if check_file("sentiment_model.keras"):
            st.warning("TensorFlow model detected but disabled for deployment stability.")
            result = sentiment_fallback(text)
        else:
            result = sentiment_fallback(text)

        st.success(f"Sentiment: {result}")


if col2.button("Semantic Search"):
    if text:
        results = search_similar(text)

        for r in results:
            st.write(r["text"])
            st.progress(min(r["score"], 1.0))

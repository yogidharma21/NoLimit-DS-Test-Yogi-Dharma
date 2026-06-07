import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import gdown

st.set_page_config(
    page_title="DANA Review Intelligence",
    layout="centered"
)

# =========================
# FILE CONFIG
# =========================

KNN_FILE = "nearest_neighbors.pkl"
KNN_DRIVE_ID = "1S0aAwqsFzawpaMHZG8XG4Nt7JNSlfsaK"

# =========================
# DOWNLOAD MODEL
# =========================

def download_knn():
    if not os.path.exists(KNN_FILE):
        url = f"https://drive.google.com/uc?id={KNN_DRIVE_ID}"
        gdown.download(url, KNN_FILE, quiet=False)

# =========================
# LOAD RESOURCES
# =========================

@st.cache_resource
def load_embedding():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "intfloat/multilingual-e5-large"
    )


@st.cache_resource
def load_knn():
    download_knn()
    return joblib.load(KNN_FILE)


@st.cache_resource
def load_reviews():
    return joblib.load(
        "reviews.pkl"
    )


@st.cache_resource
def load_label_encoder():
    return joblib.load(
        "label_encoder.pkl"
    )


@st.cache_resource
def load_sentiment_model():
    import tensorflow as tf

    return tf.keras.models.load_model(
        "sentiment_model.keras"
    )

# =========================
# EMBEDDING
# =========================

def encode(text, model):
    return model.encode(
        [f"query: {text}"],
        normalize_embeddings=True
    )

# =========================
# SENTIMENT
# =========================

def predict_sentiment(text):

    emb = emb_model.encode(
        [f"passage: {text}"],
        normalize_embeddings=True
    )

    pred = sentiment_model.predict(
        emb,
        verbose=0
    )

    label_idx = np.argmax(
        pred,
        axis=1
    )

    sentiment = label_encoder.inverse_transform(
        label_idx
    )[0]

    return sentiment

# =========================
# SEMANTIC SEARCH
# =========================

def search_similar(text):

    emb = encode(
        text,
        emb_model
    )

    distances, indices = knn.kneighbors(
        emb,
        n_neighbors=5
    )

    results = []

    for idx, dist in zip(
        indices[0],
        distances[0]
    ):

        results.append(
            {
                "text": str(reviews.iloc[idx]),
                "score": float(1 - dist)
            }
        )

    return results

# =========================
# LOAD ALL MODELS
# =========================

with st.spinner("Loading models..."):

    emb_model = load_embedding()

    knn = load_knn()

    reviews = load_reviews()

    label_encoder = load_label_encoder()

    sentiment_model = load_sentiment_model()

# =========================
# UI
# =========================

st.title("DANA Review Intelligence System")

st.markdown(
    """
    Analyze DANA user reviews using:
    - Sentiment Analysis
    - Semantic Search
    """
)

text = st.text_area(
    "Input Review",
    height=150
)

col1, col2 = st.columns(2)

# =========================
# SENTIMENT BUTTON
# =========================

if col1.button("Sentiment Analysis"):

    if text.strip():

        try:

            result = predict_sentiment(
                text
            )

            st.success(
                f"Sentiment: {result}"
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

# =========================
# SEARCH BUTTON
# =========================

if col2.button("Semantic Search"):

    if text.strip():

        try:

            results = search_similar(
                text
            )

            st.subheader(
                "Most Similar Reviews"
            )

            for r in results:

                st.write(
                    r["text"]
                )

                st.progress(
                    min(
                        r["score"],
                        1.0
                    )
                )

        except Exception as e:

            st.error(
                f"Search Error: {e}"
            )

import streamlit as st
import numpy as np
import joblib
import os
import gdown
from pathlib import Path
import tensorflow as tf
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="DANA Review Intelligence",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

KNN_FILE = BASE_DIR / "nearest_neighbors.pkl"
REVIEWS_FILE = BASE_DIR / "reviews.pkl"
LABEL_ENCODER_FILE = BASE_DIR / "label_encoder.pkl"
MODEL_FILE = BASE_DIR / "sentiment_model.h5"

KNN_DRIVE_ID = "1S0aAwqsFzawpaMHZG8XG4Nt7JNSlfsaK"



def download_knn():
    if not KNN_FILE.exists():
        url = f"https://drive.google.com/uc?id={KNN_DRIVE_ID}"
        gdown.download(url, str(KNN_FILE), quiet=False)


@st.cache_resource
def load_embedding():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_resource
def load_knn():
    download_knn()
    return joblib.load(KNN_FILE)


@st.cache_resource
def load_reviews():
    return joblib.load(REVIEWS_FILE)


@st.cache_resource
def load_label_encoder():
    return joblib.load(LABEL_ENCODER_FILE)


@st.cache_resource
def load_sentiment_model():
    return tf.keras.models.load_model(
        MODEL_FILE,
        compile=False,
        custom_objects={
            "BatchNormalization": tf.keras.layers.BatchNormalization
        }
    )


with st.spinner("Loading models..."):
    emb_model = load_embedding()
    knn = load_knn()
    reviews = load_reviews()
    label_encoder = load_label_encoder()
    sentiment_model = load_sentiment_model()


def predict_sentiment(text):
    emb = emb_model.encode(
        [text],
        normalize_embeddings=True
    )

    pred = sentiment_model.predict(emb, verbose=0)
    label_idx = np.argmax(pred, axis=1)

    sentiment = label_encoder.inverse_transform(label_idx)[0]
    return sentiment


def search_similar(text):
    emb = emb_model.encode(
        [text],
        normalize_embeddings=True
    )

    distances, indices = knn.kneighbors(
        emb,
        n_neighbors=5
    )

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "text": str(reviews.iloc[idx]),
            "score": float(1 - dist)
        })

    return results


st.write("Current Directory:", os.getcwd())
st.write("BASE_DIR:", BASE_DIR)

try:
    st.write("Files in BASE_DIR:")
    st.write(os.listdir(BASE_DIR))
except Exception as e:
    st.write(e)


st.title("DANA Review Intelligence System")

st.markdown("""
Analyze DANA user reviews using:
- Sentiment Analysis
- Semantic Search
""")


text = st.text_area("Input Review", height=150)

col1, col2 = st.columns(2)



if col1.button("Sentiment Analysis"):
    if text.strip():
        try:
            result = predict_sentiment(text)
            st.success(f"Sentiment: {result}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")


if col2.button("Semantic Search"):
    if text.strip():
        try:
            results = search_similar(text)

            st.subheader("Most Similar Reviews")

            for r in results:
                st.write(r["text"])
                st.progress(min(r["score"], 1.0))

        except Exception as e:
            st.error(f"Search Error: {e}")

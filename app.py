import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# =====================
# LOAD FILES
# =====================

@st.cache_resource
def load_resources():

    embedding_model = SentenceTransformer(
        "intfloat/multilingual-e5-base"
    )

    model = tf.keras.models.load_model(
        "sentiment_model.keras"
    )

    le = joblib.load(
        "label_encoder.pkl"
    )

    reviews = joblib.load(
        "reviews.pkl"
    )

    # =====================
    # BUILD NN ON STARTUP
    # =====================

    review_emb = embedding_model.encode(
        reviews.tolist(),
        show_progress_bar=False
    )

    nn = NearestNeighbors(
        n_neighbors=5,
        metric="cosine"
    )

    nn.fit(review_emb)

    return (
        embedding_model,
        model,
        le,
        reviews,
        nn
    )


embedding_model, model, le, reviews, nn = load_resources()

# =====================
# UI
# =====================

st.title("DANA Sentiment Analysis")

text = st.text_area(
    "Masukkan Review"
)

# =====================
# SENTIMENT
# =====================

if st.button("Prediksi Sentimen"):

    emb = embedding_model.encode(
        [text]
    )

    pred = model.predict(
        emb,
        verbose=0
    )

    label = le.inverse_transform(
        [np.argmax(pred)]
    )[0]

    st.success(
        f"Prediksi Sentimen: {label}"
    )

# =====================
# SIMILARITY SEARCH
# =====================

if st.button("Cari Review Mirip"):

    query_emb = embedding_model.encode(
        [text]
    )

    distances, indices = nn.kneighbors(
        query_emb
    )

    st.subheader(
        "Top 5 Similar Reviews"
    )

    for i, idx in enumerate(indices[0], start=1):

        st.write(
            f"{i}. {reviews.iloc[idx]}"
        )

import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

from sentence_transformers import SentenceTransformer

# Load model
embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

model = tf.keras.models.load_model(
    "sentiment_model.keras"
)

le = joblib.load(
    "label_encoder.pkl"
)

nn = joblib.load(
    "nearest_neighbors.pkl"
)

reviews = joblib.load(
    "reviews.pkl"
)

st.title("DANA Sentiment Analysis")

text = st.text_area(
    "Masukkan Review"
)

if st.button("Predict"):

    emb = embedding_model.encode([text])

    pred = model.predict(emb)

    label = le.inverse_transform(
        [np.argmax(pred)]
    )[0]

    st.success(
        f"Prediksi Sentimen: {label}"
    )

if st.button("Find Similar Reviews"):

    emb = embedding_model.encode([text])

    distances, indices = nn.kneighbors(emb)

    st.subheader("Top Similar Reviews")

    for idx in indices[0]:
        st.write(
            reviews.iloc[idx]
        )
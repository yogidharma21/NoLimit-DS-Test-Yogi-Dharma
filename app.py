import streamlit as st
import numpy as np
import joblib
import os
import re
import string
import emoji
import nltk
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(page_title="Analisis Sentimen Dana", page_icon="💚", layout="centered")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    border-radius: 20px;
    padding: 40px 36px 32px;
    margin-bottom: 28px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,176,155,0.18);
}
.hero h1 { color: white; font-size: 2.1rem; font-weight: 800; margin: 0 0 6px; }
.hero p  { color: rgba(255,255,255,0.88); font-size: 1rem; margin: 0; }

.card {
    background: white;
    border-radius: 16px;
    padding: 28px 28px 24px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}

.result-card {
    border-radius: 16px;
    padding: 26px 28px;
    margin-top: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    animation: fadeIn .4s ease;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }

.sentiment-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 28px;
    border-radius: 999px;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: .5px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.10);
}

.conf-label {
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .5px;
    text-transform: uppercase;
    opacity: .65;
    margin-bottom: 4px;
}
.conf-value {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 16px;
}

.bar-wrap { margin-bottom: 10px; }
.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: .82rem;
    font-weight: 600;
    margin-bottom: 4px;
}
.bar-track {
    background: rgba(0,0,0,0.07);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width .6s cubic-bezier(.4,0,.2,1);
}

.chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
    margin: 3px 3px 0 0;
    cursor: pointer;
    border: none;
    opacity: .85;
}

.footer {
    text-align: center;
    color: #aaa;
    font-size: .78rem;
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #f0f0f0;
}
</style>
""", unsafe_allow_html=True)


def casefoldingtext(text): return text.lower()

def cleaningtext(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#[A-Za-z0-9]+", "", text)
    text = re.sub(r"RT[\s]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"(\w+)²", r"\1", text)
    text = re.sub(r"\b(\w+)\1\b", r"\1", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = " ".join([w for w in text.split() if len(w) > 1])
    text = emoji.demojize(text)
    text = re.sub(":[A-Za-z_-]+:", " ", text)
    text = re.sub(r"([xX;:]'?[dDpPvVoO3)(])", " ", text)
    text = text.replace("\n", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def fix_slangwords(text, d):
    from nltk.tokenize import word_tokenize
    return " ".join([d.get(w, w) for w in word_tokenize(text)])

def preprocess(text, slang_dict):
    text = casefoldingtext(text)
    text = cleaningtext(text)
    return fix_slangwords(text, slang_dict)


@st.cache_resource(show_spinner=False)
def load_models():
    required = ["sentiment_model.keras", "label_encoder.pkl"]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, f"File tidak ditemukan: {missing}"

    model = tf.keras.models.load_model("sentiment_model.keras")
    le    = joblib.load("label_encoder.pkl")

    emb_dim = model.input_shape[1]
    EMB_MAP = {
        384:  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        768:  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        1024: "sentence-transformers/all-roberta-large-v1",
    }
    emb_name = os.environ.get("EMB_MODEL_NAME", EMB_MAP.get(emb_dim, EMB_MAP[384]))
    emb_model = SentenceTransformer(emb_name)

    try:
        url = "https://raw.githubusercontent.com/ezaaputra/Sentiment-Analysis-Using-BERT/refs/heads/main/kamus_alay.csv"
        slang_dict = {r[0]: r[1] for _, r in pd.read_csv(url).iterrows()}
    except Exception:
        slang_dict = {}

    return {"model": model, "le": le, "emb": emb_model, "slang": slang_dict}, None


CFG = {
    "POSITIVE": {
        "icon": "😊", "label": "Positif",
        "color": "#16a34a", "bg": "#f0fdf4", "bar": "#22c55e",
        "gradient": "linear-gradient(135deg,#d1fae5,#f0fdf4)",
    },
    "NEGATIVE": {
        "icon": "😞", "label": "Negatif",
        "color": "#dc2626", "bg": "#fef2f2", "bar": "#ef4444",
        "gradient": "linear-gradient(135deg,#fee2e2,#fef2f2)",
    },
    "NEUTRAL": {
        "icon": "😐", "label": "Netral",
        "color": "#d97706", "bg": "#fffbeb", "bar": "#f59e0b",
        "gradient": "linear-gradient(135deg,#fef3c7,#fffbeb)",
    },
}

EXAMPLES = [
    "aplikasi sangat membantu dan mudah digunakan",
    "sering error dan tidak bisa transfer sama sekali",
    "biasa saja, tidak ada yang istimewa",
    "dana terbaik untuk pembayaran sehari-hari",
    "saldo hilang setelah transaksi gagal",
]


st.markdown("""
<div class="hero">
    <h1>💚 Analisis Sentimen</h1>
    <p>Deteksi sentimen ulasan aplikasi Dana menggunakan AI · SentenceTransformer + Deep Learning</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Memuat model AI..."):
    res, err = load_models()

if err:
    st.error(f"❌ {err}")
    st.stop()


st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### ✍️ Masukkan Ulasan")


st.markdown("**Coba contoh:**")
cols = st.columns(len(EXAMPLES))
clicked_example = None
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    if col.button(f"#{i+1}", key=f"ex_{i}", help=ex, use_container_width=True):
        clicked_example = ex

if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if clicked_example:
    st.session_state.input_text = clicked_example

user_input = st.text_area(
    label="Teks ulasan",
    value=st.session_state.input_text,
    placeholder="Ketik ulasan aplikasi Dana di sini...",
    height=130,
    label_visibility="collapsed",
    key="main_input",
)

char_count = len(user_input)
st.caption(f"{char_count} karakter")

predict_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


if predict_btn:
    if not user_input.strip():
        st.warning("⚠️ Teks ulasan tidak boleh kosong.")
    else:
        with st.spinner("Menganalisis..."):
            clean = preprocess(user_input, res["slang"])
            if not clean.strip():
                clean = user_input
            emb   = res["emb"].encode([clean], normalize_embeddings=True)
            probs = res["model"].predict(emb, verbose=0)[0]
            idx   = int(np.argmax(probs))
            label = res["le"].inverse_transform([idx])[0]
            conf  = float(probs[idx])
            prob_dict = dict(zip(res["le"].classes_, probs.tolist()))

        c = CFG[label]


        st.markdown(
            f'<div class="result-card" style="background:{c["gradient"]};'
            f'border:1.5px solid {c["color"]}44;padding-bottom:12px">',
            unsafe_allow_html=True,
        )

  
        st.markdown(
            f'<div class="sentiment-badge" style="background:{c["color"]}18;'
            f'color:{c["color"]};border:2px solid {c["color"]}44;">'
            f'<span style="font-size:1.5rem">{c["icon"]}</span>'
            f'<span>{c["label"].upper()}</span></div>',
            unsafe_allow_html=True,
        )

    
        st.markdown(
            f'<div class="conf-label">Confidence</div>'
            f'<div class="conf-value" style="color:{c["color"]}">{conf*100:.1f}%</div>',
            unsafe_allow_html=True,
        )


        st.markdown(
            '<div class="conf-label" style="margin:16px 0 8px">Distribusi Probabilitas</div>',
            unsafe_allow_html=True,
        )
        for cls, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
            cc  = CFG[cls]
            pct = prob * 100
            st.markdown(
                f'<div class="bar-wrap">'
                f'<div class="bar-label">'
                f'<span>{cc["icon"]} {cc["label"]}</span>'
                f'<span style="color:{cc["color"]}">{pct:.1f}%</span>'
                f'</div>'
                f'<div class="bar-track">'
                f'<div class="bar-fill" style="width:{pct:.2f}%;background:{cc["bar"]}"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🔎 Detail preprocessing"):
            col1, col2 = st.columns(2)
            col1.markdown("**Teks Asli**")
            col1.info(user_input)
            col2.markdown("**Setelah Preprocessing**")
            col2.success(clean)

st.markdown("""
<div class="footer">
    NoLimit Data Science Test &nbsp;·&nbsp; Yogi Dharma &nbsp;·&nbsp;
    Model: SentenceTransformer + Keras
</div>
""", unsafe_allow_html=True)

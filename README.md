# DANA Review Intelligence System
## Multilingual Sentiment Classification & Semantic Similarity Engine

> **NoLimit Indonesia — Data Scientist Hiring Assessment**  
> Submitted by: Yogi Dharma | Mathematics, Universitas Pendidikan Indonesia

---

## Problem Statement

DANA, one of Indonesia's leading digital wallets with 140M+ registered users, receives thousands of Google Play Store reviews daily. Manual sentiment triaging is operationally infeasible. This system provides:

- **Automated 3-class sentiment classification** (POSITIVE / NEGATIVE / NEUTRAL) on Indonesian user reviews
- **Semantic similarity retrieval** to surface contextually similar complaints or praises in real time
- A production-ready Streamlit interface ready for internal analyst tooling

The business value is direct: negative feedback clusters can be surfaced immediately for customer experience teams, while positive clusters inform marketing messaging.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| Source | [DANA App Sentiment Review — Kaggle](https://www.kaggle.com/datasets/alexmariosimanjuntak/dana-app-sentiment-review-on-playstore-indonesia) |
| License | Community Data License Agreement (CDLA-Sharing-1.0) |
| Size | 50,000 reviews |
| Language | Bahasa Indonesia (colloquial, slang-heavy) |
| Classes | POSITIVE (26,555) · NEGATIVE (17,073) · NEUTRAL (6,372) |
| Class Imbalance | ~4.2:2.7:1 — addressed via `compute_class_weight` |

**Why this dataset**: Google Play reviews represent real, unfiltered user voice with high linguistic diversity — abbreviations, emojis, alay (Indonesian internet slang), and code-switching. This makes it a strong stress test for multilingual NLP pipelines.

---

## Data Preprocessing Pipeline

```
Raw Review Text
      │
      ▼
[1] Case Folding           → lowercase normalization
      │
      ▼
[2] Text Cleaning          → strip URLs, mentions, hashtags, numbers,
                             punctuation, emoji demojize + strip,
                             repeated character normalization
      │
      ▼
[3] Slang Normalization    → kamus_alay.csv dictionary lookup
                             (tokenize → replace alay → rejoin)
      │
      ▼
Clean Text (text_final)
```

**Key preprocessing decisions:**
- `emoji.demojize()` followed by regex removal converts "😊" → `:smiling_face:` → ` ` — preserves sentiment signal at tokenization level before removal
- Repeated character normalization (`r"(.)\1{2,}"`) handles informal elongation ("bagusssss" → "bagus")
- The `kamus_alay` dictionary covers ~4,500 informal Indonesian abbreviations, crucial for sub-word normalization before embedding

---

## Model Architecture

### Embedding Layer: `intfloat/multilingual-e5-large`

| Property | Value |
|----------|-------|
| Model | `intfloat/multilingual-e5-large` |
| Embedding Dimension | 1024 |
| Languages | 100+ (strong Indonesian support) |
| Parameters | ~560M |
| Max Sequence | 512 tokens |

**Why multilingual-e5-large over alternatives:**

| Alternative | Why Not Chosen |
|-------------|----------------|
| `bert-base-multilingual-cased` | Lower semantic coherence; token-level not sentence-level |
| `paraphrase-multilingual-mpnet-base-v2` | English-biased training, weaker on Indonesian colloquial |
| `indobert-base-p1` | Indonesian-only; no cross-lingual transfer for mixed queries |
| `multilingual-e5-base` | 768-dim vs 1024-dim; measurably lower retrieval quality on Indonesian |

**⚠️ Critical usage note:** `intfloat/multilingual-e5` requires task prefixes:
- Queries / inference: `"query: {text}"`
- Corpus / training passages: `"passage: {text}"`

Omitting these prefixes degrades embedding quality significantly.

### Classification Head

```
Input (1024-dim embedding)
    │
    Dense(256, ReLU) → BatchNorm → Dropout(0.3)
    │
    Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    │
    Dense(64, ReLU)
    │
    Dense(3, Softmax)
    │
Output: [P(POSITIVE), P(NEGATIVE), P(NEUTRAL)]
```

- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Sparse Categorical Crossentropy
- **Regularization**: BatchNorm + Dropout prevents overfitting on imbalanced classes
- **Class weights**: Computed via `sklearn.utils.class_weight.compute_class_weight` — upweights NEUTRAL class (~3.8× relative)
- **EarlyStopping**: patience=3, monitors val_loss, restores best weights

### Similarity Retrieval: KNN (Cosine)

```python
NearestNeighbors(n_neighbors=5, metric='cosine')
```

Cosine distance is appropriate here because embedding magnitude is not semantically meaningful — only directional similarity matters. At 50K training samples, brute-force KNN is tractable (~0.5s query time). For >500K samples, replace with `faiss.IndexFlatIP` with normalized vectors.

---

## Full System Architecture

```
User Input Text
        │
        ▼
  ┌─────────────────────────────┐
  │   Preprocessing Pipeline    │
  │  casefolding → cleaning     │
  │  → slang normalization      │
  └─────────────┬───────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │  SentenceTransformer Encode │
  │  intfloat/multilingual-e5   │
  │  prefix: "query: {text}"   │
  │  output: float32[1024]      │
  └──────────┬──────────────────┘
             │
     ┌───────┴──────────┐
     │                  │
     ▼                  ▼
┌──────────┐    ┌─────────────────┐
│  Keras   │    │  NearestNeighbors│
│ Classifier│   │  (cosine, k=5)  │
│ (3-class)│    │  on train embeds │
└────┬─────┘    └───────┬─────────┘
     │                  │
     ▼                  ▼
Sentiment Label    Top-5 Similar Reviews
(+ confidence)     (+ distances)
```

---

## Evaluation Strategy

| Metric | Rationale |
|--------|-----------|
| Accuracy | Baseline; misleading under imbalance |
| Macro F1 | Equal weight across all 3 classes — primary metric |
| Per-class Precision/Recall | Diagnose NEUTRAL class degradation |
| Confusion Matrix | Identify POSITIVE↔NEUTRAL confusion patterns |

Expected targets for production acceptance: Macro F1 ≥ 0.82, NEUTRAL Recall ≥ 0.70.

---

## Error Analysis

Common failure modes on this dataset:

1. **NEUTRAL/POSITIVE confusion**: Short positive reviews ("bagus", "oke") sometimes classified as NEUTRAL due to low lexical content
2. **Sarcasm**: "Mantap, error mulu" — surface positive, semantically negative
3. **Mixed sentiment**: Reviews that praise one feature, criticize another

---

## Sample Outputs

```
Input:  "aplikasi sangat membantu dan mudah digunakan"
Output: POSITIVE (confidence: 0.94)

Input:  "aplikasi dana sering error saat transfer"
Similar: ["dana error terus pas mau transfer", "gagal terus transfer nya", ...]

Input:  "biasa aja sih, lumayan"
Output: NEUTRAL (confidence: 0.71)
```

---

## Limitations & Future Work

| Limitation | Proposed Improvement |
|------------|---------------------|
| Static slang dictionary | Replace with CharacterBERT or BPE-based normalization |
| No temporal analysis | Add time-series sentiment trend visualization |
| KNN at scale | Migrate to FAISS HNSW index for sub-linear retrieval |
| No aspect extraction | Add aspect-based sentiment (ABSA) for feature-level insights |
| Monolingual corpus | Fine-tune e5 on Indonesian domain data for domain adaptation |

---

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/yogidharma21/dana-sentiment-analysis
cd dana-sentiment-analysis
pip install -r requirements.txt

# 2. Download dataset (requires Kaggle credentials)
python scripts/download_data.py

# 3. Run full pipeline (train + save artifacts)
jupyter nbconvert --to notebook --execute notebook.ipynb

# 4. Launch Streamlit app
streamlit run app.py
```

---

## Repository Structure

```
dana-sentiment-analysis/
├── app.py                        # Streamlit production app
├── notebook.ipynb                # Full pipeline notebook
├── requirements.txt
├── README.md
├── artifacts/
│   ├── sentiment_model.keras     # Trained Keras model
│   ├── label_encoder.pkl
│   ├── nearest_neighbors.pkl
│   └── reviews.pkl
├── data/
│   └── Clean_DataSet_DANA.csv    # Preprocessed dataset
├── scripts/
│   ├── download_data.py
│   └── preprocess.py
├── docs/
│   ├── system_flowchart.png
│   ├── system_flowchart.pdf
│   └── technical_report.pdf
└── tests/
    └── test_pipeline.py
```

---

## Requirements

```
sentence-transformers>=2.7.0
tensorflow>=2.15.0
scikit-learn>=1.4.0
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
nltk>=3.8.0
emoji>=2.10.0
joblib>=1.3.0
faiss-cpu>=1.8.0
matplotlib>=3.8.0
seaborn>=0.13.0
wordcloud>=1.9.0
kagglehub>=0.2.0
```

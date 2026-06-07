# DANA Sentiment Analysis
## Sentiment Classification & Semantic Similarity Engine

> **NoLimit Indonesia — Data Scientist Hiring Test**  
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
| Size | 50,000 reviews |
| Language | Bahasa Indonesia (colloquial, slang-heavy) |
| Classes | POSITIVE (26,555) · NEGATIVE (17,073) · NEUTRAL (6,372) |

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

### Embedding Layer: "paraphrase-multilingual-MiniLM-L12-v2"

| Property | Value |
|----------|-------|
| Model | "paraphrase-multilingual-MiniLM-L12-v2" |
| Embedding Dimension | 384 |
| Languages | 50+ (strong Indonesian support) |
| Parameters | ~118M |
| Max Sequence | 256 tokens |

### Similarity Retrieval: KNN (Cosine)

```python
NearestNeighbors(n_neighbors=5, metric='cosine')
```
## Sample Outputs

```
Input:  "aplikasi sangat membantu dan mudah digunakan"
Output: POSITIVE (confidence: 0.94)

Input:  "aplikasi dana sering error saat transfer"
Similar: ["dana error terus pas mau transfer", "gagal terus transfer nya", ...]

Input:  "biasa aja sih, lumayan"
Output: NEUTRAL (confidence: 0.71)
```
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

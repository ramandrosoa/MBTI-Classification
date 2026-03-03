# 🧠 MBTI Personality Predictor

This notebook covers the full pipeline from **exploratory analysis to prediction** — using textual data to classify Myers-Briggs Type Indicator (MBTI) personality types with classical machine learning.

🚀 **Live Demo:** https://huggingface.co/spaces/ramandrosoa/mbti-predictor

---

## 🗂️ Overview

Rather than predicting all 16 MBTI types at once, the problem is decomposed into **four independent binary classifiers**, one per dimension:

| Dimension | Class 1 | Class 0 |
|-----------|---------|---------|
| IE | Introvert (I) | Extrovert (E) |
| NS | Intuitive (N) | Sensing (S) |
| TF | Thinking (T) | Feeling (F) |
| JP | Judging (J) | Perceiving (P) |

---

## 🔬 Pipeline

### 📊 1. Exploratory Analysis
- Examined post counts per MBTI type
- Dataset was found to be **imbalanced** across types

### 🧹 2. Preprocessing
- **Tokenization** — split text into individual words/tokens
- **Stopword removal** — remove low-meaning words (e.g. "and", "the")
- **Lemmatization** — reduce words to base form (e.g. "running" → "run")
- 🔗 Links are stripped before any text analysis

### ⚙️ 3. Feature Engineering

| Feature | Description |
|---------|-------------|
| 📝 Post Length | Word count per post — captures conciseness of writing style |
| 📖 Readability | Gunning Fog Index — measures complexity via sentence length and word difficulty |
| 🏷️ NER | Named Entity Recognition — captures topical focus differences across types |
| 💬 Sentiment | Polarity scores — analyzes emotional tone of posts |
| 🔢 Vectorization | TF-IDF or word embeddings — converts text to numerical representations |

### ⚖️ 4. Resampling

The dataset is resampled to match the **real-world MBTI population distribution**:
- 📉 **Undersampling** applied to overrepresented types
- 📈 **Oversampling** applied to underrepresented types

### 🏋️ 5. Training

Three classifiers were evaluated per dimension using **grid search cross-validation**:
- Logistic Regression
- K-Nearest Neighbors
- Random Forest

The best-performing model per dimension was selected.

### 🚀 6. Inference and Deployment
- 🔮 Predictions made on new raw text input
- ⚡ REST API built with **FastAPI**
- 🤗 Model hosted on **Hugging Face**

---

- ## 🛠️ Tech Stack

- 🐼 **pandas / numpy** — data manipulation and numerical computing
- 📊 **matplotlib / seaborn** — visualization
- 📚 **nltk / spaCy** — tokenization, lemmatization, sentiment analysis (VADER), stopword removal
- 🔢 **scikit-learn** — TF-IDF vectorization, modeling, grid search, evaluation
- ⚖️ **imbalanced-learn** — SMOTE oversampling and random undersampling
- 🚀 **XGBoost** — gradient boosted classifier (evaluated alongside sklearn models)
- ⚡ **FastAPI** — API and interface
- 🤗 **Hugging Face Spaces** — deployment
---

## 📚 References

-  [Predicting Myers-Briggs Type Indicator with Text Classification](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6839354.pdf)



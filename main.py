import os
import pickle
import requests
import nltk
import pandas as pd

# Download NLTK data before importing preprocessing
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from preprocessing import test_prep

MODEL_PATH = "model_artifacts.pkl"
MODEL_URL = "https://huggingface.co/ramandrosoa/mbti-predictor-artifacts/resolve/main/model_artifacts.pkl"

def download_model():
    # Re-download if file doesn't exist or is smaller than 100MB
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100 * 1024 * 1024:
        print("Downloading model from Hugging Face...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model downloaded. Size: {os.path.getsize(MODEL_PATH)} bytes")
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise
    else:
        print(f"Model already exists. Size: {os.path.getsize(MODEL_PATH)} bytes")

download_model()

with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

vectorizer  = artifacts['vectorizer']
svd         = artifacts['svd']
scaler      = artifacts['scaler']
best_models = artifacts['best_models']

class TextInput(BaseModel): 
    text: str

def predict_type(text): 
    df = pd.DataFrame({"posts": [text]})
    preprocessed_text = test_prep(df, vectorizer, svd, scaler)

    model_map = {
        'IE': ('LogisticRegression',     'I', 'E'), 
        'NS': ('RandomForestClassifier', 'N', 'S'), 
        'TF': ('LogisticRegression',     'T', 'F'),
        'JP': ('LogisticRegression',     'J', 'P'),
    }

    results = {}
    for dim, (clf_name, pos_label, neg_label) in model_map.items(): 
        model = best_models[dim][clf_name][0]
        pred       = model.predict(preprocessed_text)[0]
        class_pred = pos_label if pred == 1 else neg_label
        probs      = model.predict_proba(preprocessed_text)[0]
        prob_pos   = round(probs[1] * 100, 2)
        prob_neg   = round(probs[0] * 100, 2)

        results[dim] = {
            'predicted': class_pred, 
            'probabilities': {pos_label: prob_pos, neg_label: prob_neg}
        }

    mbti_type = "".join(v['predicted'] for v in results.values())
    return mbti_type, results

app = FastAPI()

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root(): 
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: TextInput): 
    mbti_type, details = predict_type(payload.text)
    return {"mbti_type": mbti_type, "details": details}




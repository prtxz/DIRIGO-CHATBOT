# app/model_api.py
"""
Minimal model_api.py scaffold.
- Downloads model file(s) from Google Drive on first run (using gdown) if not present.
- Exposes:
    - load_model()
    - generate_response(prompt) -> str
    - generate_response_stream(prompt) -> generator of strings (for SSE)
Replace MODEL_DOWNLOADS with your Drive file IDs and adjust load_model() to your real model loading.
"""

import os
import time
from typing import Generator
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG: update these with your real filenames and Google Drive file IDs ---
# Example: {"app/models/model.pt": "GOOGLE_DRIVE_FILE_ID"}
MODEL_DOWNLOADS = {
    # "app/models/model.pt": "1NgCEe0BQCCmmGvXLV-j05JpA-SKuOoMP",
    # "app/models/tokenizer.json": "ANOTHER_DRIVE_FILE_ID_IF_NEEDED"
}

# Ensure models folder exists
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Simple downloader using gdown (already installed per you)
def _download_from_drive(dest_path: str, file_id: str):
    try:
        import gdown
    except Exception as e:
        raise RuntimeError("gdown is required to download model files. Install with `pip install gdown`.") from e

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

# Lazy-loaded global placeholder
MODEL = None

def load_model():
    """
    Load (or train) the lightweight ML models and dataframes used by DIRIGO.
    - Loads professors.csv and labs.csv from BASE_MODELS_DIR
    - Builds a simple CountVectorizer from the notebook's training_data
    - Trains the small set of classifiers and keeps them in trained_models
    """
    global MODEL
    # If already loaded, return a simple marker object
    if MODEL is not None:
        return MODEL

    # ---------- load CSVs ----------
    prof_path = os.path.join(BASE_MODELS_DIR, "professors.csv")
    labs_path = os.path.join(BASE_MODELS_DIR, "labs.csv")
    if not os.path.exists(prof_path) or not os.path.exists(labs_path):
        raise FileNotFoundError(
            f"professors.csv or labs.csv not found in {BASE_MODELS_DIR}. "
            "Place them there or update the paths."
        )

    professors_df = pd.read_csv(prof_path)
    labs_df = pd.read_csv(labs_path)

    # ---------- training data from your notebook ----------
    training_data = [
        ("Where can I meet the professor?", "professor"),
        ("What is the room number for a professor?", "professor"),
        ("Where is the lab for Physics?", "labs"),
        ("Which subject is Galileo Lab related to?", "labs"),
        ("Locate the professor's office.", "professor"),
        ("Where is the Robotics Lab?", "labs"),
        ("Tell me about Tesla Lab.", "labs"),
        ("Find Analog Circuits lab.", "labs"),
    ]

    queries, labels = zip(*training_data)

    # Use CountVectorizer (same as notebook)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([q.lower() for q in queries])
    y = labels

    # split and train small models (mirrors notebook)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "SVM": SVC(kernel="linear"),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    trained_models = {}
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[model_name] = acc
        except Exception:
            # ignore training errors for any model and continue
            results[model_name] = 0.0

    # pick best model name (fallback to Naive Bayes if tie)
    best_model_name = max(results, key=results.get) if results else "Naive Bayes"
    if best_model_name not in trained_models:
        # ensure at least Naive Bayes exists
        nb = MultinomialNB()
        nb.fit(X, y)
        trained_models["Naive Bayes"] = nb
        best_model_name = "Naive Bayes"

    # Save into MODEL dict for global access
    MODEL = {
        "vectorizer": vectorizer,
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "professors_df": professors_df,
        "labs_df": labs_df,
    }

    print(f"Loaded DIRIGO model. Best model: {best_model_name}")
    return MODEL



def generate_response(prompt: str) -> str:
    """
    Uses the trained models and dataframes to return a single string response.
    """
    data = load_model()
    vectorizer = data["vectorizer"]
    trained_models = data["trained_models"]
    professors_df = data["professors_df"]
    labs_df = data["labs_df"]

    user_input = str(prompt).strip().lower()
    user_vector = vectorizer.transform([user_input])

    # Use the selected best model (but fallback to Naive Bayes classifier)
    best_model = trained_models.get(data.get("best_model_name")) or trained_models.get("Naive Bayes")

    try:
        query_type = best_model.predict(user_vector)[0]
    except Exception:
        query_type = "unknown"

    response = ""

    if query_type == "professor":
        for _, row in professors_df.iterrows():
            if str(row.get("name", "")).lower() in user_input:
                dept = row.get("department", "")
                floor = row.get("floor", "")
                room = row.get("room", "")
                name = row.get("name", "")
                if pd.notna(dept) and dept != "":
                    response = f"You can meet {name} in room {room} on the {floor} floor of the {dept} department."
                else:
                    response = f"You can meet {name} in room {room} on the {floor} floor."
                break

    elif query_type == "labs":
        found_lab = False
        for _, row in labs_df.iterrows():
            lab_name = str(row.get("lab_name", "")).lower()
            subject = str(row.get("subject", "")).lower()
            if lab_name in user_input or subject in user_input:
                found_lab = True
                dept = row.get("department", "")
                floor = row.get("floor", "")
                room = row.get("room", "")
                lab = row.get("lab_name", "")
                if pd.notna(dept) and dept != "":
                    response = f"The {lab} lab is located in room {room} on the {floor} floor of the {dept} department."
                else:
                    response = f"The {lab} lab is located in room {room} on the {floor} floor."
                break
        if not found_lab:
            response = "Sorry, I couldn't find the lab you're looking for."

    if not response:
        response = "Sorry, I couldn't find the information you're looking for."

    return response

def generate_response_stream(prompt: str) -> Generator[str, None, None]:
    """
    Simple streaming wrapper: yields small chunks (words) from the
    full generate_response() output to emulate token streaming.
    """
    full = generate_response(prompt)
    for chunk in full.split():
        yield chunk + " "
        time.sleep(0.02)
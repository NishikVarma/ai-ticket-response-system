"""ML step: TF-IDF + LogisticRegression predicts a one-line solution from subject + description.

Train:  uv run python ml.py
"""
import csv
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATA = "data/tickets.csv"
MODEL_PATH = "model.joblib"
MIN_CONFIDENCE = float(os.environ.get("ML_MIN_CONFIDENCE", "0.25"))


def _text(subject, description):
    return f"{subject}. {description}"


def train():
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    X = [_text(r["subject"], r["description"]) for r in rows]
    y = [r["solution"] for r in rows]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    make = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                 LogisticRegression(C=10, max_iter=1000))
    held_out = make().fit(Xtr, ytr)
    print(f"held-out accuracy: {accuracy_score(yte, held_out.predict(Xte)):.2f} ({len(yte)} samples)")
    model = make().fit(X, y)  # final model uses all data
    joblib.dump(model, MODEL_PATH)
    print(f"saved {MODEL_PATH} ({len(set(y))} solution classes, {len(X)} tickets)")


_model = None


def predict(subject, description):
    """Return (solution, confidence). Raises FileNotFoundError if the model was not trained."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    text = _text(subject, description)
    probs = _model.predict_proba([text])[0]
    i = probs.argmax()
    return _model.classes_[i], float(probs[i])


if __name__ == "__main__":
    train()

"""Fonctions utilitaires partagées entre les modules."""

import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def save_model(model, filename: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(filename: str):
    path = MODELS_DIR / filename
    return joblib.load(path)

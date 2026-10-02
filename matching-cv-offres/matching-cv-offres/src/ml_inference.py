"""
Inférence avec le modèle ML entraîné (src/train_model.py) sur une nouvelle
paire (CV, offre) — à utiliser dans l'app Streamlit ou pour l'évaluation
comparative, en complément (pas en remplacement) du score par embeddings.
"""

import numpy as np
from src.utils import load_model
from src.feature_extraction import get_embeddings
from src.pair_features import build_pair_features, ML_FEATURE_NAMES


def ml_predict_proba(cv_raw_text: str, job_raw_text: str,
                      cv_clean_text: str, job_clean_text: str) -> float:
    """
    Renvoie la probabilité de match (0-1) prédite par le modèle ML entraîné.
    Nécessite d'avoir lancé `python -m src.train_model` au moins une fois.

    Important : on réutilise le vectorizer TF-IDF entraîné sur tout le
    corpus (sauvegardé pendant l'entraînement), plutôt que d'en re-fitter un
    nouveau sur seulement ces 2 textes. Un TF-IDF ajusté sur 2 documents
    produit un espace vectoriel totalement différent de celui vu à
    l'entraînement, ce qui rendrait la prédiction incohérente.
    """
    model = load_model("best_matching_model.joblib")
    scaler = load_model("scaler.joblib")
    model_name = load_model("best_model_name.joblib")
    tfidf_vectorizer = load_model("tfidf_vectorizer.joblib")

    cv_tfidf = tfidf_vectorizer.transform([cv_clean_text])
    job_tfidf = tfidf_vectorizer.transform([job_clean_text])

    embeddings = get_embeddings([cv_clean_text, job_clean_text])
    cv_emb, job_emb = embeddings[0], embeddings[1]

    features = build_pair_features(
        cv_raw_text, job_raw_text, cv_tfidf, job_tfidf, cv_emb, job_emb
    )
    # On ne garde que tfidf_cosine et embedding_cosine (indices 0 et 1),
    # exactement les features utilisées à l'entraînement (voir ML_FEATURE_NAMES).
    ml_features = features[:2]
    X = np.array(ml_features).reshape(1, -1)

    if model_name == "Logistic Regression":
        X = scaler.transform(X)

    proba = model.predict_proba(X)[0][1]
    return round(float(proba) * 100, 1)

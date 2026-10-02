"""
Construction de features numériques pour une paire (CV, offre), utilisées
comme entrée du modèle de Machine Learning supervisé (méthode 3 du PDF).

Features :
- tfidf_cosine       : similarité cosinus TF-IDF entre le CV et l'offre
- embedding_cosine   : similarité cosinus des embeddings sémantiques
- n_matched_skills   : nombre de compétences en commun
- n_missing_skills   : nombre de compétences demandées mais absentes du CV
- matched_ratio      : proportion des compétences de l'offre couvertes par le CV
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_extraction import extract_skills

FEATURE_NAMES = [
    "tfidf_cosine", "embedding_cosine",
    "n_matched_skills", "n_missing_skills", "matched_ratio",
]

# Features réellement données au modèle ML (train_model.py) : uniquement la
# similarité textuelle, JAMAIS les colonnes dérivées des compétences (qui ont
# servi à créer le label). Voir weak_label() pour l'explication du choix.
ML_FEATURE_NAMES = ["tfidf_cosine", "embedding_cosine"]


def build_pair_features(cv_raw_text: str, job_raw_text: str,
                         cv_tfidf_vec, job_tfidf_vec,
                         cv_embedding: np.ndarray, job_embedding: np.ndarray) -> list[float]:
    """Construit le vecteur de features pour UNE paire (CV, offre)."""
    tfidf_sim = float(cosine_similarity(cv_tfidf_vec, job_tfidf_vec)[0][0])
    emb_sim = float(cosine_similarity(
        cv_embedding.reshape(1, -1), job_embedding.reshape(1, -1)
    )[0][0])

    cv_skills = set(extract_skills(cv_raw_text))
    job_skills = set(extract_skills(job_raw_text))

    matched = cv_skills & job_skills
    missing = job_skills - cv_skills
    matched_ratio = len(matched) / len(job_skills) if job_skills else 0.0

    return [tfidf_sim, emb_sim, len(matched), len(missing), matched_ratio]


def weak_label(matched_ratio: float, skill_threshold: float = 0.5) -> int:
    """
    Règle d'étiquetage faible (weak supervision), utilisée en l'absence de
    vraies annotations humaines match/no-match dans les datasets Kaggle.

    Le label se base UNIQUEMENT sur le chevauchement réel de compétences
    (matched_ratio), jamais sur la similarité sémantique (embeddings/TF-IDF).
    C'est volontaire : si le label utilisait aussi la similarité sémantique,
    le modèle entraîné sur ces mêmes similarités "tricherait" en retrouvant
    la règle de seuil au lieu d'apprendre une vraie relation (fuite de
    données / data leakage) — ce qui produirait une accuracy artificiellement
    proche de 100%, sans valeur réelle.
    """
    return int(matched_ratio >= skill_threshold)

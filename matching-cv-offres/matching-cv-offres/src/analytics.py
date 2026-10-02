"""
Statistiques descriptives du corpus (page Analytics de l'application).

Répond aux six indicateurs demandés au cahier des charges (section 13, page 4) :
nombre de CV analysés, nombre d'offres, compétences les plus fréquentes,
technologies les plus demandées, distribution des scores, performance des
modèles (cette dernière étant produite par src/train_model.py).

Note de performance : le comptage des compétences s'effectue sur le TEXTE
BRUT, sans lemmatisation. C'est à la fois plus rapide (aucun appel spaCy) et
plus correct — la lemmatisation casse les expressions composées du type
"machine learning" (voir src/preprocessing.py).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_extraction import SKILL_KEYWORDS, get_embeddings


# ---------------------------------------------------------------------------
# Comptage des compétences
# ---------------------------------------------------------------------------

def skill_frequency(texts) -> pd.Series:
    """
    Compte, pour chaque compétence du référentiel, le nombre de documents qui
    la mentionnent. Renvoie une série triée par fréquence décroissante.

    On compte les DOCUMENTS et non les occurrences : un CV répétant dix fois
    "Python" ne pèse pas plus qu'un CV le mentionnant une fois.
    """
    counts = {skill: 0 for skill in SKILL_KEYWORDS}
    for text in texts:
        if not isinstance(text, str):
            continue
        norm = text.lower().replace("-", " ")
        for skill in SKILL_KEYWORDS:
            if skill.replace("-", " ") in norm:
                counts[skill] += 1

    serie = pd.Series(counts, name="documents")
    return serie[serie > 0].sort_values(ascending=False)


def skill_coverage(texts) -> dict:
    """Part des documents mentionnant au moins une compétence du référentiel."""
    total = len(texts)
    with_skill = 0
    for text in texts:
        if not isinstance(text, str):
            continue
        norm = text.lower().replace("-", " ")
        if any(s.replace("-", " ") in norm for s in SKILL_KEYWORDS):
            with_skill += 1
    return {
        "total": total,
        "avec_competences": with_skill,
        "taux": round(100 * with_skill / total, 1) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Distribution des scores de compatibilité
# ---------------------------------------------------------------------------

def score_distribution(cv_clean_texts, job_clean_texts,
                        n_cv: int = 30, n_jobs: int = 150, seed: int = 42) -> np.ndarray:
    """
    Calcule la distribution des scores de compatibilité sur un échantillon
    aléatoire de paires (CV, offre).

    L'échantillonnage est nécessaire : le produit complet du corpus
    représenterait plusieurs millions de paires, et l'encodage de tous les
    documents prendrait plusieurs minutes. Un échantillon de quelques milliers
    de paires suffit largement à caractériser la forme de la distribution.
    """
    rng = np.random.default_rng(seed)

    cv_texts = list(cv_clean_texts)
    job_texts = list(job_clean_texts)

    cv_idx = rng.choice(len(cv_texts), size=min(n_cv, len(cv_texts)), replace=False)
    job_idx = rng.choice(len(job_texts), size=min(n_jobs, len(job_texts)), replace=False)

    cv_emb = get_embeddings([cv_texts[i] for i in cv_idx])
    job_emb = get_embeddings([job_texts[j] for j in job_idx])

    sims = cosine_similarity(cv_emb, job_emb) * 100
    return sims.flatten()


def distribution_summary(scores: np.ndarray) -> pd.DataFrame:
    """Statistiques descriptives de la distribution des scores."""
    return pd.DataFrame({
        "Indicateur": ["Minimum", "1er quartile", "Médiane", "Moyenne",
                        "3e quartile", "Maximum", "Écart-type"],
        "Score (%)": [
            round(float(np.min(scores)), 1),
            round(float(np.percentile(scores, 25)), 1),
            round(float(np.median(scores)), 1),
            round(float(np.mean(scores)), 1),
            round(float(np.percentile(scores, 75)), 1),
            round(float(np.max(scores)), 1),
            round(float(np.std(scores)), 1),
        ],
    })


# ---------------------------------------------------------------------------
# Comparaison CV / offres : où sont les écarts de compétences ?
# ---------------------------------------------------------------------------

def supply_demand_gap(cv_freq: pd.Series, job_freq: pd.Series,
                       n_cv: int, n_jobs: int, top_n: int = 15) -> pd.DataFrame:
    """
    Compare la fréquence relative de chaque compétence côté candidats (offre
    de compétences) et côté employeurs (demande). Un écart négatif signale une
    compétence plus demandée que disponible — l'information la plus utile de
    toute la page.
    """
    rows = []
    for skill in set(cv_freq.index) | set(job_freq.index):
        pct_cv = 100 * cv_freq.get(skill, 0) / n_cv if n_cv else 0
        pct_job = 100 * job_freq.get(skill, 0) / n_jobs if n_jobs else 0
        rows.append({
            "Compétence": skill,
            "% des CV": round(pct_cv, 1),
            "% des offres": round(pct_job, 1),
            "Écart": round(pct_cv - pct_job, 1),
        })

    df = pd.DataFrame(rows)
    df = df[(df["% des CV"] > 0) | (df["% des offres"] > 0)]
    return df.sort_values("Écart").head(top_n).reset_index(drop=True)

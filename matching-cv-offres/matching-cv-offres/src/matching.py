"""
Calcul du score de compatibilité CV / offre d'emploi.

Fournit :
- compute_similarity()  : similarité cosinus entre un CV et une offre (TF-IDF ou embeddings)
- match_cv_to_jobs()    : classe toutes les offres pour un CV donné (Top-K)
- explain_match()       : renvoie compétences correspondantes / manquantes
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_extraction import get_embeddings, extract_skills


def compute_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Similarité cosinus entre deux vecteurs (1D)."""
    a = vec_a.reshape(1, -1)
    b = vec_b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])


def explain_match(cv_text: str, job_text: str) -> dict:
    """Compétences en commun / manquantes entre un CV et une offre."""
    cv_skills = set(extract_skills(cv_text))
    job_skills = set(extract_skills(job_text))
    matched = sorted(cv_skills & job_skills)
    missing = sorted(job_skills - cv_skills)
    return {"matched_skills": matched, "missing_skills": missing}


def match_cv_to_jobs(cv_clean_text: str, jobs_df: pd.DataFrame, top_k: int = 5,
                      cv_raw_text: str | None = None) -> pd.DataFrame:
    """Classe les offres par compatibilité avec un CV et renvoie le top-K.

    cv_clean_text : texte lemmatisé, utilisé pour les plongements.
    cv_raw_text   : texte brut, utilisé pour l'extraction des compétences.
                    À défaut, cv_clean_text sert aux deux, au prix d'une
                    extraction moins fiable sur les expressions composées.
    jobs_df       : doit porter les colonnes « clean_text » et « text ».
    """
    cv_skill_text = cv_raw_text if cv_raw_text is not None else cv_clean_text

    cv_embedding = get_embeddings([cv_clean_text])[0]
    job_embeddings = get_embeddings(jobs_df["clean_text"].tolist())

    scores = cosine_similarity([cv_embedding], job_embeddings)[0]

    results = jobs_df.copy()
    results["compatibility_score"] = (scores * 100).round(1)
    results = results.sort_values("compatibility_score", ascending=False).head(top_k)

    explanations = results["text"].apply(lambda t: explain_match(cv_skill_text, t))
    results["matched_skills"] = explanations.apply(lambda e: e["matched_skills"])
    results["missing_skills"] = explanations.apply(lambda e: e["missing_skills"])

    return results.reset_index(drop=True)


if __name__ == "__main__":
    # Exemple rapide (à remplacer par de vraies données via data_loader.py)
    cv = "Python developer with Machine Learning, NLP and SQL experience"
    jobs = pd.DataFrame({
        "id": [1, 2],
        "text": [
            "AI Engineer needed: Python, Machine Learning, Docker, AWS",
            "Data Analyst position: SQL, Power BI, Tableau",
        ],
    })
    jobs["clean_text"] = jobs["text"]
    print(match_cv_to_jobs(cv, jobs, top_k=2))

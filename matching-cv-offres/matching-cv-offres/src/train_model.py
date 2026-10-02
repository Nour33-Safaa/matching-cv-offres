"""
Méthode 3 du PDF — Machine Learning supervisé.

Entraîne et compare Logistic Regression, Random Forest et XGBoost pour
prédire si un CV correspond à une offre, à partir de features numériques
(similarité TF-IDF, similarité embeddings, chevauchement de compétences).

Les vraies étiquettes match/no-match n'existant pas dans les datasets
Kaggle utilisés, un jeu d'entraînement est généré par weak supervision
(voir pair_features.weak_label). À documenter comme limite dans le rapport.

Lancer avec : python -m src.train_model
"""

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_loader import load_data
from src.preprocessing import clean_dataframe_fast
from src.feature_extraction import get_tfidf_vectors, get_embeddings
from src.pair_features import build_pair_features, weak_label, FEATURE_NAMES, ML_FEATURE_NAMES
from src.evaluate import evaluate_predictions, compare_methods
from src.utils import save_model

RANDOM_SEED = 42
N_JOBS_PER_CV = 6  # nb d'offres échantillonnées par CV pour construire les paires


def build_training_pairs(cv_df: pd.DataFrame, jobs_df: pd.DataFrame,
                          n_jobs_per_cv: int = N_JOBS_PER_CV):
    """
    Construit un ensemble de paires (CV, offre) avec features + label faible.

    Sélection des paires candidates basée sur le VRAI chevauchement de
    compétences (calcul matriciel vectorisé, rapide même sur tout le corpus) :
    pour chaque CV, on prend les offres avec le plus fort recouvrement de
    compétences (candidats positifs) + des offres aléatoires (candidats
    négatifs). Une sélection basée sur la similarité d'embeddings donnait
    très peu de vrais positifs (~4% des paires), d'où un déséquilibre de
    classes qui empêchait les modèles d'apprendre quoi que ce soit.

    Les CV sans aucune compétence technique reconnue (ex. profils RH) sont
    exclus : un tel CV ne pourra jamais atteindre le seuil de matched_ratio,
    et n'ajoute donc que du bruit à la classe négative.
    """
    from src.feature_extraction import SKILL_KEYWORDS

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("  Extraction vectorisée des compétences (CV et offres)...")

    def skill_matrix(texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(SKILL_KEYWORDS)), dtype=int)
        for i, text in enumerate(texts):
            norm = text.lower().replace("-", " ")
            for k, skill in enumerate(SKILL_KEYWORDS):
                if skill.replace("-", " ") in norm:
                    mat[i, k] = 1
        return mat

    cv_skills_mat = skill_matrix(cv_df["text"].tolist())
    job_skills_mat = skill_matrix(jobs_df["text"].tolist())

    # On ne garde que les CV et offres qui mentionnent au moins une compétence
    # de la liste — sinon matched_ratio est toujours 0, sans information utile.
    cv_keep = cv_skills_mat.sum(axis=1) > 0
    job_keep = job_skills_mat.sum(axis=1) > 0
    cv_df = cv_df[cv_keep].reset_index(drop=True)
    jobs_df = jobs_df[job_keep].reset_index(drop=True)
    cv_skills_mat = cv_skills_mat[cv_keep]
    job_skills_mat = job_skills_mat[job_keep]
    print(f"  {len(cv_df)} CV et {len(jobs_df)} offres avec compétences reconnues.")

    # Chevauchement de compétences pour TOUTES les paires (CV x offre) d'un coup.
    matched_counts = cv_skills_mat @ job_skills_mat.T          # (n_cv, n_job)
    job_totals = job_skills_mat.sum(axis=1)                     # (n_job,)
    matched_ratio_mat = matched_counts / np.maximum(job_totals, 1)

    print("  Calcul TF-IDF sur le corpus complet...")
    corpus = cv_df["clean_text"].tolist() + jobs_df["clean_text"].tolist()
    tfidf_matrix, tfidf_vectorizer = get_tfidf_vectors(corpus)
    cv_tfidf = tfidf_matrix[: len(cv_df)]
    job_tfidf = tfidf_matrix[len(cv_df):]

    print("  Calcul des embeddings sur le corpus complet...")
    cv_embeddings = get_embeddings(cv_df["clean_text"].tolist())
    job_embeddings = get_embeddings(jobs_df["clean_text"].tolist())

    print("  Construction des paires (CV, offre)...")
    n_jobs_total = len(jobs_df)
    half = n_jobs_per_cv // 2

    rows, labels = [], []
    for i in range(len(cv_df)):
        ratios = matched_ratio_mat[i]
        top_job_idx = np.argsort(ratios)[::-1][:half]                     # meilleurs recouvrements
        random_job_idx = np.random.choice(n_jobs_total, size=half, replace=False)
        candidate_idx = np.unique(np.concatenate([top_job_idx, random_job_idx]))

        for j in candidate_idx:
            features = build_pair_features(
                cv_df.iloc[i]["text"], jobs_df.iloc[j]["text"],
                cv_tfidf[i], job_tfidf[j],
                cv_embeddings[i], job_embeddings[j],
            )
            label = weak_label(matched_ratio=features[4])
            rows.append(features)
            labels.append(label)

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(labels, name="label")
    return X, y, tfidf_vectorizer


def train_and_compare(X: pd.DataFrame, y: pd.Series):
    X = X[ML_FEATURE_NAMES]  # on ne garde QUE les features non corrélées au label par construction
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Filet de sécurité : même avec un échantillonnage plus équilibré, la
    # classe positive reste probablement minoritaire. class_weight="balanced"
    # (et son équivalent scale_pos_weight pour XGBoost) pénalise davantage
    # les erreurs sur la classe minoritaire au lieu de la négliger.
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced"),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_SEED, scale_pos_weight=scale_pos_weight),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        results[name] = evaluate_predictions(y_test.tolist(), y_pred.tolist())
        trained_models[name] = model

    comparison_df = compare_methods(results)
    return comparison_df, trained_models, scaler


def main():
    print("Chargement des données...")
    cv_df, jobs_df = load_data()

    print("Nettoyage des textes...")
    cv_df = clean_dataframe_fast(cv_df, batch_size=32)
    jobs_df = clean_dataframe_fast(jobs_df, batch_size=32)

    print("Construction du jeu d'entraînement (paires CV/offre)...")
    X, y, tfidf_vectorizer = build_training_pairs(cv_df, jobs_df)
    print(f"  {len(X)} paires construites — répartition des labels :")
    print(y.value_counts())

    print("\nEntraînement et comparaison des modèles...")
    comparison_df, trained_models, scaler = train_and_compare(X, y)
    print("\nComparaison des modèles :")
    print(comparison_df)

    best_name = comparison_df["f1_score"].idxmax()
    best_model = trained_models[best_name]
    print(f"\nMeilleur modèle : {best_name}")

    save_model(best_model, "best_matching_model.joblib")
    save_model(scaler, "scaler.joblib")
    save_model(best_name, "best_model_name.joblib")
    save_model(tfidf_vectorizer, "tfidf_vectorizer.joblib")
    comparison_df.to_csv("models/model_comparison.csv")
    print("Modèle, scaler, vectorizer et comparatif sauvegardés dans models/")


if __name__ == "__main__":
    main()

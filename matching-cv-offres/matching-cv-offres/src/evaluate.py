"""
Évaluation et comparaison des méthodes de matching (TF-IDF, Sentence
Transformers, Machine Learning supervisé).

Nécessite un jeu de test annoté : pour chaque paire (CV, offre), un label
1 (pertinent) / 0 (non pertinent).
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_predictions(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


def compare_methods(results: dict[str, dict]) -> pd.DataFrame:
    """
    results = {
        "TF-IDF": {"accuracy": .., "precision": .., "recall": .., "f1_score": ..},
        "Sentence Transformers": {...},
        "Machine Learning": {...},
    }
    """
    return pd.DataFrame(results).T


if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    print(evaluate_predictions(y_true, y_pred))

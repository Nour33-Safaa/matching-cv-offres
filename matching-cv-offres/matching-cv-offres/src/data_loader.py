"""
Chargement des données brutes (CV et offres d'emploi) depuis data/raw.

Adapte les noms de colonnes selon le dataset Kaggle que tu utilises réellement
(ils varient d'un dataset à l'autre). Les deux fonctions ci-dessous renvoient
toujours un DataFrame avec des colonnes normalisées : "id" et "text".
"""

from pathlib import Path
import pandas as pd

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_resumes(filename: str = "Resume/Resume.csv") -> pd.DataFrame:
    """
    Charge le dataset de CV.
    Exemple : snehaanbhawal/resume-dataset -> colonnes ["ID", "Resume_str", "Category"]
    """
    path = DATA_RAW / filename
    df = pd.read_csv(path)

    # --- adapte ce mapping aux vraies colonnes de ton dataset ---
    rename_map = {"ID": "id", "Resume_str": "text", "Category": "category"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "id" not in df.columns:
        df["id"] = df.index

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return df[["id", "text"] + [c for c in ["category"] if c in df.columns]]


def load_job_offers(filename: str = "job_title_des.csv") -> pd.DataFrame:
    """
    Charge le dataset d'offres d'emploi.
    Exemple : kshitizregmi/jobs-and-job-description -> colonnes ["Job Title", "Job Description"]
    """
    path = DATA_RAW / filename
    df = pd.read_csv(path)

    # les exports CSV laissent une colonne d'index « Unnamed: 0 » sans intérêt
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
                 errors="ignore")

    rename_map = {"Job Title": "title", "Job Description": "text"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "id" not in df.columns:
        df["id"] = df.index

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return df[["id", "text"] + [c for c in ["title"] if c in df.columns]]


def save_processed(df: pd.DataFrame, filename: str) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / filename, index=False)


def load_data():
    """Raccourci utilisé par main.py pour charger CV + offres en une seule fois."""
    cv_df = load_resumes()
    jobs_df = load_job_offers()
    return cv_df, jobs_df


if __name__ == "__main__":
    cvs, jobs = load_data()
    print(f"CV chargés : {len(cvs)}")
    print(f"Offres chargées : {len(jobs)}")

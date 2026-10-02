"""
Point d'entrée du pipeline complet :
chargement → nettoyage → extraction des compétences → matching → top-5.

Lancer avec : python main.py
"""

from src.data_loader import load_data
from src.preprocessing import clean_dataframe_fast
from src.matching import match_cv_to_jobs


def main():
    print("Chargement des données...")
    cv_df, jobs_df = load_data()

    print(f"Nettoyage des textes ({len(cv_df)} CV, {len(jobs_df)} offres)...")
    # n_process reste à 1 : au-delà, Windows exige que l'appel soit protégé
    # par if __name__ == "__main__", ce qui est le cas ici mais pas partout.
    cv_df = clean_dataframe_fast(cv_df, batch_size=32)
    jobs_df = clean_dataframe_fast(jobs_df, batch_size=32)

    print("Catégories disponibles :")
    if "category" in cv_df.columns:
        print(cv_df["category"].value_counts().head(10))

    # un CV technique : les profils non techniques ne partagent aucune
    # compétence du référentiel et produisent des listes vides
    tech = cv_df[cv_df.get("category", "") == "INFORMATION-TECHNOLOGY"]
    ligne = tech.iloc[0] if len(tech) else cv_df.iloc[0]

    print("\nMatching en cours...")
    top = match_cv_to_jobs(
        ligne["clean_text"], jobs_df, top_k=5, cv_raw_text=ligne["text"],
    )

    print("\nTop 5 des offres recommandées :")
    print(top[["id", "compatibility_score", "matched_skills", "missing_skills"]])


if __name__ == "__main__":
    main()

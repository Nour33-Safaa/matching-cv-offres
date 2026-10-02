"""
Évaluation DeepEval du module d'explication (partie IA générative).

Version simple et séquentielle : pas d'asyncio, pas de parallélisme.
Chaque métrique est mesurée une par une, avec affichage au fur et à mesure.

Lancer :  python -m src.eval_genai
"""

import os

# IMPORTANT : doit être défini AVANT l'import de deepeval.
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "900"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Juge par défaut : modèle SANS raisonnement, sinon chaque appel dure >90s.
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "llama3.2:3b")

RESULTS_PATH = Path(__file__).resolve().parents[1] / "models" / "genai_evaluation.csv"

CASES = [
    {"job_title": "AI Engineer", "score": 84.0,
     "matched": ["python", "machine learning", "sql"], "missing": ["kubernetes", "aws"]},
    {"job_title": "Data Analyst", "score": 52.0,
     "matched": ["sql", "power bi"], "missing": ["python", "tableau", "spark"]},
    {"job_title": "DevOps Engineer", "score": 31.0,
     "matched": [], "missing": ["docker", "kubernetes", "aws", "linux", "git"]},
    {"job_title": "Backend Developer", "score": 76.0,
     "matched": ["python", "sql", "git", "docker"], "missing": []},
]


def build_context(case):
    """
    Contexte factuel transmis au juge, en UN SEUL bloc coherent.

    Ne pas decouper en fragments separes : DeepEval evalue chaque element de
    la liste independamment, et un fragment isole du type "Poste : X." ne peut
    appuyer aucune phrase de l'explication. Le decoupage produisait donc un
    score constant de 0.50, artefact de la structure et non mesure reelle.
    """
    matched = ", ".join(case["matched"]) if case["matched"] else "aucune"
    missing = ", ".join(case["missing"]) if case["missing"] else "aucune"
    return [
        f"Le poste evalue est : {case['job_title']}. "
        f"Le score de compatibilite calcule est de {case['score']}%. "
        f"Les competences presentes dans le CV et demandees par l'offre sont : {matched}. "
        f"Les competences demandees par l'offre et absentes du CV sont : {missing}."
    ]


def main():
    from deepeval.models import OllamaModel
    from deepeval.metrics import (FaithfulnessMetric, HallucinationMetric,
                                   AnswerRelevancyMetric)
    from deepeval.test_case import LLMTestCase

    from src.genai_explain import explain_score

    judge = OllamaModel(model=JUDGE_MODEL, base_url=OLLAMA_URL)
    print(f"Juge : {JUDGE_MODEL} (Ollama local)\n")

    # async_mode=False -> mesure sequentielle, une requete Ollama a la fois.
    # Faithfulness decoupe la sortie en affirmations et verifie chacune contre
    # le contexte : score gradue, bien plus informatif que Hallucination qui
    # raisonne au niveau du bloc de contexte entier.
    metrics = [
        FaithfulnessMetric(threshold=0.8, model=judge, async_mode=False),
        HallucinationMetric(threshold=0.3, model=judge, async_mode=False),
        AnswerRelevancyMetric(threshold=0.7, model=judge, async_mode=False),
    ]

    rows = []
    for i, case in enumerate(CASES, 1):
        print(f"[{i}/{len(CASES)}] {case['job_title']} ({case['score']}%)")

        output = explain_score(case["job_title"], case["score"],
                                case["matched"], case["missing"])

        tc = LLMTestCase(
            input=f"Explique la compatibilite avec le poste {case['job_title']}.",
            actual_output=output,
            context=build_context(case),
            retrieval_context=build_context(case),
        )

        for metric in metrics:
            try:
                metric.measure(tc)
                score, reason, ok = metric.score, metric.reason, metric.success
                print(f"    {metric.__class__.__name__:26s} {score:.2f}  "
                      f"{'OK' if ok else 'ECHEC'}")
            except Exception as e:
                score, reason, ok = None, f"erreur: {e}", False
                print(f"    {metric.__class__.__name__:26s} ERREUR  {e}")

            rows.append({
                "cas": case["job_title"],
                "metrique": metric.__class__.__name__,
                "score": round(score, 3) if score is not None else None,
                "seuil": metric.threshold,
                "reussi": ok,
                "justification": str(reason)[:300],
            })
        print()

    df = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    print("Moyenne par metrique :")
    print(df.groupby("metrique")["score"].mean().round(3))
    print(f"\nResultats enregistres dans {RESULTS_PATH}")
    print("\nLecture : Faithfulness -> HAUT = bon (part des affirmations appuyees).")
    print("          Hallucination -> BAS = bon.")
    print("          AnswerRelevancy -> HAUT = bon.")


if __name__ == "__main__":
    main()

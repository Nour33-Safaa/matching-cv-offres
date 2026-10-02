"""
Interface Streamlit : upload CV -> analyse -> matching -> recommandations
-> analytics -> assistant agentique.

Lancer avec : streamlit run app/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pypdf import PdfReader

from src.preprocessing import clean_text, clean_dataframe_fast
from src.feature_extraction import extract_skills
from src.matching import match_cv_to_jobs
from src.data_loader import load_job_offers
from src.genai_explain import explain_score
from src.ml_inference import ml_predict_proba
from src.agent_tools import AgentToolExecutor

st.set_page_config(page_title="Matching CV / Offres", layout="wide")

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner="Chargement et nettoyage des offres (une seule fois)...")
def get_jobs_df():
    jobs_df = load_job_offers()
    jobs_df = clean_dataframe_fast(jobs_df, batch_size=32)
    return jobs_df


PAGES = ["1. Upload CV", "2. Matching", "3. Recommandations",
         "4. Analytics", "5. Assistant IA (Agent)"]
page = st.sidebar.radio("Navigation", PAGES)


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""
if "cv_raw_text" not in st.session_state:
    st.session_state.cv_raw_text = ""


# ---------------------------------------------------------------------------
if page == "1. Upload CV":
    st.title("📄 Upload CV")
    uploaded_file = st.file_uploader("Dépose ton CV (PDF)", type=["pdf"])

    if uploaded_file:
        raw_text = extract_text_from_pdf(uploaded_file)
        st.session_state.cv_text = clean_text(raw_text)
        st.session_state.cv_raw_text = raw_text

        st.subheader("Compétences détectées")
        skills = extract_skills(raw_text)
        st.write(", ".join(skills) if skills else "Aucune compétence détectée.")

        with st.expander("Voir le texte extrait"):
            st.text(raw_text[:3000])


# ---------------------------------------------------------------------------
elif page == "2. Matching":
    st.title("🎯 Matching avec une offre")

    if not st.session_state.cv_text:
        st.warning("Upload d'abord un CV dans la page 1.")
    else:
        jobs_df = get_jobs_df()

        job_choice = st.selectbox("Choisir une offre", jobs_df["id"])
        job_row = jobs_df[jobs_df["id"] == job_choice].iloc[0]

        result = match_cv_to_jobs(
            st.session_state.cv_text, jobs_df, top_k=len(jobs_df),
            cv_raw_text=st.session_state.cv_raw_text,
        )
        row = result[result["id"] == job_choice]

        if not row.empty:
            score = row["compatibility_score"].values[0]
            matched = row["matched_skills"].values[0]
            missing = row["missing_skills"].values[0]
            job_title = job_row["title"] if "title" in job_row else f"Offre #{job_choice}"

            st.metric("Compatibilité", f"{score}%")
            st.write("✅ Compétences correspondantes :", matched)
            st.write("❌ Compétences manquantes :", missing)

            with st.expander("Voir l'offre complète"):
                st.write(f"**{job_title}**")
                st.write(job_row["text"][:3000])

            st.subheader("🤖 Explication (Generative AI)")
            if st.button("Générer l'explication"):
                with st.spinner("Génération de l'explication..."):
                    explanation = explain_score(job_title, score, matched, missing)
                st.write(explanation)

            st.subheader("🧪 Score du modèle ML supervisé")
            st.caption("Modèle entraîné sur la seule similarité textuelle "
                       "(TF-IDF + embeddings) — à comparer au score ci-dessus.")
            if st.button("Calculer le score ML"):
                with st.spinner("Prédiction en cours..."):
                    try:
                        ml_score = ml_predict_proba(
                            cv_raw_text=st.session_state.cv_raw_text,
                            job_raw_text=job_row["text"],
                            cv_clean_text=st.session_state.cv_text,
                            job_clean_text=job_row["clean_text"],
                        )
                        c1, c2 = st.columns(2)
                        c1.metric("Score embeddings", f"{score}%")
                        c2.metric("Score modèle ML", f"{ml_score}%")
                    except FileNotFoundError:
                        st.error("Aucun modèle entraîné. Lance `python -m src.train_model`.")


# ---------------------------------------------------------------------------
elif page == "3. Recommandations":
    st.title("⭐ Meilleures offres recommandées")

    if not st.session_state.cv_text:
        st.warning("Upload d'abord un CV dans la page 1.")
    else:
        jobs_df = get_jobs_df()
        top_matches = match_cv_to_jobs(
            st.session_state.cv_text, jobs_df, top_k=5,
            cv_raw_text=st.session_state.cv_raw_text,
        )
        st.dataframe(top_matches[["id", "compatibility_score",
                                   "matched_skills", "missing_skills"]])


# ---------------------------------------------------------------------------
elif page == "4. Analytics":
    st.title("📊 Analytics")

    jobs_df = get_jobs_df()
    st.metric("Nombre d'offres chargées", len(jobs_df))

    st.divider()
    st.subheader("🧪 Comparaison des modèles de Machine Learning")

    comparison_path = ROOT / "models" / "model_comparison.csv"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path, index_col=0)
        st.dataframe(comparison_df.style.highlight_max(axis=0, color="lightgreen"))

        fig, ax = plt.subplots(figsize=(8, 4))
        comparison_df.plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_title("Logistic Regression / Random Forest / XGBoost")
        ax.legend(loc="lower right")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    else:
        st.warning("Lance d'abord `python -m src.train_model`.")

    st.divider()
    st.subheader("🔍 Évaluation de l'IA générative (DeepEval)")

    genai_path = ROOT / "models" / "genai_evaluation.csv"
    if genai_path.exists():
        genai_df = pd.read_csv(genai_path)
        st.caption("Hallucination : score BAS = bon. "
                   "AnswerRelevancy : score HAUT = bon.")
        st.dataframe(genai_df)

        moyennes = genai_df.groupby("metrique")["score"].mean().round(3)
        cols = st.columns(len(moyennes))
        for col, (nom, val) in zip(cols, moyennes.items()):
            col.metric(nom, f"{val:.3f}")
    else:
        st.warning("Lance d'abord `python -m src.eval_genai`.")


# ---------------------------------------------------------------------------
elif page == "5. Assistant IA (Agent)":
    st.title("🕵️ Assistant IA agentique")
    st.caption("L'agent décide lui-même quels outils utiliser avant de répondre.")

    if not st.session_state.cv_text:
        st.warning("Upload d'abord un CV dans la page 1.")
    else:
        impl = st.radio("Implémentation de l'agent",
                        ["Manuelle (agent.py)", "LangChain (agent_langchain.py)"],
                        horizontal=True)

        examples = [
            "Quelle est la meilleure offre pour moi et pourquoi ?",
            "Quelles sont mes compétences techniques ?",
            "Compare les 3 meilleures offres pour moi.",
        ]
        st.caption("Exemples : " + " | ".join(f"*{q}*" for q in examples))

        user_question = st.text_input("Ta question :", placeholder=examples[0])

        if st.button("Envoyer") and user_question:
            jobs_df = get_jobs_df()
            executor = AgentToolExecutor(
                cv_raw_text=st.session_state.cv_raw_text,
                cv_clean_text=st.session_state.cv_text,
                jobs_df=jobs_df,
            )

            if impl.startswith("LangChain"):
                from src.agent_langchain import run_agent as run_impl
            else:
                from src.agent import run_agent as run_impl

            with st.spinner("L'agent réfléchit et utilise ses outils..."):
                answer, trace = run_impl(user_question, executor)

            st.subheader("Réponse")
            st.write(answer)

            if trace:
                with st.expander(f"🔧 {len(trace)} appel(s) d'outil"):
                    for i, step in enumerate(trace, 1):
                        st.write(f"**{i}. {step['tool']}**")
                        st.json(step["input"])

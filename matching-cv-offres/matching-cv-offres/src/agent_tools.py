"""
Outils (tools) exposés à l'agent LLM. Chaque fonction ici correspond à une
capacité que l'agent peut décider d'invoquer de façon autonome : chercher des
offres, consulter le détail d'une offre, calculer un score de compatibilité,
etc. L'agent (src/agent.py) décide LUI-MÊME quand et dans quel ordre les
appeler — ce n'est plus une séquence fixe écrite par nous.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_extraction import get_embeddings, extract_skills
from src.matching import explain_match
from src.ml_inference import ml_predict_proba


TOOLS_SCHEMA = [
    {
        "name": "search_jobs",
        "description": (
            "Recherche les offres d'emploi les plus proches sémantiquement "
            "du CV du candidat (ou d'une requête libre). Renvoie une liste "
            "d'offres avec leur id, titre et score de similarité."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Ce sur quoi baser la recherche : soit 'cv' pour "
                        "utiliser le CV du candidat tel quel, soit un texte "
                        "libre (ex. 'poste de data analyst junior')."
                    ),
                },
                "top_k": {"type": "integer", "description": "Nombre d'offres à renvoyer (défaut 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_job_details",
        "description": "Renvoie le titre et la description complète d'une offre à partir de son id.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer", "description": "Identifiant de l'offre."}},
            "required": ["job_id"],
        },
    },
    {
        "name": "score_cv_against_job",
        "description": (
            "Calcule le score de compatibilité détaillé entre le CV du "
            "candidat et une offre précise : score sémantique (embeddings), "
            "score du modèle ML supervisé, compétences correspondantes et "
            "manquantes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer", "description": "Identifiant de l'offre à évaluer."}},
            "required": ["job_id"],
        },
    },
    {
        "name": "list_cv_skills",
        "description": "Renvoie la liste des compétences techniques détectées dans le CV du candidat.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


class AgentToolExecutor:
    """
    Lie les outils au CV et au jeu d'offres courants (contexte de la session
    Streamlit), et exécute l'appel demandé par l'agent.
    """

    def __init__(self, cv_raw_text: str, cv_clean_text: str, jobs_df):
        self.cv_raw_text = cv_raw_text
        self.cv_clean_text = cv_clean_text
        self.jobs_df = jobs_df
        self._job_embeddings = None  # calculé une seule fois, à la demande (lazy)

    def _get_job_embeddings(self):
        if self._job_embeddings is None:
            self._job_embeddings = get_embeddings(self.jobs_df["clean_text"].tolist())
        return self._job_embeddings

    def execute(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "search_jobs":
            return self._search_jobs(tool_input.get("query", "cv"), tool_input.get("top_k", 5))
        if tool_name == "get_job_details":
            return self._get_job_details(tool_input["job_id"])
        if tool_name == "score_cv_against_job":
            return self._score_cv_against_job(tool_input["job_id"])
        if tool_name == "list_cv_skills":
            return self._list_cv_skills()
        return f"Erreur : outil inconnu '{tool_name}'"

    def _search_jobs(self, query: str, top_k: int) -> str:
        query_text = self.cv_clean_text if query.strip().lower() == "cv" else query
        query_embedding = get_embeddings([query_text])[0]
        job_embeddings = self._get_job_embeddings()
        sims = cosine_similarity([query_embedding], job_embeddings)[0]

        top_idx = np.argsort(sims)[::-1][:top_k]
        lines = []
        for idx in top_idx:
            row = self.jobs_df.iloc[idx]
            title = row["title"] if "title" in row else "(sans titre)"
            lines.append(f"id={row['id']} | {title} | similarité={sims[idx] * 100:.1f}%")
        return "\n".join(lines) if lines else "Aucune offre trouvée."

    def _get_job_details(self, job_id: int) -> str:
        matches = self.jobs_df[self.jobs_df["id"] == job_id]
        if matches.empty:
            return f"Aucune offre trouvée avec l'id {job_id}."
        row = matches.iloc[0]
        title = row["title"] if "title" in row else "(sans titre)"
        return f"Titre : {title}\n\nDescription :\n{row['text'][:2000]}"

    def _score_cv_against_job(self, job_id: int) -> str:
        matches = self.jobs_df[self.jobs_df["id"] == job_id]
        if matches.empty:
            return f"Aucune offre trouvée avec l'id {job_id}."
        row = matches.iloc[0]

        cv_embedding = get_embeddings([self.cv_clean_text])[0]
        job_embedding = get_embeddings([row["clean_text"]])[0]
        emb_score = float(cosine_similarity(
            cv_embedding.reshape(1, -1), job_embedding.reshape(1, -1)
        )[0][0]) * 100

        explanation = explain_match(self.cv_raw_text, row["text"])

        try:
            ml_score = ml_predict_proba(
                self.cv_raw_text, row["text"], self.cv_clean_text, row["clean_text"]
            )
        except FileNotFoundError:
            ml_score = None

        result = (
            f"Score embeddings : {emb_score:.1f}%\n"
            f"Score modèle ML : {ml_score if ml_score is not None else 'modèle non entraîné'}%\n"
            f"Compétences correspondantes : {explanation['matched_skills']}\n"
            f"Compétences manquantes : {explanation['missing_skills']}"
        )
        return result

    def _list_cv_skills(self) -> str:
        skills = extract_skills(self.cv_raw_text)
        return ", ".join(skills) if skills else "Aucune compétence technique détectée dans le CV."

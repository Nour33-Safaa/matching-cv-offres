"""
Implémentation LangChain de l'assistant agentique.

Ce module est une ALTERNATIVE à src/agent.py, pas un remplacement. Les deux
implémentations exposent la même signature run_agent(question, executor) et
utilisent les mêmes outils métier (src/agent_tools.py), ce qui permet de les
comparer directement dans le rapport :

  - src/agent.py            : boucle de tool-calling écrite à la main,
                              appels HTTP directs à Ollama / SDK Anthropic.
  - src/agent_langchain.py  : même comportement via LangChain
                              (ChatOllama + bind_tools + messages typés).

Intérêt pédagogique de conserver les deux : la version manuelle montre le
mécanisme (cycle appel / exécution / réinjection du résultat), la version
LangChain montre l'usage d'un framework standard du marché.

Dépendances : pip install langchain-ollama langchain-core
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans le matching entre CV et \
offres d'emploi. Tu as accès à des outils pour rechercher des offres, \
consulter leur détail, calculer des scores de compatibilité et lister les \
compétences du candidat.

Règles :
- Utilise les outils autant de fois que nécessaire pour répondre précisément \
à la question.
- Base-toi UNIQUEMENT sur les résultats renvoyés par les outils. N'invente \
jamais de compétence, d'expérience ou de score qui ne provient pas d'un appel \
d'outil.
- Une fois que tu as assez d'informations, réponds en français, de façon \
claire et argumentée.
"""


def _build_langchain_tools(tool_executor):
    """
    Convertit les outils du projet en outils LangChain.

    Les fonctions métier vivent dans AgentToolExecutor (src/agent_tools.py) ;
    on se contente ici de les envelopper avec le décorateur @tool, qui lit la
    signature et la docstring pour générer automatiquement le schéma transmis
    au modèle.
    """
    from langchain_core.tools import tool

    @tool
    def search_jobs(query: str, top_k: int = 5) -> str:
        """Recherche les offres d'emploi les plus proches sémantiquement du CV
        du candidat ou d'une requête libre. Utiliser la valeur 'cv' comme query
        pour partir du CV du candidat. Renvoie id, titre et score de similarité."""
        return tool_executor.execute("search_jobs", {"query": query, "top_k": top_k})

    @tool
    def get_job_details(job_id: int) -> str:
        """Renvoie le titre et la description complète d'une offre à partir de son id."""
        return tool_executor.execute("get_job_details", {"job_id": job_id})

    @tool
    def score_cv_against_job(job_id: int) -> str:
        """Calcule le score de compatibilité détaillé entre le CV du candidat et
        une offre précise : score sémantique, score du modèle ML supervisé,
        compétences correspondantes et manquantes."""
        return tool_executor.execute("score_cv_against_job", {"job_id": job_id})

    @tool
    def list_cv_skills() -> str:
        """Renvoie la liste des compétences techniques détectées dans le CV du candidat."""
        return tool_executor.execute("list_cv_skills", {})

    return [search_jobs, get_job_details, score_cv_against_job, list_cv_skills]


def run_agent(user_message: str, tool_executor, max_rounds: int = MAX_TOOL_ROUNDS):
    """
    Même signature et même valeur de retour que src/agent.run_agent :
    (réponse_finale, trace) où trace liste les outils appelés.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    except ImportError:
        return (
            "⚠️ LangChain n'est pas installé. Lance :\n"
            "pip install langchain-ollama langchain-core",
            [],
        )

    tools = _build_langchain_tools(tool_executor)
    tools_by_name = {t.name: t for t in tools}

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    trace = []

    for _ in range(max_rounds):
        try:
            ai_message = llm_with_tools.invoke(messages)
        except Exception as e:
            return f"Erreur lors de l'appel au modèle via LangChain : {e}", trace

        messages.append(ai_message)

        # Pas d'appel d'outil demandé -> le modèle a produit sa réponse finale.
        if not getattr(ai_message, "tool_calls", None):
            return ai_message.content, trace

        for call in ai_message.tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            trace.append({"tool": name, "input": args})

            selected = tools_by_name.get(name)
            if selected is None:
                result = f"Erreur : outil inconnu '{name}'"
            else:
                try:
                    result = selected.invoke(args)
                except Exception as e:
                    result = f"Erreur lors de l'exécution de l'outil {name} : {e}"

            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=call.get("id", name),
            ))

    return "L'agent n'a pas pu conclure dans le nombre d'étapes imparti.", trace

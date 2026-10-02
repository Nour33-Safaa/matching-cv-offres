"""
Boucle agentique : le LLM reçoit une question libre du candidat, et décide
LUI-MÊME quels outils appeler (src/agent_tools.py), combien de fois, et dans
quel ordre, jusqu'à pouvoir formuler une réponse finale argumentée.

Contrairement à src/genai_explain.py (un seul appel LLM, entrée/sortie
fixes), ici le modèle peut enchaîner plusieurs appels d'outils avant de
répondre — c'est ce qui définit un comportement "agentique".

Fonctionne avec Ollama (local, gratuit) OU l'API Anthropic, via le module
centralisé src/llm_provider.py. Contrairement à genai_explain.py, il n'y a
PAS de mode fallback ici : un agent a besoin d'un LLM pour décider de ses
actions, un template ne peut pas le remplacer.
"""

from src.llm_provider import get_active_provider, chat_with_tools

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans le matching entre CV et \
offres d'emploi. Tu as accès à des outils pour rechercher des offres, \
consulter leur détail, calculer des scores de compatibilité et lister les \
compétences du candidat.

Règles :
- Utilise les outils autant de fois que nécessaire pour répondre précisément \
à la question — n'hésite pas à consulter plusieurs offres avant de conclure.
- Base-toi UNIQUEMENT sur les résultats renvoyés par les outils. N'invente \
jamais de compétence, d'expérience ou de score qui ne provient pas d'un appel \
d'outil.
- Une fois que tu as assez d'informations, réponds en français, de façon \
claire et argumentée (pas de liste brute de scores sans explication).
- Limite-toi à 8 appels d'outils maximum avant de conclure.
"""

MAX_TOOL_ROUNDS = 8


def run_agent(user_message: str, tool_executor, max_rounds: int = MAX_TOOL_ROUNDS):
    """
    Lance la boucle agentique. Renvoie (réponse_finale, trace) où trace est
    la liste des outils appelés (utile pour affichage transparent dans l'UI).
    """
    if get_active_provider() == "none":
        return (
            "⚠️ Aucun LLM disponible. Lance un serveur Ollama local "
            "(`ollama pull qwen3:8b` puis relance l'app), ou configure une "
            "clé ANTHROPIC_API_KEY dans .env — l'assistant agentique a "
            "besoin d'un LLM pour décider de ses actions, il n'y a pas de "
            "mode de secours possible ici.",
            [],
        )

    from src.agent_tools import TOOLS_SCHEMA

    messages = [{"role": "user", "content": user_message}]
    trace = []

    for _ in range(max_rounds):
        result = chat_with_tools(SYSTEM_PROMPT, messages, TOOLS_SCHEMA)

        if not result["tool_calls"]:
            return result["content"], trace

        # Reconstruction de l'historique, différente selon le fournisseur
        # (Anthropic exige les blocs bruts avec tool_use_id ; Ollama accepte
        # une simple liste de messages "tool").
        if result["provider"] == "anthropic":
            messages.append({"role": "assistant", "content": result["raw_content_blocks"]})
            for tc in result["tool_calls"]:
                trace.append({"tool": tc["name"], "input": tc["arguments"]})
                tool_result = tool_executor.execute(tc["name"], tc["arguments"])
                messages.append({
                    "role": "tool",
                    "tool_use_id": tc["id"],
                    "content": tool_result,
                })
        else:  # ollama
            messages.append({"role": "assistant", "content": result["content"]})
            for tc in result["tool_calls"]:
                trace.append({"tool": tc["name"], "input": tc["arguments"]})
                tool_result = tool_executor.execute(tc["name"], tc["arguments"])
                messages.append({
                    "role": "tool",
                    "tool_name": tc["name"],
                    "content": tool_result,
                })

    return "L'agent n'a pas pu conclure dans le nombre d'étapes imparti.", trace

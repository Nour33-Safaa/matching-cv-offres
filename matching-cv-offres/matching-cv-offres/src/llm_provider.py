"""
Fournisseur LLM centralisé : choisit automatiquement entre Ollama (local,
gratuit) et l'API Anthropic (payante), selon ce qui est disponible.

Ordre de priorité :
1. Ollama, si le serveur local répond (http://localhost:11434) — gratuit,
   aucune clé nécessaire. C'est le mode recommandé pour ce projet.
2. API Anthropic, si une clé ANTHROPIC_API_KEY est configurée dans .env.
3. Aucun LLM disponible → les modules appelants gèrent leur propre fallback
   (ex. genai_explain.py utilise un template ; agent.py n'a pas de fallback
   possible, un agent a besoin d'un LLM pour décider de ses actions).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Le tout premier appel doit charger le modèle (plusieurs Go) en mémoire, ce
# qui peut prendre 1-2 minutes selon le disque. Les appels suivants sont
# beaucoup plus rapides tant qu'Ollama garde le modèle chargé.
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))


def ollama_available() -> bool:
    """Vérifie rapidement si un serveur Ollama local répond."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def get_active_provider() -> str:
    """Renvoie 'ollama', 'anthropic' ou 'none' selon ce qui est disponible."""
    if ollama_available():
        return "ollama"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return "none"


# ---------------------------------------------------------------------------
# Appel simple (une question, une réponse, sans outils) — utilisé par
# genai_explain.py.
# ---------------------------------------------------------------------------

def simple_chat(system_prompt: str, user_prompt: str, max_tokens: int = 250) -> str:
    provider = get_active_provider()

    if provider == "ollama":
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    raise RuntimeError("Aucun LLM disponible (ni Ollama, ni clé Anthropic).")


# ---------------------------------------------------------------------------
# Appel avec outils (tool use / function calling) — utilisé par agent.py.
# Renvoie un format normalisé, indépendant du fournisseur :
#   {"content": str, "tool_calls": [{"name": str, "arguments": dict}]}
# ---------------------------------------------------------------------------

def _anthropic_tools_to_openai(tools_schema: list[dict]) -> list[dict]:
    """Convertit le format d'outils Anthropic vers le format OpenAI/Ollama."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools_schema
    ]


def chat_with_tools(system_prompt: str, messages: list[dict], tools_schema: list[dict]):
    """
    messages : liste de dicts {"role": "user"/"assistant"/"tool", "content": ...}
    (format simplifié interne, reconstruit pour chaque fournisseur ci-dessous)
    """
    provider = get_active_provider()

    if provider == "ollama":
        openai_tools = _anthropic_tools_to_openai(tools_schema)
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": full_messages,
                "tools": openai_tools,
                "stream": False,
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        msg = response.json()["message"]
        tool_calls = [
            {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
            for tc in msg.get("tool_calls", [])
        ]
        return {"content": msg.get("content", ""), "tool_calls": tool_calls, "provider": "ollama"}

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        anthropic_messages = []
        for m in messages:
            if m["role"] == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m["tool_use_id"],
                        "content": m["content"],
                    }],
                })
            else:
                anthropic_messages.append(m)

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            system=system_prompt,
            tools=tools_schema,
            messages=anthropic_messages,
        )

        tool_calls = [
            {"name": b.name, "arguments": b.input, "id": b.id}
            for b in response.content if b.type == "tool_use"
        ]
        text = "".join(b.text for b in response.content if b.type == "text")
        return {
            "content": text, "tool_calls": tool_calls, "provider": "anthropic",
            "raw_content_blocks": response.content,
        }

    raise RuntimeError("Aucun LLM disponible (ni Ollama, ni clé Anthropic).")

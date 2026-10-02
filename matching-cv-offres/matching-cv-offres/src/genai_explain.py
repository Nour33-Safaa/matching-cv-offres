"""
Partie Generative AI (section 7 du cahier des charges) :
générer une explication en langage naturel du score de compatibilité
CV / offre, à partir des compétences correspondantes et manquantes.

Contrainte imposée par le sujet : le système ne doit jamais présenter
comme certaines des informations qui ne sont pas présentes dans les
données (matched_skills / missing_skills). Le prompt impose donc au LLM
de se baser STRICTEMENT sur les listes fournies, sans extrapoler.

Trois modes possibles (voir src/llm_provider.py) :
- Ollama (local, gratuit) si un serveur tourne sur la machine.
- API Anthropic si une clé est configurée.
- Fallback local (template, aucun LLM) si aucun des deux n'est disponible.
"""

from src.llm_provider import simple_chat, get_active_provider

SYSTEM_PROMPT = """Tu es un assistant qui explique un score de compatibilité \
entre un CV et une offre d'emploi, à destination du candidat.

Règles strictes :
- Base-toi UNIQUEMENT sur les listes de compétences fournies (correspondantes \
et manquantes). N'invente et ne suppose AUCUNE information qui n'y figure pas.
- Ne mentionne pas l'expérience, la formation ou les soft skills du candidat \
sauf si ces informations sont explicitement données.
- Reste factuel, concis (3-4 phrases maximum), et bienveillant.
- Réponds en français.
"""


def _build_user_prompt(job_title: str, score: float,
                        matched_skills: list, missing_skills: list) -> str:
    return f"""Poste : {job_title}
Score de compatibilité : {score}%
Compétences correspondantes : {", ".join(matched_skills) if matched_skills else "aucune"}
Compétences manquantes : {", ".join(missing_skills) if missing_skills else "aucune"}

Explique en 3-4 phrases pourquoi ce candidat correspond ou non à cette offre."""


def _fallback_explanation(job_title: str, score: float,
                           matched_skills: list, missing_skills: list) -> str:
    """Explication générée sans LLM, purement à partir d'un template."""
    if score >= 75:
        niveau = "une très bonne adéquation"
    elif score >= 50:
        niveau = "une adéquation correcte"
    else:
        niveau = "une adéquation limitée"

    parts = [f"Le candidat présente {niveau} avec le poste de {job_title} "
             f"(score de compatibilité : {score}%)."]

    if matched_skills:
        parts.append(f"Il maîtrise notamment : {', '.join(matched_skills)}.")
    else:
        parts.append("Aucune compétence technique commune identifiée dans les données disponibles.")

    if missing_skills:
        parts.append(f"En revanche, l'offre demande également : {', '.join(missing_skills)}, "
                      "qui ne sont pas présentes dans le CV.")

    return " ".join(parts)


def explain_score(job_title: str, score: float,
                   matched_skills: list, missing_skills: list) -> str:
    """Point d'entrée principal. Choisit automatiquement Ollama > Anthropic > template."""
    if get_active_provider() == "none":
        return _fallback_explanation(job_title, score, matched_skills, missing_skills)

    try:
        user_prompt = _build_user_prompt(job_title, score, matched_skills, missing_skills)
        return simple_chat(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        fallback = _fallback_explanation(job_title, score, matched_skills, missing_skills)
        return f"{fallback}\n\n[Note : explication générée localement, LLM indisponible: {e}]"


if __name__ == "__main__":
    print(explain_score(
        job_title="AI Engineer",
        score=84.0,
        matched_skills=["python", "machine learning", "sql"],
        missing_skills=["kubernetes", "aws"],
    ))

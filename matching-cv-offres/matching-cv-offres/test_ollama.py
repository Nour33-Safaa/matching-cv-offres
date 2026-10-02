"""
Script de test rapide et isolé pour vérifier qu'Ollama fonctionne
correctement avec le projet, sans passer par Streamlit.

Lancer avec : python test_ollama.py
"""

from src.llm_provider import ollama_available, get_active_provider, simple_chat

print("1. Ollama détecté ?", ollama_available())
print("2. Fournisseur actif :", get_active_provider())

if get_active_provider() != "none":
    print("\n3. Test d'une génération simple...")
    response = simple_chat(
        system_prompt="Tu réponds en une seule phrase, en français.",
        user_prompt="Dis bonjour et confirme que tu fonctionnes.",
    )
    print("Réponse :", response)
else:
    print("\nAucun LLM disponible — vérifie qu'Ollama tourne bien "
          "(icône dans la barre des tâches) et que le modèle est téléchargé "
          "(`ollama list` doit afficher qwen3:8b).")

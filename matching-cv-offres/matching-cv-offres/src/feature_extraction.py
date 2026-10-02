"""
Extraction de features : deux méthodes comparées dans le projet.

1) TF-IDF (baseline)          -> get_tfidf_vectors()
2) Sentence Transformers      -> get_embeddings()

Une liste de compétences techniques prédéfinie permet aussi une extraction
simple par correspondance de mots-clés (utile pour le score explicable).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# --- Modèle d'embeddings (léger et rapide) ---
_EMBED_MODEL = None


def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def get_tfidf_vectors(corpus: list[str]):
    """Renvoie (matrice TF-IDF, vectorizer fitté) pour un corpus de textes."""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer


def get_embeddings(texts: list[str]):
    """Renvoie les embeddings sémantiques (numpy array) pour une liste de textes."""
    model = get_embed_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


# --- Extraction de compétences par dictionnaire (à enrichir selon ton domaine) ---
SKILL_KEYWORDS = [
    "python", "sql", "java", "javascript", "machine learning", "deep learning",
    "nlp", "docker", "kubernetes", "aws", "azure", "gcp", "scikit-learn",
    "tensorflow", "pytorch", "pandas", "numpy", "xgboost", "fastapi",
    "streamlit", "git", "linux", "spark", "airflow", "power bi", "tableau",
]


def extract_skills(text: str) -> list[str]:
    """Extraction de compétences par correspondance avec le référentiel.

    À appliquer au texte BRUT, non lemmatisé : la lemmatisation casse les
    expressions à plusieurs mots (« machine learning » devient « machine
    learn ») et le nettoyage supprime les traits d'union, de sorte que les
    mots-clés ne s'y retrouvent plus.
    """
    text_norm = text.lower().replace("-", " ")
    return sorted({
        skill for skill in SKILL_KEYWORDS
        if skill.replace("-", " ") in text_norm
    })


if __name__ == "__main__":
    sample_texts = ["Python developer with Machine Learning and Docker experience"]
    print("Skills:", extract_skills(sample_texts[0]))
    print("Embedding shape:", get_embeddings(sample_texts).shape)

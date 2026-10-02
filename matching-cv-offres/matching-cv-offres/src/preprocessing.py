"""
Nettoyage et prétraitement du texte (CV et offres d'emploi).

Étapes : mise en minuscule, suppression des emails/téléphones/URLs,
suppression de la ponctuation, tokenisation, suppression des stopwords,
lemmatisation (spaCy).
"""

import re
import spacy

# Charger une seule fois le modèle spaCy (anglais par défaut, adapte si besoin)
# python -m spacy download en_core_web_md
# On désactive parser + ner : on n'a besoin que de la tokenisation/lemmatisation,
# et ça accélère énormément le traitement (souvent 5-10x plus rapide).
try:
    NLP = spacy.load("en_core_web_md", disable=["parser", "ner"])
except OSError:
    NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])

EMAIL_RE = re.compile(r"\S+@\S+")
URL_RE = re.compile(r"http\S+|www\.\S+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
PUNCT_RE = re.compile(r"[^\w\s]")
MULTISPACE_RE = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    """Nettoyage regex simple, avant tokenisation spaCy."""
    text = text.lower()
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def lemmatize(text: str) -> str:
    """Lemmatisation + suppression des stopwords via spaCy."""
    doc = NLP(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
        and len(token.text) > 1
    ]
    return " ".join(tokens)


def clean_text(text: str) -> str:
    """Pipeline complet de nettoyage utilisé partout dans le projet."""
    if not isinstance(text, str) or not text.strip():
        return ""
    return lemmatize(basic_clean(text))


def clean_dataframe(df, column: str = "text", new_column: str = "clean_text"):
    """Version simple (lente) : garde-la seulement pour de petits volumes de test."""
    df[new_column] = df[column].apply(clean_text)
    return df


def clean_dataframe_fast(df, column: str = "text", new_column: str = "clean_text",
                          batch_size: int = 64, n_process: int = 1):
    """
    Version rapide utilisant nlp.pipe() pour traiter les textes en batch
    (et en parallèle si n_process > 1). À utiliser pour tout le dataset.

    n_process > 1 ne fonctionne que si le script est protégé par
    `if __name__ == "__main__":` (obligatoire sur Windows).
    """
    raw_texts = df[column].fillna("").astype(str)
    basic_cleaned = raw_texts.apply(basic_clean).tolist()

    lemmatized = []
    for doc in NLP.pipe(basic_cleaned, batch_size=batch_size, n_process=n_process):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
            and len(token.text) > 1
        ]
        lemmatized.append(" ".join(tokens))

    df[new_column] = lemmatized
    return df


if __name__ == "__main__":
    sample = "Contact me at john.doe@email.com or +33 6 12 34 56 78. Python & Machine Learning expert!"
    print("Avant :", sample)
    print("Après :", clean_text(sample))

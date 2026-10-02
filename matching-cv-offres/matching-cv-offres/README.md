# Matching CV / Offres d'emploi

Système de mise en correspondance automatique entre un CV et une offre
d'emploi, combinant traitement du langage, apprentissage supervisé et
intelligence artificielle générative.

Projet de Fin d'Année — EMSI, stage réalisé chez Capgemini France.

---

## Sommaire

1. [Ce que fait le projet](#1-ce-que-fait-le-projet)
2. [Prérequis](#2-prérequis)
3. [Installation](#3-installation)
4. [Récupération des données](#4-récupération-des-données)
5. [Modèle de langage local](#5-modèle-de-langage-local)
6. [Ordre d'exécution](#6-ordre-dexécution)
7. [Structure du dépôt](#7-structure-du-dépôt)
8. [L'application](#8-lapplication)
9. [Résultats obtenus](#9-résultats-obtenus)
10. [Dépannage](#10-dépannage)
11. [Limites connues](#11-limites-connues)

---

## 1. Ce que fait le projet

À partir d'un CV au format PDF, le système :

- extrait le texte et les compétences techniques ;
- calcule un **score de compatibilité** avec une offre d'emploi ;
- liste les **compétences partagées et manquantes** ;
- recommande les **cinq offres** les plus pertinentes ;
- **explique le score en langue naturelle** via un modèle local ;
- répond à une question libre via un **assistant agentique**.

Trois méthodes de matching sont comparées : TF-IDF, plongements sémantiques
et apprentissage supervisé.

---

## 2. Prérequis

| Élément | Remarque |
|---|---|
| Python | 3.10 ou supérieur |
| RAM | 8 Go minimum, 16 Go confortables |
| Disque | environ 10 Go (modèles et données) |
| GPU | facultatif ; accélère Ollama |
| Compte Kaggle | gratuit, pour télécharger les jeux de données |

Aucune clé d'API payante n'est nécessaire : le modèle de langage tourne en
local.

---

## 3. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt

python -m spacy download en_core_web_md
python -m spacy download fr_core_news_md
```

Sous Windows, ajouter aussi :

```powershell
pip install "portalocker[win32]"
```

Sans ce paquet, DeepEval ne peut pas mettre ses résultats en cache et
recalcule tout à chaque exécution.

---

## 4. Récupération des données

### Authentification Kaggle

Kaggle a changé son système : l'ancien fichier `kaggle.json` est remplacé par
un jeton au format `KGAT_`.

1. Aller sur https://www.kaggle.com/settings, section **API**
2. Cliquer sur **Create New Token** et copier le jeton affiché
3. L'enregistrer :

```bash
mkdir -p ~/.kaggle && echo "KGAT_votre_jeton" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

```powershell
mkdir $env:USERPROFILE\.kaggle -Force
"KGAT_votre_jeton" | Out-File -Encoding ascii $env:USERPROFILE\.kaggle\access_token
```

Vérification : `kaggle datasets list`

### Téléchargement

```bash
kaggle datasets download -d snehaanbhawal/resume-dataset -p data/raw --unzip
kaggle datasets download -d kshitizregmi/jobs-and-job-description -p data/raw --unzip
```

Arborescence attendue :

```
data/raw/
├── Resume/
│   └── Resume.csv          # 2 484 CV — ID, Resume_str, Resume_html, Category
└── job_title_des.csv       # 2 277 offres — Job Title, Job Description
```

Les chemins et le renommage des colonnes sont gérés dans
`src/data_loader.py` : c'est le seul fichier à adapter si vous changez de
jeu de données.

---

## 5. Modèle de langage local

```bash
ollama pull qwen3:8b              # génération et agent
ollama pull qwen2.5:7b-instruct   # juge DeepEval
```

Deux modèles pour deux usages. `qwen3:8b` raisonne avant de répondre, ce qui
sert à l'agent mais rend chaque appel trop lent pour un rôle de juge, où l'on
enchaîne des dizaines d'appels.

Vérification :

```bash
python test_ollama.py
```

Sortie attendue :

```
1. Ollama détecté ? True
2. Fournisseur actif : ollama
3. Test d'une génération simple...
Réponse : Bonjour ! Je fonctionne correctement.
```

---

## 6. Ordre d'exécution

| # | Commande | Durée | Produit |
|---|---|---|---|
| 1 | `python main.py` | ~5 min | Vérifie le pipeline de bout en bout |
| 2 | `python -m src.train_model` | ~10 min | `models/*.joblib`, `model_comparison.csv` |
| 3 | `python -u -m src.eval_genai` | ~5 min | `models/genai_evaluation.csv` |
| 4 | `streamlit run app/app.py` | — | Lance l'application |
| 5 | `pytest tests/` | — | Tests unitaires |

Les étapes 2 et 3 sont nécessaires : sans elles, les pages Matching (score
ML) et Analytics affichent un avertissement.

---

## 7. Structure du dépôt

```
matching-cv-offres/
├── data/
│   ├── raw/                  # CSV téléchargés depuis Kaggle
│   ├── processed/            # données nettoyées
│   └── external/
├── src/
│   ├── data_loader.py        # chargement, normalisation des colonnes
│   ├── preprocessing.py      # nettoyage, lemmatisation, traitement par lots
│   ├── feature_extraction.py # TF-IDF, plongements, extraction de compétences
│   ├── matching.py           # score de compatibilité, top-K
│   ├── pair_features.py      # descripteurs de paire, supervision faible
│   ├── train_model.py        # entraînement et comparaison des 3 modèles
│   ├── ml_inference.py       # prédiction sur une nouvelle paire
│   ├── evaluate.py           # métriques et tableau comparatif
│   ├── llm_provider.py       # sélection du fournisseur LLM
│   ├── genai_explain.py      # explication générée du score
│   ├── agent.py              # agent — implémentation manuelle
│   ├── agent_langchain.py    # agent — implémentation LangChain
│   ├── agent_tools.py        # les 4 outils exposés à l'agent
│   ├── eval_genai.py         # évaluation DeepEval
│   ├── analytics.py          # statistiques du corpus (non branché)
│   └── utils.py              # sauvegarde / chargement des modèles
├── app/app.py                # interface Streamlit, 5 pages
├── models/                   # modèles entraînés et résultats
├── tests/                    # tests unitaires
├── main.py                   # pipeline complet
├── test_ollama.py            # vérification du LLM local
├── requirements.txt
└── .env.example              # à copier en .env
```

---

## 8. L'application

```bash
streamlit run app/app.py
```

| Page | Contenu |
|---|---|
| **1. Upload CV** | Dépôt d'un PDF, extraction du texte et des compétences |
| **2. Matching** | Score face à une offre, compétences, explication générée, score ML |
| **3. Recommandations** | Top 5 des offres pour le CV déposé |
| **4. Analytics** | Comparatif des modèles et résultats DeepEval |
| **5. Assistant IA** | Agent conversationnel, bascule manuelle / LangChain, trace des outils |

Le premier accès aux pages 2, 3 ou 4 déclenche le nettoyage du corpus, mis en
cache ensuite. Compter une à deux minutes au premier chargement.

---

## 9. Résultats obtenus

### Corpus

| Indicateur | Valeur |
|---|---|
| CV | 2 484, répartis en 24 catégories |
| Offres d'emploi | 2 277 |
| CV retenus pour l'entraînement | 815 |
| Offres retenues | 1 869 |
| Paires construites | 4 888 — 2 742 positives, 2 146 négatives |

### Modèles supervisés

| Modèle | Exactitude | Précision | Rappel | F1 |
|---|---|---|---|---|
| Régression logistique | 0,540 | 0,635 | 0,424 | 0,509 |
| Forêt aléatoire | 0,532 | 0,587 | 0,559 | 0,573 |
| **XGBoost** | **0,546** | **0,601** | **0,568** | **0,584** |

Un F1 de 0,584 pour un tirage aléatoire équilibré à 0,50 : le gain est réel
mais modeste. C'est un résultat honnête, et instructif — il montre que la
similarité textuelle globale ne suffit pas à prédire la présence de
compétences techniques précises. D'où l'architecture retenue, qui combine
score sémantique et extraction explicite de compétences.

### Partie générative

Comparaison de deux juges sur les quatre mêmes cas :

| Métrique | Juge `llama3.2:3b` | Juge `qwen2.5:7b-instruct` |
|---|---|---|
| Hallucination (bas = bon) | 0,750 | **0,000** |
| Fidélité (haut = bon) | 0,334 | **0,750** |
| Pertinence (haut = bon) | 0,771 | 0,675 |

Le juge 3B produisait des scores dont les justifications se contredisaient.
La fiabilité d'un LLM-as-judge dépend directement de la taille du juge.

---

## 10. Dépannage

### `ollama : le terme n'est pas reconnu`

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen3:8b
```

Correctif permanent, puis rouvrir le terminal :

```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:LOCALAPPDATA\Programs\Ollama", "User")
```

### `FileNotFoundError: ... Resume.csv`

Le fichier se trouve dans un sous-dossier : `data/raw/Resume/Resume.csv`.

### Le prétraitement semble bloqué

Vérifier que `clean_dataframe_fast()` est utilisé et non `clean_text()`
appliqué ligne par ligne, et que spaCy est chargé avec
`disable=["parser", "ner"]`.

### DeepEval : `TimeoutError: call timed out after 88.5s`

Le juge est trop lent — typiquement un modèle à raisonnement. Utiliser
`qwen2.5:7b-instruct`, et vérifier que `async_mode=False` est bien positionné
sur les métriques.

### DeepEval : score constant de 0,50

Le contexte est fragmenté. `build_context()` doit renvoyer **une liste d'un
seul élément** contenant toutes les informations.

### `huggingface-hub requires click>=8.4.2, but you have click 8.3.3`

Conflit bénin introduit par DeepEval. Vérification :

```bash
python -c "from src.feature_extraction import get_embeddings; print(get_embeddings(['test']).shape)"
```

Si la sortie est `(1, 384)`, tout fonctionne.

### `matched_skills` et `missing_skills` vides

Comportement attendu si le CV ne contient aucune compétence du référentiel —
un profil RH, par exemple. Tester avec un CV technique.

---

## 11. Limites connues

- **Aucune annotation humaine.** Les étiquettes viennent d'une règle
  automatique ; le modèle apprend à prédire cette règle, pas la pertinence
  telle qu'un recruteur l'évaluerait.
- **Extraction de compétences par référentiel fixe** — 26 termes, sans
  reconnaissance d'entités nommées.
- **Périmètre restreint aux profils techniques.**
- **TF-IDF n'est pas évalué comme méthode autonome** ; il n'intervient que
  comme descripteur du modèle supervisé.
- **Extraction structurée du CV absente** — seules les compétences sont
  extraites, ni le nom, ni la formation, ni les expériences.
- **Couche API REST non développée.**
- **`src/analytics.py` présent mais non branché** dans l'application.
- **Juge d'évaluation de taille modeste** — les scores DeepEval illustrent
  une démarche outillée, non des mesures définitives.
- **Tests unitaires limités** au module de prétraitement.

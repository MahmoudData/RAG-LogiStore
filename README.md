# RAG-LogiStore

Moteur de recherche semantique de tickets support, base sur un pipeline RAG (Retrieval-Augmented Generation) avec Qdrant Cloud.

## Architecture

```
app/
  ingestion.py      Ingestion des tickets dans Qdrant (dense + sparse + ColBERT)
  rag_engine.py     Moteur de recherche unifie (dense, sparse, hybrid, rerank)
  app.py            Interface Streamlit

eval/
  eval_utils.py     Fonctions d'evaluation (Hit@K, MRR, RAGAS, correlations)
  rag_eval.ipynb    Pipeline d'evaluation complet

notebooks/
  eda_tickets.ipynb         Analyse exploratoire
  cleaning_tickets.ipynb    Nettoyage des donnees

docs/
  veille.md         Veille technologique (embeddings, types de RAG, evaluation)

data/
  raw/              Dataset brut (tickets multilingues)
  processed/        Tickets nettoyes + resultats d'evaluation
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Base vectorielle | Qdrant Cloud (inference server-side) |
| Embeddings dense | `intfloat/multilingual-e5-small` (384d) |
| Embeddings sparse | `Qdrant/bm25` |
| Reranking | `answerdotai/answerai-colbert-small-v1` (ColBERT, 96d) |
| Frontend | Streamlit |
| LLM (synthese + eval) | Gemini Flash |
| Evaluation | RAGAS (ContextPrecision, ContextRecall, ContextEntityRecall) |

## Pipeline de recherche

```
Query
  |
  v
[E5 dense] + [BM25 sparse]     (prefetch K=20 chacun)
  |
  v
[ColBERT reranking]              (rescore server-side)
  |
  v
Top-K resultats                  (K=7 optimal)
  |
  v
[Synthese LLM] (optionnel)      (Gemini Flash via OpenRouter)
```

## Methodes de recherche

| Methode | Description |
|---------|-------------|
| `dense` | Similarite cosinus sur embeddings E5 |
| `sparse` | BM25 (mots-cles) |
| `hybrid` | Fusion RRF (dense + sparse) |
| `hybrid_rerank` | Hybrid + reranking ColBERT (meilleure config) |

## Evaluation

Pipeline d'evaluation en entonnoir (du gratuit vers le couteux) :

| Etape | Objectif | Metrique | Cout |
|-------|----------|----------|------|
| 1. Screening | Meilleure methode/filtre | Hit@K / MRR (384 samples) | 0 |
| 2. Embedding | E5 vs MiniLM | Hit@K / MRR | 0 |
| 3. Fenetre K | K optimal (3, 5, 7, 10) | Hit@K | 0 |
| 4. RAGAS Baseline | Qualite des contextes | ContextPrecision / Recall (180 samples) | LLM |
| 5. Reranking | Impact ColBERT | RAGAS baseline vs reranked | LLM |
| 6. Robustesse | Stabilite inter-juges | Correlation Pearson/Spearman | LLM |

### Resultats cles

- **Meilleure config** : E5 / hybrid_rerank / queue_filter / K=7
- **ColBERT reranking** : ameliore les scores RAGAS
- **E5 > MiniLM** sur toutes les metriques (Hit@K et RAGAS)

## Installation

```bash
git clone https://github.com/MahmoudData/RAG-LogiStore.git
cd RAG-LogiStore
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Variables d'environnement

Creer un fichier `.env` a la racine :

```
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-api-key
OPENROUTER_API_KEY=your-openrouter-key
```

### Ingestion

```bash
cd app
python ingestion.py --model e5 --colbert
```

### Lancer l'application

```bash
cd app
streamlit run app.py
```

## Donnees

- **Source** : dataset de tickets support multilingues (~28 000 tickets)
- **Filtrage** : tickets anglais avec sujet et reponse (13 729 tickets)
- **Stratification** : 4 types (Change, Incident, Problem, Request) x 3 priorites (high, medium, low)

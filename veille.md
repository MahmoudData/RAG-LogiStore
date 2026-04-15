## Indexation sémantique par plongement vectoriel

L’indexation sémantique par plongement repose sur la transformation de textes (mots, phrases, fragments, documents) en vecteurs numériques denses dans un espace de dimension réduite, où la proximité géométrique reflète la proximité de sens. [ibm](https://www.ibm.com/fr-fr/think/topics/word-embeddings)
Ces plongements sont généralement produits par des modèles d’embeddings (ex. Sentence‑Transformers, BGE, E5, etc.) qui capturent la sémantique contextuelle et les relations entre unités linguistiques. [modalbgroup](https://www.modalbgroup.com/actualites/rag-production-patterns-evaluation-couts)

### Principes clés  
- **Similarité sémantique** : les requêtes et les documents sont vectorisés et comparés via des métriques de similarité (cosinus, produit scalaire) afin de retrouver des contenus proches en sens, même avec vocabulaire différent. [elastic](https://www.elastic.co/fr/what-is/vector-embedding)
- **Chunking et indexation** : pour les RAG, le corpus est découpé en blocs de contexte (chunks) que l’on encode, puis stocke dans une base vectorielle (ex. Weaviate, Qdrant, Milvus) pour permettre des recherches rapides. [ibm](https://www.ibm.com/fr-fr/think/topics/rag-techniques)

### Avantages par rapport aux approches classiques  
- Meilleure tolérance aux reformulations, synonymes, et expressions variées qu’un moteur purement basé sur mots‑clés ou TF‑IDF. [ibm](https://www.ibm.com/fr-fr/think/topics/word-embeddings)
- Intégration naturelle dans les pipelines RAG : la requête utilisateur est embeddée et servie comme « point d’interrogation » dans l’espace vectoriel. [meilisearch](https://www.meilisearch.com/blog/rag-types)

***

## Types de RAG (Retrieval‑Augmented Generation)

Le RAG (Retrieval‑Augmented Generation) est un paradigme qui combine une phase de recherche d’information dans une base externe et une phase de génération par un LLM. [meilisearch](https://www.meilisearch.com/blog/rag-types)
Plusieurs architectures dérivées ont émergé, chacune optimisées pour des cas d’usage spécifiques (interactivité, précision, réactivité, agents autonomes, etc.). [digitalmate](https://www.digitalmate.fr/rag-agent-multi-agents/)

### Principaux types de RAG

| Type de RAG              | Description synthétique                                                                 |
|--------------------------|-----------------------------------------------------------------------------------------|
| **Simple RAG (classique)** | Recherche en une étape, puis génération unique à partir des documents récupérés.  [meilisearch](https://www.meilisearch.com/blog/rag-types) |
| **Simple RAG + mémoire**  | Ajout d’un mécanisme de mémoire (conversation, sessions utilisateur) pour adapter la base de contexte.  [meilisearch](https://www.meilisearch.com/blog/rag-types) |
| **Agentic RAG**           | Le LLM agit comme un agent : planifie requêtes, itère sur la recherche, choisit des outils.  [meilisearch](https://www.meilisearch.com/blog/rag-types) |
| **Graph RAG**             | Les connaissances sont structurées sous forme de graphe (entités, relations) pour améliorer la précision et la multi‑sauts.  [meilisearch](https://www.meilisearch.com/blog/rag-types) |
| **Self‑RAG**              | Le modèle décide lui‑même quand interroger la base de connaissances (self‑query) et filtre ses résultats.  [meilisearch](https://www.meilisearch.com/blog/rag-types) |
| **Modular RAG**           | Architecture modulaire : composants de recherche, de filtrage, de reclassement, et de génération séparés et composable.  [meilisearch](https://www.meilisearch.com/blog/rag-types) |
| **Multimodal RAG**        | Extension aux données non textuelles (images, audio, vidéo) via des embeddings multimodaux.  [meilisearch](https://www.meilisearch.com/blog/rag-types) |
| **Corrective RAG**        | Intègre un feedback loop pour corriger les réponses erronées ou incohérentes issues du RAG.  [meilisearch](https://www.meilisearch.com/blog/rag-types) |
| **Advanced / Naive RAG**  | Variantes exploratoires (ex. requêtes multiples, optimisations de chunks, filtrages avancés).  [meilisearch](https://www.meilisearch.com/blog/rag-types) |

### Tendances et usages émergents  
- Les architectures **Agentic RAG** et **multi‑agents** se généralisent pour des scénarios complexes (veille continue, pilotage de workflows, analyse de risques). [digitalmate](https://www.digitalmate.fr/rag-agent-multi-agents/)
- Les schémas **modulaires** facilitent le déploiement en production, la maintenance des pipelines et l’optimisation coûts/performance. [modalbgroup](https://www.modalbgroup.com/actualites/rag-production-patterns-evaluation-couts)

***

## Évaluation des solutions RAG et métriques associées

L’évaluation d’un système RAG se fait en deux axes :  
1. la qualité de la **recherche** (documents récupérés),  
2. la qualité de la **réponse générée** par le LLM à partir de ces documents. [lemagit](https://www.lemagit.fr/conseil/Comment-evaluer-un-systeme-RAG)

### Métriques liées à la recherche (retrieval)

Pour quantifier la pertinence des documents retournés :  
- **Recall (rappel)** : proportion des documents pertinents du corpus qui sont effectivement retournés. [louisbouchard](https://www.louisbouchard.ca/blog-ia/evaluer-rag)
- **Precision (précision)** : proportion des documents retournés qui sont réellement pertinents. [lemagit](https://www.lemagit.fr/conseil/Comment-evaluer-un-systeme-RAG)
- **MRR (Mean Reciprocal Rank)** : mesures la position du premier document pertinent dans le classement. [blog.octo](https://blog.octo.com/evaluation-rag--bonnes-pratiques-pour-assurer-la-mise-en-production)
- **MAP (Mean Average Precision)** : moyenne sur plusieurs requêtes de la précision moyenne pondérée par le rang. [blog.octo](https://blog.octo.com/evaluation-rag--bonnes-pratiques-pour-assurer-la-mise-en-production)

Ces métriques permettent d’ajuster le **chunking**, la **taille de la base**, le **modèle d’embeddings** et le **reclassement** (cross‑encoder). [ibm](https://www.ibm.com/fr-fr/think/topics/rag-techniques)

### Métriques liées à la génération

La réponse doit être à la fois factuellement correcte, cohérente et alignée avec le contexte. On utilise :  
- **Fidélité (faithfulness / answer correctness)** : mesure dans quelle mesure la réponse est cohérente avec les documents récupérés, sans hallucinations. [louisbouchard](https://www.louisbouchard.ca/blog-ia/evaluer-rag)
- **PERTINENCE de la réponse (answer relevance)** : évaluation de la qualité sémantique par rapport à la question (souvent avec des outils comme RAGAS ou BERTScore). [blog.octo](https://blog.octo.com/evaluation-rag--bonnes-pratiques-pour-assurer-la-mise-en-production)
- **BERTScore** : compare la réponse générée et une référence (ou le contexte) via des embeddings de BERT, en tenant compte des synonymes et reformulations. [blog.octo](https://blog.octo.com/evaluation-rag--bonnes-pratiques-pour-assurer-la-mise-en-production)

### Cadres et bonnes pratiques

- Des frameworks spécialisés comme **RAGAS** permettent d’automatiser l’évaluation sur plusieurs dimensions (fidélité, pertinence, robustesse, toxicité, etc.). [lemagit](https://www.lemagit.fr/conseil/Comment-evaluer-un-systeme-RAG)
- En production, on combine métriques automatiques et **évaluation humaine** (A/B tests, notations sur la clarté, exhaustivité, ton) pour ajuster prompts, filtres et architectures. [modalbgroup](https://www.modalbgroup.com/actualites/rag-production-patterns-evaluation-couts)


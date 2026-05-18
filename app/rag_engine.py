"""Moteur de recherche RAG - Qdrant Cloud Inference (dense + sparse + hybride)."""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

# Mapping collection -> modele dense
COLLECTIONS_CONFIG = {
    "tickets": "sentence-transformers/all-MiniLM-L6-v2",
    "tickets_e5": "intfloat/multilingual-e5-small",
}

DEFAULT_COLLECTION = "tickets_e5"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    cloud_inference=True,
    timeout=60,
)


def _build_filter(type_=None, queue=None, priority=None, tag=None):
    """Construit un filtre Qdrant a partir des parametres optionnels."""
    conditions = []
    if type_:
        conditions.append(
            models.FieldCondition(key="type", match=models.MatchValue(value=type_))
        )
    if queue:
        conditions.append(
            models.FieldCondition(key="queue", match=models.MatchValue(value=queue))
        )
    if priority:
        conditions.append(
            models.FieldCondition(key="priority", match=models.MatchValue(value=priority))
        )
    if tag:
        conditions.append(
            models.FieldCondition(key="tags", match=models.MatchValue(value=tag))
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def search(query_text, method="hybrid", limit=10, collection=None, **filter_kwargs):
    """Recherche unifiee : dense, sparse, hybride ou hybrid_rerank.

    Args:
        query_text: texte de la requete
        method: "dense", "sparse", "hybrid" ou "hybrid_rerank"
        limit: nombre de resultats
        collection: nom de la collection ("tickets", "tickets_e5"). Defaut: tickets
        **filter_kwargs: type_, queue, priority, tag (tous optionnels)

    Returns:
        liste de resultats avec score et payload
    """
    col = collection or DEFAULT_COLLECTION
    dense_model = COLLECTIONS_CONFIG[col]
    qf = _build_filter(**filter_kwargs)

    common = dict(
        collection_name=col,
        limit=limit,
        query_filter=qf,
    )

    if method == "dense":
        response = client.query_points(
            query=models.Document(text=query_text, model=dense_model),
            using="dense",
            **common,
        )
    elif method == "sparse":
        response = client.query_points(
            query=models.Document(text=query_text, model=SPARSE_MODEL),
            using="bm25",
            **common,
        )
    elif method == "hybrid_rerank":
        response = client.query_points(
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query_text, model=dense_model),
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=models.Document(text=query_text, model=SPARSE_MODEL),
                    using="bm25",
                    limit=20,
                ),
            ],
            query=models.Document(text=query_text, model=COLBERT_MODEL),
            using="colbert",
            **common,
        )
    else:
        response = client.query_points(
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query_text, model=dense_model),
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=models.Document(text=query_text, model=SPARSE_MODEL),
                    using="bm25",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            **common,
        )

    return [
        {
            "score": pt.score,
            "subject": pt.payload.get("subject") or "",
            "body": pt.payload.get("body", ""),
            "answer": pt.payload.get("answer", ""),
            "type": pt.payload.get("type", ""),
            "queue": pt.payload.get("queue", ""),
            "priority": pt.payload.get("priority", ""),
            "tags": pt.payload.get("tags", []),
        }
        for pt in response.points
    ]


def get_filter_options(collection=None):
    """Recupere les valeurs distinctes pour les filtres depuis la collection."""
    col = collection or DEFAULT_COLLECTION
    info = client.get_collection(col)
    count = info.points_count

    # Scroll un echantillon pour extraire les valeurs uniques
    records, _ = client.scroll(collection_name=col, limit=min(count, 500))

    types = sorted({r.payload["type"] for r in records if r.payload.get("type")})
    queues = sorted({r.payload["queue"] for r in records if r.payload.get("queue")})
    priorities = sorted({r.payload["priority"] for r in records if r.payload.get("priority")})
    tags = sorted({t for r in records for t in r.payload.get("tags", [])})

    return {
        "types": types,
        "queues": queues,
        "priorities": priorities,
        "tags": tags,
    }

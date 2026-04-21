"""Moteur de recherche RAG - Qdrant Cloud Inference (dense + sparse + hybride)."""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION = "tickets_test"  # Nom de la collection Qdrant a utiliser
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    cloud_inference=True,
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


def search(query_text, method="hybrid", limit=10, **filter_kwargs):
    """Recherche unifiee : dense, sparse ou hybride avec filtres optionnels.

    Args:
        query_text: texte de la requete
        method: "dense", "sparse" ou "hybrid"
        limit: nombre de resultats
        **filter_kwargs: type_, queue, priority, tag (tous optionnels)

    Returns:
        liste de resultats avec score et payload
    """
    qf = _build_filter(**filter_kwargs)

    common = dict(
        collection_name=COLLECTION,
        limit=limit,
        query_filter=qf,
    )

    if method == "dense":
        response = client.query_points(
            query=models.Document(text=query_text, model=DENSE_MODEL),
            using="dense",
            **common,
        )
    elif method == "sparse":
        response = client.query_points(
            query=models.Document(text=query_text, model=SPARSE_MODEL),
            using="bm25",
            **common,
        )
    else:
        response = client.query_points(
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query_text, model=DENSE_MODEL),
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


def get_filter_options():
    """Recupere les valeurs distinctes pour les filtres depuis la collection."""
    info = client.get_collection(COLLECTION)
    count = info.points_count

    # Scroll un echantillon pour extraire les valeurs uniques
    records, _ = client.scroll(collection_name=COLLECTION, limit=min(count, 500))

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

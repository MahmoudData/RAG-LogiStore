"""
Ingestion des tickets EN dans Qdrant (dense + sparse via Cloud Inference).

Usage:
    python app/ingestion.py              # Collection 'tickets' (MiniLM)
    python app/ingestion.py --model e5   # Collection 'tickets_e5' (E5 multilingual)
    python app/ingestion.py --model e5 --colbert  # E5 + ColBERT reranker
"""

import os
import ast
import argparse
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

# Configurations des modeles disponibles
MODELS_CONFIG = {
    "minilm": {
        "collection": "tickets",
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dense_size": 384,
    },
    "e5": {
        "collection": "tickets_e5",
        "dense_model": "intfloat/multilingual-e5-small",
        "dense_size": 384,
    },
}

SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"
COLBERT_SIZE = 96
BATCH_SIZE = 50


def run(model_key="minilm", colbert=False):
    config = MODELS_CONFIG[model_key]
    collection = config["collection"]
    dense_model = config["dense_model"]
    dense_size = config["dense_size"]

    print(f"Modele : {dense_model}")
    print(f"Collection : {collection}")
    if colbert:
        print(f"ColBERT reranker : {COLBERT_MODEL}")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        cloud_inference=True,
    )

    # Chargement dataset
    df = pd.read_csv("data/processed/tickets_clean.csv")
    df["tags"] = df["tags"].apply(ast.literal_eval)
    df_en = df[df["language"] == "en"].reset_index(drop=True)
    total = len(df_en)
    print(f"Tickets EN a ingerer : {total}")

    # Creation collection + index
    if client.collection_exists(collection):
        client.delete_collection(collection)

    vectors_config = {
        "dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
    }
    if colbert:
        vectors_config["colbert"] = models.VectorParams(
            size=COLBERT_SIZE,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        )

    client.create_collection(
        collection_name=collection,
        vectors_config=vectors_config,
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(),
        },
    )

    for field in ["type", "queue", "priority", "tags"]:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    print(f"Collection '{collection}' creee avec index.")

    # Ingestion par batch
    for start in range(0, total, BATCH_SIZE):
        batch = df_en.iloc[start:start + BATCH_SIZE]
        points = []

        for i, row in batch.iterrows():
            vector = {
                "dense": models.Document(text=row["document"], model=dense_model),
                "bm25": models.Document(text=row["document"], model=SPARSE_MODEL),
            }
            if colbert:
                vector["colbert"] = models.Document(
                    text=row["document"], model=COLBERT_MODEL
                )

            points.append(
                models.PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "subject": row["subject"],
                        "body": row["body"],
                        "answer": row["answer"],
                        "type": row["type"],
                        "queue": row["queue"],
                        "priority": row["priority"],
                        "tags": row["tags"],
                    },
                )
            )

        client.upsert(collection_name=collection, points=points)
        print(f"  {min(start + BATCH_SIZE, total)}/{total}", end="\r")

    info = client.get_collection(collection)
    print(f"\nIngestion terminee : {info.points_count} points dans '{collection}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion tickets dans Qdrant")
    parser.add_argument(
        "--model",
        choices=list(MODELS_CONFIG.keys()),
        default="minilm",
        help="Modele d'embedding dense (default: minilm)",
    )
    parser.add_argument(
        "--colbert",
        action="store_true",
        help="Ajouter le vecteur ColBERT pour reranking",
    )
    args = parser.parse_args()
    run(model_key=args.model, colbert=args.colbert)

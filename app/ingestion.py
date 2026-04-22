"""
Ingestion des tickets EN dans Qdrant (dense + sparse via Cloud Inference).
python app/ingestion.py pour lancer l'ingestion.
"""

import os
import ast
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION = "tickets"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
BATCH_SIZE = 50


def run():
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
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(),
        },
    )

    for field in ["type", "queue", "priority", "tags"]:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    print(f"Collection '{COLLECTION}' creee avec index.")

    # Ingestion par batch
    for start in range(0, total, BATCH_SIZE):
        batch = df_en.iloc[start:start + BATCH_SIZE]
        points = []

        for i, row in batch.iterrows():
            points.append(
                models.PointStruct(
                    id=i,
                    vector={
                        "dense": models.Document(text=row["document"], model=DENSE_MODEL),
                        "bm25": models.Document(text=row["document"], model=SPARSE_MODEL),
                    },
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

        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  {min(start + BATCH_SIZE, total)}/{total}", end="\r")

    info = client.get_collection(COLLECTION)
    print(f"\nIngestion terminee : {info.points_count} points dans '{COLLECTION}'")


if __name__ == "__main__":
    run()

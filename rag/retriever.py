import chromadb


def retrieve(query: str, collection: chromadb.Collection, k: int = 4) -> list[dict]:
    if collection.count() == 0:
        return []

    n = min(k, collection.count())
    results = collection.query(query_texts=[query], n_results=n)

    docs = results["documents"][0]
    # cosine space: distance = 1 - cosine_similarity; clamp to [0, 1] for normalized embeddings
    distances = results["distances"][0]

    return [
        {"text": doc, "score": round(max(0.0, 1.0 - dist), 4)}
        for doc, dist in zip(docs, distances)
    ]

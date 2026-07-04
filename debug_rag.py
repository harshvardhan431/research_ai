"""
debug_rag.py -- run this to see exactly what's in your ChromaDB collections
and what distance scores real queries produce.

Usage:
    python debug_rag.py
"""

import chromadb
from gemini_embedding_function import GeminiEmbeddingFunction

client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = GeminiEmbeddingFunction(task_type="RETRIEVAL_QUERY")

faculty_collection = client.get_or_create_collection(
    "faculty", metadata={"hnsw:space": "cosine"}
)
papers_collection = client.get_or_create_collection(
    "papers", metadata={"hnsw:space": "cosine"}
)

print("=" * 60)
print("COLLECTION COUNTS")
print("=" * 60)
print(f"faculty collection: {faculty_collection.count()} items")
print(f"papers collection:  {papers_collection.count()} items")

if faculty_collection.count() > 0:
    sample = faculty_collection.peek(limit=2)
    print("\n--- sample faculty doc (first 200 chars) ---")
    for doc in sample["documents"]:
        print(doc[:200])
        print("...")

if papers_collection.count() > 0:
    sample = papers_collection.peek(limit=2)
    print("\n--- sample papers doc (first 200 chars) ---")
    for doc in sample["documents"]:
        print(doc[:200])
        print("...")

print("\n" + "=" * 60)
print("TEST QUERY")
print("=" * 60)

test_query = input("\nEnter a test question (something you KNOW is in your files): ")

query_embedding = embedding_fn([test_query])

for name, coll in [("faculty", faculty_collection), ("papers", papers_collection)]:
    if coll.count() == 0:
        print(f"\n[{name}] collection is empty, skipping query")
        continue

    results = coll.query(
        query_embeddings=query_embedding,
        n_results=4,
        include=["documents", "distances", "metadatas"],
    )

    print(f"\n[{name}] top matches:")
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        print("  No results returned at all.")
        continue

    for doc, dist, meta in zip(docs, distances, metadatas):
        print(f"  distance={dist:.4f}  source={meta.get('source', '?')}")
        print(f"  preview: {doc[:120]}...")
        print()
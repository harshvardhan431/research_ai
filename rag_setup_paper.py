import os
import chromadb
from gemini_embedding_function import GeminiEmbeddingFunction

DATA_DIR = "papers_txt"
CHUNK_SIZE = 1000       # characters per chunk -- papers are dense, so a bit bigger than faculty profiles
CHUNK_OVERLAP = 150     # overlap so we don't cut sentences/ideas in half at chunk boundaries

client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")

# no embedding_function passed here -- we embed manually below and pass
# raw vectors to collection.add(), same pattern as rag_setup_faculty.py
#
# ✅ FIXED: forced cosine distance (Chroma defaults to L2, which produces
# unbounded huge distances for high-dim Gemini embeddings and silently
# breaks any relevance threshold). Must match rag_setup_faculty.py and main.py.
collection = client.get_or_create_collection(
    name="papers",
    metadata={"hnsw:space": "cosine"},
)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def ingest():
    if not os.path.isdir(DATA_DIR):
        print(f"{DATA_DIR} folder missing")
        return

    existing_ids = set(collection.get()["ids"])
    total = 0

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".txt"):
            continue

        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text)

        documents = []
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file}_{i}"
            if chunk_id in existing_ids:
                continue  # already ingested, skip

            documents.append(chunk)
            ids.append(chunk_id)
            metadatas.append({
                "source": file,   # matches the "source" key main.py reads
                "file": file,
                "chunk_index": i,
            })

        if not documents:
            print(f"{file}: 0 new chunks (already ingested or empty)")
            continue

        embeddings = embedding_fn(documents)

        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        total += len(documents)
        print(f"{file}: {len(documents)} chunks added")

    print(f"\nTotal paper chunks added: {total}")


if __name__ == "__main__":
    ingest()
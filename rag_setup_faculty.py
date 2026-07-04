import os
import re
import chromadb
from gemini_embedding_function import GeminiEmbeddingFunction

DATA_DIR = "faculty_txt"

client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")

# ✅ FIXED: no embedding_function passed here, since we embed manually below
# and pass raw vectors to collection.add(). Passing it here too was a
# conflict risk (it requires GeminiEmbeddingFunction to match Chroma's
# exact interface, e.g. a name() method on newer versions).
#
# ✅ FIXED (root cause of "RAG never triggers"): Chroma defaults to L2
# distance, which is unbounded and huge for high-dimensional, non-normalized
# Gemini embeddings (3072-dim). That made every distance blow past any
# sane threshold, so retrieval always looked "irrelevant" even for perfect
# matches. Forcing cosine distance here keeps values in a predictable
# 0 (identical) to 2 (opposite) range that a threshold can actually work with.
collection = client.get_or_create_collection(
    name="faculty",
    metadata={"hnsw:space": "cosine"},
)


def split_faculty_profiles(text):
    """
    Splits text into individual professor profiles using the ===PROFILE===
    delimiter. Each profile is expected to contain a "NAME: ..." field
    somewhere in its text, which is extracted separately for metadata/IDs.
    """
    # split on the literal delimiter, tolerating surrounding whitespace/newlines
    raw_chunks = re.split(r"={2,}\s*PROFILE\s*={2,}", text, flags=re.IGNORECASE)

    profiles = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # pull the name out of "NAME: <name> DESIGNATION: ..." (case-insensitive,
        # stops at the next ALL-CAPS field label or end of line)
        match = re.search(
            r"NAME:\s*(.+?)\s*(?:DESIGNATION:|RESEARCH:|PHD:|$)",
            chunk,
            flags=re.IGNORECASE,
        )
        name = match.group(1).strip() if match else None

        if not name:
            print(f"  Warning: couldn't extract a name from chunk, skipping: {chunk[:80]}...")
            continue

        profiles.append((name, chunk))

    return profiles


def ingest():
    if not os.path.isdir(DATA_DIR):
        print("faculty_txt folder missing")
        return

    total = 0

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".txt"):
            continue

        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            text = f.read()

        profiles = split_faculty_profiles(text)

        documents = []
        ids = []
        metadatas = []

        for i, (name, content) in enumerate(profiles):
            doc = f"FACULTY PROFILE\n{content}"

            documents.append(doc)
            ids.append(f"{file}_{name}_{i}")
            metadatas.append({
                "source": name,   # NOTE: matches the "source" key main.py expects
                "name": name,
                "file": file
            })

        if not documents:
            print(f"{file}: 0 profiles added")
            continue

        # ✅ embed manually (no conflict with collection's embedding function)
        embeddings = embedding_fn(documents)

        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        total += len(documents)
        print(f"{file}: {len(documents)} profiles added")

    print(f"\nTotal faculty profiles: {total}")


if __name__ == "__main__":
    ingest()
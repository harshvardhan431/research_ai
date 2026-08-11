"""
rag_setup_faculty.py
Parses faculty_txt/faculty_profiles.txt using the ===PROFILE=== delimiter
format (NAME: / DESIGNATION: / RESEARCH: per block), and ingests each
professor as one chunk into a separate "faculty" collection in ChromaDB.

Expected file format:

    ===PROFILE===
    NAME: sutapa roy rahmanan
    DESIGNATION: Senior Professor. PhD: Jadavpur University.
    RESEARCH: Nano-biomaterials for labeling and drug delivery, ...

    ===PROFILE===
    NAME: srinivas ramaswami
    ...

To add a new professor: add another ===PROFILE=== block with the same
three fields, then re-run this script.
"""

import os
import re
import chromadb
from gemini_embedding_function import GeminiEmbeddingFunction

FACULTY_DIR = "faculty_txt"

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    "faculty", embedding_function=GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")
)


def parse_profiles(text):
    """
    Splits text on ===PROFILE=== markers and extracts NAME/DESIGNATION/
    RESEARCH fields from each block. Returns a list of dicts.
    """
    blocks = text.split("===PROFILE===")
    profiles = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        name_match = re.search(r"NAME:\s*(.+)", block)
        designation_match = re.search(r"DESIGNATION:\s*(.+)", block)
        research_match = re.search(r"RESEARCH:\s*(.+)", block, re.DOTALL)

        if not name_match:
            # This is likely the FORMAT/instructions block at the top of
            # the file, not an actual profile -- skip it.
            continue

        name = name_match.group(1).strip()
        designation = designation_match.group(1).strip() if designation_match else ""
        research = research_match.group(1).strip() if research_match else ""

        # Include the name explicitly in the chunk text (not just as
        # metadata), so a query like "tell me about Dr. X" retrieves
        # correctly on the name itself, not just the research keywords.
        full_text = f"{name}\n{designation}\n{research}"

        profiles.append({
            "name": name,
            "designation": designation,
            "research": research,
            "text": full_text,
        })

    return profiles


def ingest():
    if not os.path.isdir(FACULTY_DIR):
        print(f"Folder '{FACULTY_DIR}' not found. Create it and add faculty_profiles.txt first.")
        return

    total_chunks = 0
    for filename in os.listdir(FACULTY_DIR):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(FACULTY_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        profiles = parse_profiles(text)
        file_id = filename.replace(".txt", "")

        for profile in profiles:
            safe_id = re.sub(r"[^a-zA-Z0-9]+", "_", profile["name"]).strip("_").lower()

            collection.add(
                documents=[profile["text"]],
                ids=[f"{file_id}_{safe_id}"],
                metadatas=[{
                    "name": profile["name"],
                    "designation": profile["designation"],
                    "file": filename,
                }],
            )
            total_chunks += 1

        print(f"Ingested {filename} -- {len(profiles)} faculty profiles")

    print(f"\nTotal faculty profiles ingested: {total_chunks}")


if __name__ == "__main__":
    ingest()
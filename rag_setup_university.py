"""
rag_setup_university.py
Parses university_txt/university.txt using the ===UNIVERSITY=== delimiter
format (NAME: / LOCATION: / RESEARCH_STRENGTHS: / NOTES: per block), and
ingests each university as one chunk into a separate "university"
collection in ChromaDB.

Same pattern as rag_setup_faculty.py -- kept as its own script and its
own collection because universities are a list of distinct entities, not
sections of one document, same reasoning that split faculty from papers.

To add a new university: add another ===UNIVERSITY=== block with the same
four fields, then re-run this script.
"""

import os
import re
import chromadb
from gemini_embedding_function import GeminiEmbeddingFunction

UNIVERSITY_DIR = "university_txt"

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    "university", embedding_function=GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")
)


def parse_universities(text):
    """
    Splits text on ===UNIVERSITY=== markers and extracts NAME/LOCATION/
    RESEARCH_STRENGTHS/NOTES fields from each block. Returns a list of dicts.
    """
    blocks = text.split("===UNIVERSITY===")
    universities = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        name_match = re.search(r"NAME:\s*(.+)", block)
        location_match = re.search(r"LOCATION:\s*(.+)", block)
        strengths_match = re.search(r"RESEARCH_STRENGTHS:\s*(.+)", block)
        notes_match = re.search(r"NOTES:\s*(.+)", block, re.DOTALL)

        if not name_match:
            # This is the FORMAT/instructions block at the top -- skip it.
            continue

        name = name_match.group(1).strip()
        location = location_match.group(1).strip() if location_match else ""
        strengths = strengths_match.group(1).strip() if strengths_match else ""
        notes = notes_match.group(1).strip() if notes_match else ""

        # Include the name explicitly in the chunk text, same reasoning as
        # faculty -- so "tell me about IIT Bombay" retrieves correctly on
        # the name itself, not just on research-area keywords.
        full_text = f"{name}\nLocation: {location}\nResearch strengths: {strengths}\n{notes}"

        universities.append({
            "name": name,
            "location": location,
            "strengths": strengths,
            "text": full_text,
        })

    return universities


def ingest():
    if not os.path.isdir(UNIVERSITY_DIR):
        print(f"Folder '{UNIVERSITY_DIR}' not found. Create it and add university.txt first.")
        return

    total_chunks = 0
    for filename in os.listdir(UNIVERSITY_DIR):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(UNIVERSITY_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        universities = parse_universities(text)
        file_id = filename.replace(".txt", "")

        for uni in universities:
            safe_id = re.sub(r"[^a-zA-Z0-9]+", "_", uni["name"]).strip("_").lower()

            collection.add(
                documents=[uni["text"]],
                ids=[f"{file_id}_{safe_id}"],
                metadatas=[{
                    "name": uni["name"],
                    "location": uni["location"],
                    "file": filename,
                }],
            )
            total_chunks += 1

        print(f"Ingested {filename} -- {len(universities)} universities")

    print(f"\nTotal universities ingested: {total_chunks}")


if __name__ == "__main__":
    ingest()
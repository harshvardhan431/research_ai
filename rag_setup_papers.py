"""
rag_setup.py
Reads all .txt files from papers_txt/, splits each into sections using
ALL-CAPS header lines (TITLE, ABSTRACT, PROBLEM, etc.), and ingests
each section as one chunk into ChromaDB.

To add a new paper: just drop a new .txt file into papers_txt/ using the
same header style (see attention_is_all_you_need.txt as the template)
and re-run this script.
"""

import os
import re
import chromadb
from gemini_embedding_function import GeminiEmbeddingFunction

PAPERS_DIR = "papers_txt"

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    "papers", embedding_function=GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")
)


def parse_paper(text):
    """Split a paper's text into (section_name, section_text) chunks
    using lines that are short and fully uppercase as headers."""
    lines = text.split("\n")
    sections = []
    current_header = "META"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        is_header = (
            stripped
            and stripped.isupper()
            and len(stripped.split()) <= 6
            and not stripped.startswith("TITLE:")
            and not stripped.startswith("AUTHORS:")
            and not stripped.startswith("VENUE:")
            and not stripped.startswith("YEAR:")
        )
        if is_header:
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))

    return sections


def extract_meta(text):
    title = re.search(r"TITLE:\s*(.+)", text)
    authors = re.search(r"AUTHORS:\s*(.+)", text)
    year = re.search(r"YEAR:\s*(.+)", text)
    return {
        "title": title.group(1).strip() if title else "Unknown",
        "authors": authors.group(1).strip() if authors else "Unknown",
        "year": year.group(1).strip() if year else "Unknown",
    }


def ingest():
    if not os.path.isdir(PAPERS_DIR):
        print(f"Folder '{PAPERS_DIR}' not found. Create it and add .txt papers first.")
        return

    total_chunks = 0
    for filename in os.listdir(PAPERS_DIR):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(PAPERS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        meta = extract_meta(text)
        sections = parse_paper(text)
        paper_id = filename.replace(".txt", "")

        for section_name, section_text in sections:
            if not section_text or section_name == "META":
                continue
            collection.add(
                documents=[section_text],
                ids=[f"{paper_id}_{section_name}"],
                metadatas=[{
                    "source": meta["title"],
                    "authors": meta["authors"],
                    "year": meta["year"],
                    "section": section_name,
                    "file": filename,
                }],
            )
            total_chunks += 1

        print(f"Ingested {filename} — {meta['title']} ({len(sections)} sections)")

    print(f"\nTotal chunks ingested: {total_chunks}")


if __name__ == "__main__":
    ingest()
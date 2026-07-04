"""
gemini_embedding_function.py
A ChromaDB-compatible embedding function that uses Gemini's embedding API
instead of Chroma's default local ONNX model (all-MiniLM-L6-v2).

Why: the default ONNX model can fail to initialize on some machines/Python
versions ("bad allocation" errors from onnxruntime). Using Gemini's hosted
embedding API sidesteps that entirely -- no local model download, no
onnxruntime dependency at all.

Use this SAME embedding function object everywhere you create or query
a collection (ingestion scripts AND app.py), or ChromaDB will complain
about mismatched embedding dimensions.

✅ IMPORTANT: gemini-embedding-001 supports a task_type parameter that
meaningfully changes the embedding space depending on use case. Without
it, retrieval quality is poor -- queries and their matching documents
don't cluster together well, because the embedding isn't optimized for
"is this query answered by this document" style matching.

Use task_type="RETRIEVAL_DOCUMENT" when embedding documents (ingestion),
and task_type="RETRIEVAL_QUERY" when embedding the user's question
(retrieval). These are asymmetric on purpose -- a query and its best
matching document are NOT expected to look identical, just related.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, task_type: str = "RETRIEVAL_DOCUMENT"):
        """
        task_type: "RETRIEVAL_DOCUMENT" when embedding content to be stored
                   and searched over (use this in ingestion scripts).
                   "RETRIEVAL_QUERY" when embedding a user's question
                   (use this in main.py at query time).
        """
        self.task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        result = _client.models.embed_content(
            model="gemini-embedding-001",
            contents=input,
            config=types.EmbedContentConfig(task_type=self.task_type),
        )
        return [e.values for e in result.embeddings]
"""
main.py -- Streamlit Research Matching Chatbot
Flow: Query -> search faculty_txt (ChromaDB) -> if no good match -> Tavily web search
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError
import chromadb
from tavily import TavilyClient
from gemini_embedding_function import GeminiEmbeddingFunction

# ---- CONFIG ----
CHROMA_PATH = "./chroma_db"
FACULTY_COLLECTION_NAME = "faculty"    # must match rag_setup_faculty.py
PAPERS_COLLECTION_NAME = "papers"      # must match rag_setup_paper.py

# ✅ TUNED based on real debug_rag.py numbers:
#   - genuine matches landed around 0.42-0.44
#   - a totally unrelated query landed around 0.48
# That's a narrow gap (embedding models often cluster everything closer
# together than intuition suggests -- this is a known effect, not a bug).
# 0.46 sits between the two. If Tavily still never triggers, tighten this
# further (e.g. 0.44); if real matches start getting rejected, loosen it
# slightly (e.g. 0.47). Re-test with debug_rag.py after any change.
DISTANCE_THRESHOLD = 0.46

# NOTE: double check this model name against Google's current model list.
# "gemma-4-31b-it" does not match any known released Gemini/Gemma model as of
# my last update -- likely typo for something like "gemini-1.5-flash" or
# "gemini-2.0-flash". Fix this before running.
MODEL_NAME = "gemma-4-31b-it"

# ---- INIT ----
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# metadata={"hnsw:space": "cosine"} must match what the ingestion scripts
# used to CREATE these collections. It only takes effect at creation time,
# but keeping it here documents the requirement and is harmless if the
# collection already exists.
faculty_collection = chroma_client.get_or_create_collection(
    FACULTY_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)
papers_collection = chroma_client.get_or_create_collection(
    PAPERS_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)
embedding_fn = GeminiEmbeddingFunction(task_type="RETRIEVAL_QUERY")


# NOTE: Ingestion is handled separately:
#   python rag_setup_faculty.py   -> populates "faculty" collection
#   python rag_setup_paper.py     -> populates "papers" collection
# Run both once (or whenever the source .txt files change) BEFORE starting this app.

# ---- RETRIEVAL: GENERIC (used for both collections) ----
def retrieve_from_collection(collection, query, n=4):
    """Returns (context_string, found: bool, sources: list) for a single collection."""
    query_embedding = embedding_fn([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        include=["documents", "distances", "metadatas"],
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return "", False, []

    relevant = [
        (doc, meta.get("source", "unknown"), dist)
        for doc, dist, meta in zip(docs, distances, metadatas)
        if dist < DISTANCE_THRESHOLD
    ]

    if not relevant:
        return "", False, []

    # sort best-match first across whichever collection we're querying
    relevant.sort(key=lambda x: x[2])

    context = "\n---\n".join(doc for doc, _, _ in relevant)
    sources = list({src for _, src, _ in relevant})
    return context, True, sources


# ---- RETRIEVAL: FACULTY + PAPERS, BEST MATCH WINS ----
def retrieve_from_rag(query, n=4):
    """
    Checks both the faculty and papers collections.
    Returns (context_string, found: bool, sources: list, which: str)
    """
    faculty_context, faculty_found, faculty_sources = retrieve_from_collection(
        faculty_collection, query, n
    )
    papers_context, papers_found, papers_sources = retrieve_from_collection(
        papers_collection, query, n
    )

    if faculty_found and papers_found:
        # both matched -- combine, since the question might touch both
        combined = f"{faculty_context}\n---\n{papers_context}"
        return combined, True, faculty_sources + papers_sources, "faculty+papers"

    if faculty_found:
        return faculty_context, True, faculty_sources, "faculty"

    if papers_found:
        return papers_context, True, papers_sources, "papers"

    return "", False, [], "none"


# ---- RETRIEVAL: TAVILY FALLBACK ----
def retrieve_from_tavily(query, max_results=4):
    try:
        response = tavily_client.search(query=query, max_results=max_results)
        results = response.get("results", [])
        if not results:
            return "", []

        context = "\n---\n".join(
            f"{r.get('title', 'Untitled')}: {r.get('content', '')}" for r in results
        )
        sources = [r.get("url", "") for r in results]
        return context, sources

    except Exception as e:
        st.warning(f"Tavily search failed: {e}")
        return "", []


# ---- COMBINED RETRIEVAL ----
def get_context(query):
    """RAG (faculty + papers) first, Tavily as fallback. Returns (context, source_label, sources)."""
    context, found, sources, which = retrieve_from_rag(query)
    if found:
        return context, which, sources

    context, sources = retrieve_from_tavily(query)
    if context:
        return context, "tavily", sources

    return "", "none", []


# ---- BUILD CONTEXT STRING FROM INTAKE ----
def build_search_string(sq):
    return (
        f"Topic: {sq['topic']}. "
        f"Desired depth: {sq['analysis_depth']}. "
        f"Goal: {sq['goal']}. "
        f"Response style: {sq['response_depth']}."
    )


# ---- SAFE LLM CALL ----
def get_response(messages, retries=3):
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
        for m in messages[-8:]  # limit history sent to the model
    ]

    for i in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
            )
            return response.text

        except ServerError:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


# ---- SESSION STATE ----
if "stage" not in st.session_state:
    st.session_state.stage = "intro"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "loading" not in st.session_state:
    st.session_state.loading = False

if faculty_collection.count() == 0 and papers_collection.count() == 0:
    st.warning(
        "Both 'faculty' and 'papers' collections are empty. Run "
        "`python rag_setup_faculty.py` and `python rag_setup_paper.py` "
        "to ingest your data first."
    )
elif faculty_collection.count() == 0:
    st.info("'faculty' collection is empty — run `python rag_setup_faculty.py`.")
elif papers_collection.count() == 0:
    st.info("'papers' collection is empty — run `python rag_setup_paper.py`.")

st.title("🎓 Research Assistant Bot")


# ---- STAGE 1: INTRO ----
if st.session_state.stage == "intro":
    st.write("Answer 5 quick questions so I can tailor my research help to you.")
    if st.button("Test Yourself"):
        st.session_state.stage = "intake"
        st.rerun()


# ---- STAGE 2: INTAKE ----
elif st.session_state.stage == "intake":
    st.subheader("Research Qualification Guide")

    with st.form("intake_form"):
        topic = st.text_input("1. Topic?")

        response_depth = st.radio(
            "2. Response type?",
            ["Quick answer", "Detailed, research-backed explanation"],
        )

        source_preference = st.radio(
            "3. Sources?",
            ["I have papers", "Search from scratch", "Use both"],
        )

        goal = st.radio(
            "4. Goal?",
            ["Assignment", "Course project", "Publication", "General understanding"],
        )

        analysis_depth = st.radio(
            "5. Depth?",
            ["Basic", "Technical", "Critical analysis"],
        )

        submitted = st.form_submit_button("Start Chat")

    if submitted:
        if not topic.strip():
            st.warning("Enter a topic.")
        else:
            structured_query = {
                "topic": topic,
                "response_depth": response_depth,
                "source_preference": source_preference,
                "goal": goal,
                "analysis_depth": analysis_depth,
            }

            context_string = build_search_string(structured_query)

            st.session_state.messages = [
                {
                    "role": "user",
                    "content": f"Context: {context_string}. Use this internally.",
                },
                {
                    "role": "model",
                    "content": "Understood. Ask your question.",
                },
            ]

            st.session_state.stage = "chat"
            st.rerun()


# ---- STAGE 3: CHAT ----
elif st.session_state.stage == "chat":

    for msg in st.session_state.messages[2:]:
        st.chat_message(
            "user" if msg["role"] == "user" else "assistant"
        ).write(msg["content"])

    user_input = st.chat_input("Ask something...")

    if user_input and not st.session_state.loading:
        st.session_state.loading = True

        try:
            st.chat_message("user").write(user_input)

            context, source, sources = get_context(user_input)

            if context:
                prompt = f"""
Context:
{context}

Question: {user_input}
Answer using the context above and cite sources where relevant.
"""
            else:
                prompt = user_input

            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
            })

            reply = get_response(st.session_state.messages)

            st.session_state.messages.append({
                "role": "model",
                "content": reply,
            })

            st.chat_message("assistant").write(reply)

            if source in ("faculty", "papers", "faculty+papers"):
                st.caption(f"📄 Answered from {source} — sources: {', '.join(sources)}")
            elif source == "tavily":
                st.caption("🌐 Answered from web search (Tavily)")
            else:
                st.caption("⚠️ No matching context found — general knowledge answer")

        except Exception as e:
            st.error(f"Error: {str(e)}")

        finally:
            st.session_state.loading = False
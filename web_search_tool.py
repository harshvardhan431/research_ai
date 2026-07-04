"""
web_search_tool.py
Tavily-based web search tool, used for Step 3 (live research trends,
professor lookups outside the faculty DB, etc).

Tavily is built specifically for LLM agents -- it returns clean,
pre-summarized snippets instead of raw search-engine HTML, so you don't
need extra cleanup before feeding results to Gemini.
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_search(query, max_results=5):
    """
    Runs a Tavily search and returns a clean text block ready to drop
    into a Gemini prompt as context.
    """
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",   # use "advanced" for deeper/slower searches
    )

    results = response.get("results", [])
    if not results:
        return "No results found."

    formatted = []
    for r in results:
        formatted.append(f"{r['title']}: {r['content']} (source: {r['url']})")

    return "\n\n".join(formatted)


if __name__ == "__main__":
    # Quick manual test
    print(web_search("latest trends in NLP research 2026"))
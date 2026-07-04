"""
query_intake.py -- Step 2.5 (pre-retrieval qualification)

Runs 5 quick multiple-choice questions in the terminal BEFORE the main
chatbot loop touches RAG/tools. Answers get combined into one structured
query dict, which is what actually gets embedded for similarity search --
not the user's raw first message.

This replaces the research_qualification_guide.txt approach: instead of
that content living as a RAG chunk a user could accidentally retrieve,
it's real code that runs as a pipeline step.
"""


def ask_choice(question, options):
    """Prints a numbered question, loops until user picks a valid option."""
    print(f"\n{question}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")

    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice, try again.")


def get_structured_query():
    """
    Runs the 5-question intake flow and returns a structured dict:
    {
        "topic": str,
        "response_depth": str,
        "source_preference": str,
        "goal": str,
        "analysis_depth": str,
    }
    """
    print("=" * 50)
    print("Before we search, a few quick questions:")
    print("=" * 50)

    # Q1 - topic (free text, not multiple choice -- can't predict every topic)
    topic = input("\nWhat is the specific topic or problem you want to research?\n> ").strip()

    # Q2 - response depth
    response_depth = ask_choice(
        "Are you looking for a quick answer or a detailed research-backed explanation?",
        ["Quick answer", "Detailed, research-backed explanation"],
    )

    # Q3 - source preference
    source_preference = ask_choice(
        "Do you have source material (papers/PDFs/links), or should I retrieve from scratch?",
        ["I have papers/PDFs/links to share", "Search and retrieve from scratch", "Use both"],
    )

    # Q4 - goal
    goal = ask_choice(
        "What is your goal with this research?",
        ["Assignment", "Course project", "Publication or thesis", "General understanding"],
    )

    # Q5 - analysis depth
    analysis_depth = ask_choice(
        "How deep should the analysis go?",
        ["Basic overview", "Technical breakdown", "Critical analysis (gaps and limitations)"],
    )

    structured_query = {
        "topic": topic,
        "response_depth": response_depth,
        "source_preference": source_preference,
        "goal": goal,
        "analysis_depth": analysis_depth,
    }

    print("\n" + "=" * 50)
    print("Got it. Structured query:")
    for k, v in structured_query.items():
        print(f"  {k}: {v}")
    print("=" * 50 + "\n")

    return structured_query


def build_search_string(structured_query):
    """
    Flattens the structured query into a single string suitable for
    embedding / similarity search, weighting topic as the main signal
    and folding the rest in as context.
    """
    return (
        f"Topic: {structured_query['topic']}. "
        f"Desired depth: {structured_query['analysis_depth']}. "
        f"Goal: {structured_query['goal']}. "
        f"Response style: {structured_query['response_depth']}."
    )


if __name__ == "__main__":
    # Quick manual test
    sq = get_structured_query()
    print("Search string for embedding:")
    print(build_search_string(sq))
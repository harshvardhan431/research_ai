# AI Research & University Assistant

An AI-powered chatbot designed to assist users with **research papers, university-related information, faculty information, and general queries**. The system combines **Retrieval-Augmented Generation (RAG)** with web search to provide relevant, context-aware responses.

## Features

* **Research Paper Assistant**

  * Answers questions related to research papers and academic content.
  * Retrieves relevant information from indexed research documents.
  * Helps users understand and explore academic material.

* **Faculty RAG**

  * Retrieves faculty-related information from the university knowledge base.
  * Can answer questions about faculty members, departments, research interests, and related information.

* **University RAG**

  * Provides university-specific information using a dedicated knowledge base.
  * Useful for answering questions about university facilities, departments, programs, and other institutional information.

* **General AI Assistant**

  * Handles questions that are not available in the internal knowledge bases.
  * Uses **Tavily** for web search to retrieve up-to-date information.

* **Multiple Knowledge Bases**

  * Uses separate RAG pipelines for different domains.
  * Allows the chatbot to retrieve information from the most relevant source.

* **Context-Aware Responses**

  * Combines retrieved information with an LLM to generate natural-language responses.

## System Architecture

```text
                         ┌──────────────────┐
                         │      User        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   AI Chatbot     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Query Processing │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             Internal Knowledge           General Query
                  Retrieval                    │
                    │                         ▼
        ┌───────────┼───────────┐         Tavily Search
        │           │           │              │
        ▼           ▼           ▼              │
   Research RAG  Faculty RAG  University RAG  │
        │           │           │              │
        └───────────┴───────────┘              │
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                         ┌──────────────────┐
                         │       LLM        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Response     │
                         └──────────────────┘
```

## RAG Components

### 1. Research Paper RAG

The research paper knowledge base stores academic documents that can be retrieved based on the user's query.

Example queries:

```text
"Explain the methodology used in this research paper."

"What are the key findings of this paper?"

"Summarize the proposed approach."
```

### 2. Faculty RAG

The faculty knowledge base contains information related to university faculty.

Example queries:

```text
"Who is working on machine learning?"

"Which faculty members are from the CSE department?"

"What are Professor X's research interests?"
```

### 3. University RAG

The university knowledge base contains institutional information.

Example queries:

```text
"What departments are available?"

"What facilities does the university provide?"

"Tell me about the CSE program."
```

### 4. General Web Search

When the required information is not available in the internal knowledge bases, the chatbot can use **Tavily** to search the web.

Example:

```text
"What are the latest developments in generative AI?"
```

The system retrieves relevant web information and uses it as context for generating the final answer.

## Tech Stack

* **Python**
* **LLM**
* **Retrieval-Augmented Generation (RAG)**
* **Tavily Search API**
* **Vector Database**
* **Embeddings**
* **Python Web Framework / API Layer**
* **University Knowledge Base**

> Update this section with the exact technologies you used, such as Flask/FastAPI, ChromaDB, FAISS, Gemini, OpenAI, Ollama, LangChain, etc.

## Project Structure

```text
project/
│
├── app/
│   ├── rag/
│   │   ├── research_rag/
│   │   ├── faculty_rag/
│   │   ├── university_rag/
│   │   └── other_rag/
│   │
│   ├── chatbot/
│   ├── retrieval/
│   └── utils/
│
├── data/
│   ├── research_papers/
│   ├── faculty/
│   └── university/
│
├── requirements.txt
├── .env
├── main.py
└── README.md
```

*The structure above is an example. Modify it according to your actual repository.*

## How It Works

1. The user enters a question.
2. The system analyzes the query.
3. The appropriate knowledge source is selected.
4. Relevant documents are retrieved using RAG.
5. If the query requires external information, Tavily performs a web search.
6. Retrieved information is provided as context to the LLM.
7. The LLM generates the final response.
8. The response is returned to the user.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
TAVILY_API_KEY=your_tavily_api_key
LLM_API_KEY=your_llm_api_key
```

Add any other API keys required by your implementation.

**Never commit your `.env` file to GitHub.**

Add it to `.gitignore`:

```text
.env
venv/
__pycache__/
```

## Running the Application

Start the application using:

```bash
python main.py
```

If you are using Flask:

```bash
flask run
```

If you are using FastAPI:

```bash
uvicorn main:app --reload
```

Use the command that matches your actual implementation.

## Example Queries

### Research

```text
"Summarize this research paper."

"What methodology does the paper use?"

"What are the limitations of this research?"
```

### Faculty

```text
"Which faculty members specialize in AI?"

"Who is the faculty coordinator for this department?"
```

### University

```text
"What courses are offered by the university?"

"What facilities are available for students?"
```

### General Knowledge

```text
"What is RAG?"

"What are the latest developments in AI?"
```

## Key Concepts Used

### Retrieval-Augmented Generation

RAG allows the chatbot to retrieve relevant information from external knowledge sources before generating a response.

Instead of relying entirely on the LLM's pretrained knowledge:

```text
User Query
     ↓
Retrieve Relevant Documents
     ↓
Provide Documents as Context
     ↓
LLM
     ↓
Generated Answer
```

This helps the chatbot answer questions about **domain-specific information that may not be present in the LLM's training data**.

### Tavily

Tavily is used as a web-search component for queries requiring external or current information.

This allows the chatbot to combine:

```text
University Knowledge
        +
Research Knowledge
        +
Faculty Knowledge
        +
Web Search
        ↓
      LLM
        ↓
   Final Answer
```

## Future Improvements

* Add citation support for research-paper responses.
* Improve query routing between different RAG systems.
* Add conversation memory.
* Add document upload functionality.
* Add authentication and role-based access.
* Add evaluation metrics for RAG quality.
* Improve hallucination detection.
* Add source attribution for retrieved information.
* Deploy the application using Docker and a cloud platform.

## Use Cases

This chatbot can be used as:

* University information assistant
* Research assistant
* Faculty information assistant
* Academic question-answering system
* General-purpose AI assistant
* University knowledge management system

## Disclaimer

The accuracy of generated responses depends on the quality of the retrieved documents, web-search results, embeddings, and underlying language model. Users should verify important academic or institutional information against authoritative sources.

## Author

**Harsh Vardhan Singh Chouhan**

Computer Science Student
Interested in Backend Development, AI, RAG Systems, and Data Engineering.
 

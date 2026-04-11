# Agentic RAG Pipeline with LangGraph

This repository contains an autonomous RAG (Retrieval-Augmented Generation) agent built with **LangGraph**. It features dynamic routing, standalone query rewriting, document reranking, and a self-correction loop to ensure grounded, hallucination-free answers.

## Key Features

* **Adaptive Routing**: Uses an LLM router to decide between local PDF knowledge (RAG), general conversation (Chat), or ending the session.
* **Context-Aware Rewriting**: Analyzes chat history to rewrite follow-up questions into standalone search queries.
* **Multi-Stage Retrieval**: Combines ChromaDB vector search with a Cohere-powered reranking stage for high-precision context.
* **Self-Correction Loop**: Validates the generated answer against the context. If the answer is not grounded or context is missing, the agent automatically triggers a **Tavily Web Search** fallback.
* **Structured Logging**: Professional observability using `Loguru` to track state transitions and LLM decisions in real-time.

---

## Graph

<img width="428" height="940" alt="image" src="https://github.com/user-attachments/assets/fb791214-2ac4-42ec-981f-274a5215dbde" />

## Tech Stack

- **Orchestration**: LangGraph
- **LLM**: Grok-4.1-fast (via OpenRouter)
- **Embeddings**: Qwen3-embedding-8b
- **Vector Store**: ChromaDB
- **Search API**: Tavily
- **Reranker**: Cohere Rerank-4-fast

---

## Project Structure

```text
├── main.py             # Graph orchestration and interactive chat loop
├── nodes.py            # Logic for each node (Retrieve, Generate, Search, etc.)
├── schema.py           # Pydantic models and AgentState TypedDict
├── config.py           # Centralized configuration and hyperparameters
├── load_documents.py   # Utility to ingest, chunk, and embed PDFs
├── .env                # Environment variables (API Keys)
└── chroma_db/          # Persistent vector database storage
```

---

## Quick Start

### 1. Setup Environment
Clone the repository and install dependencies:
```bash
pip install langgraph langchain_openai langchain_chroma loguru tavily-python pydantic-settings pymupdf
```

Create a `.env` file in the root directory:
```text
OPENROUTER_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

### 2. Ingest Knowledge base
Place your PDF in the project directory and update `config.py` with your filename, then run:
```bash
python load_documents.py
```

### 3. Run the Agent
Start the interactive session:
```bash
python main.py
```

---

## Pipeline Logic

The agent follows a sophisticated state machine flow:
1.  **Context Check**: Determines if history is needed.
2.  **Rewrite**: Generates a standalone query.
3.  **Router**: Selects the path (RAG vs. Chat).
4.  **Retrieve & Rerank**: Fetches the most relevant chunks from ChromaDB.
5.  **Generate**: Drafts an answer strictly cited from context.
6.  **Self-Check**: 
    * If **Grounded**: Returns answer to user.
    * If **Not Grounded/No Docs**: Triggers Web Search and regenerates.
    * If **Max Retries Reached**: Gracefully fails.

---

## Observability
The project uses `Loguru` for clean, color-coded console output. This allows developers to see exactly why the agent chose a specific route or why a grounding check failed during execution.

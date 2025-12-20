# DI-502: Financial AI Assistant

**DI 502 Project Repository** — An advanced RAG-based Chatbot & Market Researcher for answering finance questions.

This project combines a high-performance **FastAPI** backend with a modern **React** frontend to provide real-time financial insights, document analysis (RAG), and live market research.

---

## Features

### Backend (Python/FastAPI)
* **Async Streaming API**: Delivers AI responses token-by-token (typewriter effect) using Server-Sent Events (SSE).
* **RAG (Retrieval-Augmented Generation)**: Analyze user-uploaded documents (PDF, TXT) to answer context-specific questions.
* **Online Market Research**: Fetches real-time financial news and data (e.g., via Finnhub) to answer current market queries.
* **Llama 3 Integration**: Optimized for financial reasoning using Llama 3 (via Ollama or local quantization).
* **Robust Testing**: Comprehensive `pytest` suite with async generator mocking.

### Frontend (React/Vite)
* **Modern UI**: Built with React and Vite for a fast, responsive experience.
* **Smart Composer**: Supports file uploads, online research toggling, and prompt suggestions (chips).
* **Markdown Rendering**: Renders tables, lists, and code blocks beautifully using `react-markdown`.
* **Dark/Light Mode**: Fully themable interface with persistence.
* **Interactive Controls**: Includes a **"Stop Generating"** feature to cancel requests mid-stream.

---

## Project Structure

The repository is organized into a clear separation of concerns between Client and Server.

```text
.
├── backend/
│   ├── src/              # Source code (rag_service, online_search, etc.)
│   ├── uploads/          # Temporary storage for RAG documents
│   ├── tests/            # Pytest suite (test_main.py)
│   ├── main.py           # FastAPI application entrypoint
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── src/              # React source (App.jsx, components, hooks)
│   ├── src/__tests__/    # Frontend tests (Vitest)
│   ├── public/           # Static assets
│   ├── package.json      # Node dependencies
│   └── vite.config.js    # Vite configuration
├── firebase.json         # Firebase hosting configuration
└── README.md
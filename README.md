# DI-502: Financial AI Assistant

A RAG-based financial chatbot with real-time market research capabilities. Built with FastAPI backend and React frontend.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Backend Setup](#backend-setup)
4. [Frontend Setup](#frontend-setup)
5. [Chat Modes](#chat-modes)
6. [Online Query System](#online-query-system)
7. [Memory System](#memory-system)
8. [API Reference](#api-reference)
9. [Environment Variables](#environment-variables)

---

## Architecture Overview

```
Frontend (React/Vite)  <-->  Backend (FastAPI)  <-->  Ollama (LLM)
        |                          |                       |
   localhost:5173             localhost:8000          localhost:11434
                                   |
                    +-----------------------------+
                    |    External APIs            |
                    |  - Finnhub (stocks/news)    |
                    |  - CoinGecko (crypto)       |
                    |  - Yahoo Finance (commodities)|
                    +-----------------------------+
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama with llama3 model installed
- Conda (recommended) or virtualenv

### Install Ollama and Model

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the LLM model
ollama pull llama3
```

---

## Backend Setup

### 1. Create and Activate Environment

```bash
cd backend
conda create -n economind python=3.10
conda activate economind
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the `backend` directory:

```env
FINNHUB_API_KEY=your_finnhub_api_key
NEWS_API_KEY=your_newsapi_key          # Optional
ALPHA_VANTAGE_KEY=your_alpha_key       # Optional
OLLAMA_MODEL=llama3                    # Default model
```

### 4. Start the Backend Server

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Documentation at `/docs`.

---

## Frontend Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The application will be available at `http://localhost:5173`.

### 3. Build for Production

```bash
npm run build
npm run preview
```

---

## Chat Modes

The application supports three distinct chat modes:

### 1. Plain Chat Mode

Direct conversation with the LLM without external data fetching.

- Use Case: General financial questions, explanations, educational content
- Trigger: Default mode when no document is uploaded and online research is disabled
- Features: Conversation memory, context-aware responses

### 2. Document Mode (RAG)

Retrieval-Augmented Generation using uploaded documents.

- Use Case: Analyzing SEC filings, financial reports, research papers
- Trigger: Upload a document (PDF, TXT) via the attachment button
- Features: Vector-based semantic search, document-grounded answers
- Supported Formats: PDF, TXT, DOCX

### 3. Online Research Mode

Real-time market data and news fetching with intelligent query analysis.

- Use Case: Current stock prices, market news, crypto prices, commodity data
- Trigger: Enable the globe icon toggle in the composer
- Features: LLM-based intent detection, multi-source data aggregation

---

## Online Query System

The online research mode uses a sophisticated pipeline:

### Query Intent Analysis

The system uses LLM-first intent detection to classify queries:

| Intent Type | Example Query | Data Fetched |
|-------------|---------------|--------------|
| `price_lookup` | "What is NVIDIA stock price?" | Real-time quote |
| `price_analysis` | "Will Bitcoin increase?" | Price + News |
| `news_query` | "Latest Tesla news" | Company news |
| `market_analysis` | "How is the crypto market?" | Market news |
| `company_info` | "Tell me about Apple" | Company profile |

### Supported Asset Classes

Stocks: 50+ major tickers mapped (NVDA, AAPL, TSLA, etc.)

Cryptocurrencies: BTC, ETH, SOL, ADA, XRP, DOGE, and 10+ more

Commodities: Gold (XAU), Silver (XAG), Platinum, Oil, Natural Gas

### Data Sources

| Source | Data Type | API Key Required |
|--------|-----------|------------------|
| Finnhub | Stock quotes, company profiles, news | Yes |
| CoinGecko | Cryptocurrency prices | No |
| Yahoo Finance | Commodity/futures prices | No |
| NewsAPI | Extended news coverage | Optional |

### Conversation-Aware Intent

The system maintains conversation context for follow-up questions:

```
User: "Give me NVIDIA price and technical analysis"
Assistant: [Provides comprehensive NVIDIA analysis]

User: "What about Tesla?"
Assistant: [Automatically applies same analysis type to Tesla]
```

---

## Memory System

Session-based conversation memory using LlamaIndex ChatMemoryBuffer.

### Features

- Per-session message storage
- Automatic session cleanup (1 hour timeout)
- Token-limited context window (3000 tokens default)
- Thread-safe operations

### Debug Endpoint

View session memory contents:

```
GET /session/{session_id}/debug
```

Response includes all stored messages with roles and content.

---

## API Reference

### Main Endpoints

#### POST /chat

Main chat endpoint with streaming response.

Form Parameters:
- `question` (required): User query
- `use_online_research` (optional): Enable online mode
- `document` (optional): File upload for RAG
- `session_id` (optional): Session identifier

Response: Server-Sent Events stream with `[SESSION_ID:xxx]` prefix

#### GET /session/{session_id}

Get session information.

#### GET /session/{session_id}/debug

Get full conversation history for debugging.

#### DELETE /session/{session_id}

Clear session memory.

#### POST /session/new

Create new session and return ID.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FINNHUB_API_KEY` | Yes | Finnhub API key for stock data |
| `NEWS_API_KEY` | No | NewsAPI.org key for extended news |
| `ALPHA_VANTAGE_KEY` | No | Alpha Vantage key (unused) |
| `OLLAMA_MODEL` | No | Ollama model name (default: llama3) |

---

## Project Structure

```
DI-502/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   └── src/
│       ├── rag_service.py      # Core RAG and query logic
│       └── memory_manager.py   # Session memory management
├── frontend/
│   ├── src/
│   │   └── App.jsx             # Main React component
│   ├── package.json            # Node dependencies
│   └── vite.config.js          # Vite configuration
├── data/
│   └── datasets/               # Sample financial datasets
└── tests/
    └── backend_test.py         # Backend test suite
```

---

## Testing

### Backend Tests

```bash
cd tests
pip install -r requirements.txt
pytest backend_test.py -v
```

### Frontend Tests

```bash
cd frontend
npm run test
```

---

## License

This project is developed for DI-502 coursework.

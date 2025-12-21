# Financial AI Assistant - Frontend

The client-side interface for the DI-502 Project. This is a modern, single-page React application built with Vite, designed to interact with the FastAPI backend via streaming responses.

## ✨ Features

* **Streaming Chat Interface**: Renders AI responses in real-time (typewriter effect).
* **Markdown Support**: Displays financial data in tables, lists, and formatted text using `react-markdown` and `remark-gfm`.
* **Smart Composer**:
    * File uploads for RAG analysis.
    * Online Research toggle (Global Search).
    * **"Stop Generating"** button to cancel requests.
* **Theme System**: Persistent Dark/Light mode with custom CSS variables.
* **Robust Testing**: Unit and integration tests using **Vitest** and **React Testing Library**.

## 🛠️ Tech Stack

* **Framework**: React 18 + Vite
* **Styling**: Plain CSS (Modern variable-based architecture)
* **Icons**: SVG Icons (Lucide/Custom)
* **Markdown**: `react-markdown`, `remark-gfm`
* **Testing**: `vitest`, `@testing-library/react`, `jsdom`

---

## 🚀 Getting Started

### 1. Installation

Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
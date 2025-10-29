# DI-502
DI 502 Project Repository — RAG-based Chatbot for Answering Finance Questions

## Features

* **FastAPI Backend**: A modern, high-performance web framework for building APIs.
* **Pydantic Validation**: Robust data validation for API requests and responses.
* **Dockerized**: Fully containerized with a multi-stage `Dockerfile` for small, efficient images.
* **Interactive Docs**: Automatic API documentation provided by Swagger UI and ReDoc.
* **Clear Structure**: A clean project structure separating backend, frontend, and data science concerns.

---
## Project Structure

The repository is organized to support independent development of the backend, frontend, and data science components.

```
.
├── backend.Dockerfile
├── .venv/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── data/
│   ├── datasets/
│   └── notebooks/
├── frontend/
└── README.md
```

---
## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

### Prerequisites

You'll need the following software installed on your machine:
* [Python 3.12+](https://www.python.org/downloads/)
* [Docker](https://www.docker.com/products/docker-desktop/)
* [Git](https://git-scm.com/)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GulbeycanCagri/DI-502.git
    cd DI-502
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install backend dependencies:**
    ```bash
    python -m pip install -r backend/requirements.txt
    ```

---
## Usage

You can run the application in two ways: directly on your local machine for development or within a Docker container.

### Method 1: Running Locally (for Development)

1.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

2.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

3.  **Start the Uvicorn server:**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

### Method 2: Running with Docker

1.  **Build the Docker image:**
    From the project's **root directory**, run:
    ```bash
    docker build -t chatbot-backend -f backend.Dockerfile .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -d --name chatbot-api -p 8000:8000 chatbot-backend
    ```
    The API will be available at `http://127.0.0.1:8000`.

---
## API Endpoints

The API provides the following endpoints. Test them at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Description                                       |
| :----- | :------- | :------------------------------------------------ |
| `GET`  | `/`      | Health check to confirm the API is running.       |
| `POST` | `/chat`  | Main endpoint for sending a query to the chatbot. |

---
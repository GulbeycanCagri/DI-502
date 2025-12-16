import sys
import io
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ==========================================
# CRITICAL STEP: MOCKING PROCESS
# ==========================================
mock_rag = MagicMock()

# Define the canned responses for the mock
mock_rag.plain_chat.return_value = "Test response from Ollama"
mock_rag.query_online.return_value = "Test online research response"
mock_rag.query_document.return_value = "Test document research response"

sys.modules["backend.src.rag_service"] = mock_rag
import backend.main as main_mod
from backend.main import app

CLIENT = TestClient(app)
UPLOAD_FOLDER = Path(main_mod.UPLOAD_FOLDER).resolve()


def setup_function(function):
    # Enable test mode so endpoint uses test stubs if needed
    main_mod.TEST_MODE = True
    # Ensure upload folder exists
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    # Snapshot existing files to clean up only newly created ones
    setup_function._existing_files = set(p.name for p in UPLOAD_FOLDER.iterdir())


def teardown_function(function):
    # Disable test mode after each test
    main_mod.TEST_MODE = False
    
    # Remove any new files created during tests
    existing = setup_function._existing_files
    if UPLOAD_FOLDER.exists():
        for p in UPLOAD_FOLDER.iterdir():
            if p.name not in existing:
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                except Exception:
                    pass


def test_root_endpoint():
    resp = CLIENT.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data


def test_plain_chat_endpoint():
    # POST a simple form with question only
    resp = CLIENT.post("/chat", data={"question": "What is a 10-K report?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "ai_response" in data
    # Verify the response matches our mock
    assert data["ai_response"] == "Test response from Ollama"


def test_online_research_endpoint():
    # POST with use_online_research=true
    resp = CLIENT.post(
        "/chat",
        data={"question": "Is NVIDIA hiring?", "use_online_research": "true"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ai_response" in data
    assert data["ai_response"] == "Test online research response"


def test_file_upload_endpoint():
    # Prepare an in-memory dummy PDF file
    file_content = b"%PDF-1.4\n% Dummy PDF content\n"
    file_obj = io.BytesIO(file_content)
    files = {"document": ("report.pdf", file_obj, "application/pdf")}
    
    resp = CLIENT.post(
        "/chat",
        data={"question": "Summarize this document."},
        files=files,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ai_response" in data
    assert data["ai_response"] == "Test document research response"

    # Verify that a file was actually created in the upload folder
    created_files = [p for p in UPLOAD_FOLDER.iterdir() if p.name not in setup_function._existing_files]
    assert len(created_files) >= 1


def test_missing_question_returns_422():
    # Missing 'question' field should result in a 422 Unprocessable Entity (FastAPI default)
    resp = CLIENT.post("/chat", data={})
    
    # FastAPI doğrulama hataları için 422 döndürür, 400 değil.
    assert resp.status_code == 422
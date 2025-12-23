"""
Backend API Tests - Tests for FastAPI endpoints

NOTE: This file is named z_backend_test.py to ensure it loads alphabetically
AFTER test_rag_service.py. This is critical because this file mocks
backend.src.rag_service at module load time, which would break tests in
test_rag_service.py that need the real module.
"""
import io
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient


async def mock_async_generator(*args, **kwargs):
    yield "Mocked "
    yield "Response"


# Store original module if it exists
_original_rag_service = sys.modules.get("backend.src.rag_service")

mock_rag = MagicMock()
mock_rag.plain_chat.side_effect = mock_async_generator
mock_rag.query_online.side_effect = mock_async_generator
mock_rag.query_document.side_effect = mock_async_generator

sys.modules["backend.src.rag_service"] = mock_rag

import backend.main as main_mod
from backend.main import app

CLIENT = TestClient(app)


def restore_rag_service():
    """Restore the original rag_service module after tests."""
    if _original_rag_service is not None:
        sys.modules["backend.src.rag_service"] = _original_rag_service
    elif "backend.src.rag_service" in sys.modules:
        del sys.modules["backend.src.rag_service"]
UPLOAD_FOLDER = Path(main_mod.UPLOAD_FOLDER).resolve()


def setup_function(function):
    main_mod.TEST_MODE = True
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    setup_function._existing_files = set(p.name for p in UPLOAD_FOLDER.iterdir())


def teardown_function(function):
    main_mod.TEST_MODE = False

    if UPLOAD_FOLDER.exists():
        for p in UPLOAD_FOLDER.iterdir():
            if p.name not in getattr(setup_function, "_existing_files", set()):
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                except Exception:
                    pass


# ===========================================
# Basic Endpoint Tests
# ===========================================


def test_root_endpoint():
    resp = CLIENT.get("/")
    assert resp.status_code == 200
    assert resp.json() == {
        "message": "Welcome to the RAG API. Visit /docs or /redoc to see the API documentation."
    }


def test_plain_chat_endpoint():
    resp = CLIENT.post("/chat", data={"question": "What is a 10-K report?"})
    assert resp.status_code == 200
    assert "[SESSION_ID:" in resp.text
    assert "Mocked Response" in resp.text
    assert "text/event-stream" in resp.headers["content-type"]


def test_online_research_endpoint():
    # Online research
    resp = CLIENT.post(
        "/chat",
        data={"question": "Is NVIDIA hiring?", "use_online_research": "true"},
    )
    assert resp.status_code == 200
    assert "[SESSION_ID:" in resp.text
    assert "Mocked Response" in resp.text


def test_file_upload_endpoint():
    file_content = b"%PDF-1.4\n% Dummy PDF content\n"
    file_obj = io.BytesIO(file_content)
    files = {"document": ("report.pdf", file_obj, "application/pdf")}

    resp = CLIENT.post(
        "/chat",
        data={"question": "Summarize this document."},
        files=files,
    )

    assert resp.status_code == 200
    assert "[SESSION_ID:" in resp.text
    assert "Mocked Response" in resp.text


def test_missing_question_returns_422():
    resp = CLIENT.post("/chat", data={})

    assert resp.status_code == 422


# ===========================================
# Session Management Tests
# ===========================================


def test_create_new_session():
    """Test creating a new conversation session."""
    resp = CLIENT.post("/session/new")
    assert resp.status_code == 201
    data = resp.json()
    assert "session_id" in data
    assert data["message"] == "New session created successfully."
    assert len(data["session_id"]) == 36  # UUID format


def test_get_session_info_existing():
    """Test getting info for an existing session."""
    # First create a session
    create_resp = CLIENT.post("/session/new")
    session_id = create_resp.json()["session_id"]

    # Get session info
    resp = CLIENT.get(f"/session/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert data["exists"] is True


def test_get_session_info_nonexistent():
    """Test getting info for a non-existent session."""
    resp = CLIENT.get("/session/nonexistent-session-id")
    assert resp.status_code == 200
    data = resp.json()
    assert data["exists"] is False
    assert data["message_count"] == 0


def test_clear_session_existing():
    """Test clearing an existing session."""
    # Create a session first
    create_resp = CLIENT.post("/session/new")
    session_id = create_resp.json()["session_id"]

    # Clear the session
    resp = CLIENT.delete(f"/session/{session_id}")
    assert resp.status_code == 200
    assert "cleared successfully" in resp.json()["message"]


def test_clear_session_nonexistent():
    """Test clearing a non-existent session."""
    resp = CLIENT.delete("/session/nonexistent-session-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["message"]


def test_chat_with_session_id():
    """Test chat endpoint with an existing session ID."""
    # Create a session
    create_resp = CLIENT.post("/session/new")
    session_id = create_resp.json()["session_id"]

    # Use it in chat
    resp = CLIENT.post("/chat", data={"question": "Hello", "session_id": session_id})
    assert resp.status_code == 200


def test_chat_auto_generates_session():
    """Test that chat creates a session if none provided."""
    resp = CLIENT.post("/chat", data={"question": "Hello without session"})
    assert resp.status_code == 200
    # Session ID should be in the response stream
    assert "[SESSION_ID:" in resp.text or "Mocked" in resp.text


# ===========================================
# Online Research Tests
# ===========================================


def test_online_research_with_ticker():
    """Test online research for a specific stock."""
    resp = CLIENT.post(
        "/chat",
        data={
            "question": "What's happening with NVIDIA stock?",
            "use_online_research": "true",
        },
    )
    assert resp.status_code == 200


def test_online_research_crypto():
    """Test online research for cryptocurrency."""
    resp = CLIENT.post(
        "/chat",
        data={
            "question": "What's the current Bitcoin price?",
            "use_online_research": "true",
        },
    )
    assert resp.status_code == 200


def test_online_research_general_finance():
    """Test online research for general finance questions."""
    resp = CLIENT.post(
        "/chat",
        data={
            "question": "What's happening in the stock market today?",
            "use_online_research": "true",
        },
    )
    assert resp.status_code == 200


# ===========================================
# File Upload Tests
# ===========================================


def test_upload_txt_file():
    """Test uploading a text file."""
    file_content = b"This is a test financial report content."
    file_obj = io.BytesIO(file_content)
    files = {"document": ("report.txt", file_obj, "text/plain")}

    resp = CLIENT.post(
        "/chat",
        data={"question": "Summarize this."},
        files=files,
    )
    assert resp.status_code == 200


def test_upload_with_online_research_disabled():
    """Test that file upload takes priority when online research is disabled."""
    file_content = b"Financial data content here."
    file_obj = io.BytesIO(file_content)
    files = {"document": ("data.txt", file_obj, "text/plain")}

    resp = CLIENT.post(
        "/chat",
        data={"question": "Analyze this", "use_online_research": "false"},
        files=files,
    )
    assert resp.status_code == 200


# ===========================================
# Error Handling Tests
# ===========================================


def test_empty_question_string():
    """Test that empty question string returns error."""
    resp = CLIENT.post("/chat", data={"question": ""})
    # FastAPI will reject empty required field
    assert resp.status_code in [400, 422]


def test_special_characters_in_question():
    """Test handling of special characters in question."""
    resp = CLIENT.post(
        "/chat", data={"question": "What about $NVDA & <script>alert('xss')</script>?"}
    )
    assert resp.status_code == 200


def test_very_long_question():
    """Test handling of very long questions."""
    long_question = "What is the stock price? " * 100  # Very long question
    resp = CLIENT.post("/chat", data={"question": long_question})
    assert resp.status_code == 200

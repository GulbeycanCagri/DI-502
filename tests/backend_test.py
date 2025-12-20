import sys
import io
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


async def mock_async_generator(*args, **kwargs):
    yield "Mocked "
    yield "Response"

mock_rag = MagicMock()
mock_rag.plain_chat.side_effect = mock_async_generator
mock_rag.query_online.side_effect = mock_async_generator
mock_rag.query_document.side_effect = mock_async_generator

sys.modules["backend.src.rag_service_2"] = mock_rag

import backend.main as main_mod
from backend.main import app

CLIENT = TestClient(app)
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


def test_root_endpoint():
    resp = CLIENT.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Welcome to the RAG API. Visit /docs or /redoc to see the API documentation."}


def test_plain_chat_endpoint():

    resp = CLIENT.post("/chat", data={"question": "What is a 10-K report?"})
    assert resp.status_code == 200
    assert resp.text == "Mocked Response"
    assert "text/event-stream" in resp.headers["content-type"]


def test_online_research_endpoint():
    # Online research
    resp = CLIENT.post(
        "/chat",
        data={"question": "Is NVIDIA hiring?", "use_online_research": "true"},
    )
    assert resp.status_code == 200
    assert resp.text == "Mocked Response"


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
    assert resp.text == "Mocked Response"


def test_missing_question_returns_422():
    resp = CLIENT.post("/chat", data={})
    
    assert resp.status_code == 422
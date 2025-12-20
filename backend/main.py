import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Import added
from fastapi.responses import StreamingResponse
from werkzeug.utils import secure_filename

# --- Path Correction
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent  # Folder where 'main.py' is (e.g., .../backend)
parent_root = project_root.parent  # Parent folder (e.g., .../DI-502)

# Add both paths (if they are not already present)
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(parent_root) not in sys.path:
    sys.path.append(str(parent_root))

from backend.src.rag_service_2 import plain_chat, query_document, query_online

TEST_MODE = False

# --- App Setup ---

app = FastAPI(
    title="RAG API",
    description="Chat with documents, online research, or plain chat.",
    version="1.0.0",
)

# --- CORS Setup ---
origins = [
    # 1. Production Site (Firebase)
    "https://di502-economind.web.app",
    
    # 2. Localhost (To avoid errors during development)
    "http://localhost:5173",  # Vite default
    "http://localhost:3000",  # React default
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # Only allow listed origins
    allow_credentials=True,
    allow_methods=["*"],       # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],       # Allow all headers
)

# Configuration for uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Routes ---

@app.get("/")
async def root():
    """
    Root endpoint. Provides a simple welcome message and directs to the docs.
    """
    return {
        "message": "Welcome to the RAG API. Visit /docs or /redoc to see the API documentation."
    }


@app.post("/chat")
async def chat(
    request: Request,
    question: str = Form(...),
    use_online_research: bool = Form(False),
    document: Optional[UploadFile] = File(None),
):
    """
    Main chat endpoint (Async Streaming Version).
    """
    
    # --- 1. Validation ---
    if not question:
        raise HTTPException(status_code=400, detail="There is no question provided.")

    # --- 2. File Handling (Pre-stream) ---
    saved_file_path = None
    
    if document and document.filename:
        try:
            print(f"--- Document Received: {document.filename} ---")
            base_filename = secure_filename(document.filename)
            unique_filename = f"{uuid.uuid4()}_{base_filename}"
            saved_file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

            print(f"Saving file to: {saved_file_path}")
            async with aiofiles.open(saved_file_path, "wb") as f:
                while chunk := await document.read(1024 * 1024):
                    await f.write(chunk)
            
            await document.close()
            
        except Exception as e:
            print(f"File saving error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while saving the file: {e}",
            )

    # --- 3. Define the Async Generator ---
    async def response_generator():
        try:
            if use_online_research:
                print("--- Performing Online Research (Async Stream) ---")
                async for chunk in query_online(question, test=TEST_MODE):
                    yield chunk

            elif saved_file_path:
                print(f"Querying file: {saved_file_path}")
                async for chunk in query_document(question, saved_file_path, test=TEST_MODE):
                    yield chunk

            else:
                print("--- Plain Chat (Async Stream) ---")
                async for chunk in plain_chat(question, test=TEST_MODE):
                    yield chunk

        except Exception as e:
            print(f"Streaming Error or Disconnect: {e}")
            yield f"\n[SYSTEM ERROR]: {str(e)}"
        
        finally:
            if saved_file_path and os.path.exists(saved_file_path):
                try:
                    os.remove(saved_file_path)
                    print(f"CLEANUP: Deleted file {saved_file_path}")
                except Exception as cleanup_error:
                    print(f"CLEANUP FAILED: {cleanup_error}")

    # --- 4. Return the Stream ---
    return StreamingResponse(response_generator(), media_type="text/event-stream")


# --- Running the Server ---

if __name__ == "__main__":
    print("Starting server at http://0.0.0.0:8000...")
    print(f"Uploads will be saved to the '{UPLOAD_FOLDER}' directory.")
    # 'main:app' refers to the 'app' object in the 'main.py' file
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
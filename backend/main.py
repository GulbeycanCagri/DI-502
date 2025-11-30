import os
import sys
import uuid  # <-- Required: To create unique filenames
from pathlib import Path

import aiofiles
import uvicorn

TEST_MODE = False

# --- Path Correction (More Robust Version) ---
# Add the directory containing 'main.py' and its parent directory to the Python path.
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent  # Folder where 'main.py' is (e.g., .../backend)
parent_root = project_root.parent  # Parent folder (e.g., .../DI-502)

# Add both paths (if they are not already present)
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(parent_root) not in sys.path:
    sys.path.append(str(parent_root))
# --- End Path Correction ---


from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from werkzeug.utils import secure_filename

# Assume the rag_service is in a 'src' directory relative to this file
from backend.src.rag_service import plain_chat, query_document, query_online

# --- App Setup ---

app = FastAPI(
    title="RAG API",
    description="Chat with documents, online research, or plain chat.",
    version="1.0.0",
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
    Main chat endpoint.
    Receives a question and does the following:
    1. If 'use_online_research' is True: Performs online research.
    2. If 'document' is provided: Saves the document and queries it with RAG.
    3. If neither: Returns a plain chat response.
    """

    if not question:
        raise HTTPException(status_code=400, detail="There is no question provided.")

    try:
        if use_online_research:
            print("--- Performing Online Research ---")
            answer = query_online(question, test=TEST_MODE)

        elif document and document.filename:
            print(f"--- Document Received: {document.filename} ---")

            # 1. Create a secure and unique file path
            base_filename = secure_filename(document.filename)
            unique_filename = f"{uuid.uuid4()}_{base_filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

            # 2. Save the file asynchronously to the 'uploads' folder
            try:
                print(f"Saving file to: {file_path}")
                async with aiofiles.open(file_path, "wb") as f:
                    while chunk := await document.read(
                        1024 * 1024
                    ):  # Read in 1MB chunks
                        await f.write(chunk)
            except Exception as e:
                print(f"File saving error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"An error occurred while saving the file: {e}",
                )
            finally:
                await document.close()  # Always close the stream

            # 3. After the file is saved, query it with RAG
            print(f"Querying file: {file_path}")
            answer = query_document(question, file_path, test=TEST_MODE)

        else:
            print("--- Plain Chat ---")
            answer = plain_chat(question, test=TEST_MODE)

        # Return the successful response
        return {"ai_response": answer}

    except HTTPException as e:
        # Re-raise FastAPI's HTTP errors
        raise e
    except Exception as e:
        # Catch all other unexpected errors
        print(f"Server Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected server error occurred: {str(e)}"
        )

    # 'finally' block was removed. The file will remain in the 'uploads' folder.


# --- Running the Server ---

if __name__ == "__main__":
    print("Starting server at http://0.0.0.0:8000...")
    print(f"Uploads will be saved to the '{UPLOAD_FOLDER}' directory.")
    # 'main:app' refers to the 'app' object in the 'main.py' file
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

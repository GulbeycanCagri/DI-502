from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    """
    Defines the structure for a chat request.
    Includes validation to ensure the query is not too long.
    """
    query: str = Field(
        ..., 
        max_length=500, 
        description="The user's query, limited to 500 characters."
    )

class ChatResponse(BaseModel):
    """Defines the structure for a chat response."""
    ai_response: str


# --- FastAPI App ---
app = FastAPI(
    title="Financial Chatbot API",
    description="An API for interacting with a local financial model."
    #version="0.0.1",
)


@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"status": "API is running"}


@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    This dummy endpoint simulates a response from a local model.
    It takes the user's query and returns a fixed response for now.
    """
    #
    # --- LOCAL MODEL LOGIC WILL GO HERE ---
    #
    # Example:
    # model_output = your_model.predict(request.query)
    # return ChatResponse(ai_response=model_output)
    #
    
    # For now, we just return a placeholder.
    return ChatResponse(ai_response="I do not know.")
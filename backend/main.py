from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=500)


class ChatResponse(BaseModel):
    ai_response: str


app = FastAPI(
    title="Financial Chatbot API",
    description="An API for interacting with a local financial model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "API is running"}


@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    return ChatResponse(ai_response=f"Echo: {request.query}")

import datetime
import os
import requests
import pandas as pd
from dotenv import load_dotenv
import asyncio
from typing import AsyncGenerator, Optional, List

# --- LlamaIndex & Ollama Imports ---
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Memory Manager Import
from backend.src.memory_manager import memory_manager

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Ollama
llm = Ollama(
    model="llama3", 
    request_timeout=300.0, 
    temperature=0.1
)

# Embedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024


def _build_chat_messages(
    session_id: Optional[str],
    question: str,
    system_prompt: str
) -> List[ChatMessage]:
    """
    Build chat messages list with conversation history.
    
    Args:
        session_id: Session ID for memory retrieval
        question: Current user question
        system_prompt: System instructions
        
    Returns:
        List of ChatMessage objects for the LLM
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
    ]
    
    # Add conversation history if session exists
    if session_id:
        history = memory_manager.get_chat_history(session_id)
        messages.extend(history)
    
    # Add current user message
    messages.append(ChatMessage(role=MessageRole.USER, content=question))
    
    return messages


async def plain_chat(
    question: str, 
    test: bool = False,
    session_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Plain chat with Ollama (Async Streaming) with conversation memory.
    
    Args:
        question: User's question
        test: Whether to run in test mode
        session_id: Optional session ID for conversation memory
    """
    if test:
        yield "Test response from Ollama"
        return
    
    print(f"--- Performing plain chat with Ollama (Async) | Session: {session_id} ---")
    
    system_prompt = (
        "You are a financial analyst providing investor-focused insights. Based on the conversation context, "
        "identify and explain information material to investment decisions. If relevant information is absent, indicate this. "
        "Do not use context reference labels in your answer. Never use hashtags or emojis. "
        "Remember previous messages in the conversation to provide contextually relevant answers."
    )

    try:
        # Build messages with history
        messages = _build_chat_messages(session_id, question, system_prompt)
        
        # Store user message in memory
        if session_id:
            memory_manager.add_message(session_id, "user", question)
        
        # Stream the response
        full_response = ""
        response_gen = await llm.astream_chat(messages)
        
        async for chunk in response_gen:
            delta = chunk.delta if hasattr(chunk, 'delta') else str(chunk)
            full_response += delta
            yield delta
            await asyncio.sleep(0)
        
        # Store assistant response in memory
        if session_id and full_response:
            memory_manager.add_message(session_id, "assistant", full_response)

    except Exception as e:
        print(f"Ollama Chat Error: {e}")
        yield f"Error: {str(e)}"

def generate_search_query(question: str) -> str:
    print("--- Generating search keywords ---")
    prompt = (
        "You are a search query generator. "
        "Convert the following user question into 3-5 keywords for a financial news search engine. "
        "IMPORTANT: Only output the keywords. Do not add any other text, examples, formatting, or explanations.\n\n"
        f"Question: {question}\nKeywords:"
    )
    try:
        response = llm.complete(prompt)
        keywords = str(response).strip().split("\n")[0].replace('"', "")
        print(f"Generated keywords: {keywords}")
        return keywords
    except Exception as e:
        print(f"Keyword generation failed: {e}. Falling back to raw question.")
        return question

def extract_ticker_from_keywords(question: str) -> str | None:
    lowered_question = question.lower()
    if "nvidia" in lowered_question: return "NVDA"
    if "apple" in lowered_question: return "AAPL"
    if "amd" in lowered_question or "advanced micro devices" in lowered_question: return "AMD"
    if "intel" in lowered_question: return "INTC"
    if "tesla" in lowered_question: return "TSLA"
    if "microsoft" in lowered_question: return "MSFT"
    if "google" in lowered_question or "alphabet" in lowered_question: return "GOOGL"
    if "amazon" in lowered_question: return "AMZN"
    if "meta" in lowered_question or "facebook" in lowered_question: return "META"
    return None


async def query_online(
    question: str, 
    test: bool = False,
    session_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Online research and answer generation (Async Streaming) with conversation memory.
    
    Args:
        question: User's question
        test: Whether to run in test mode
        session_id: Optional session ID for conversation memory
    """
    if test:
        yield "Test online research response"
        return
        
    print(f"--- Performing online research (Async RAG) | Session: {session_id} ---")

    ticker = extract_ticker_from_keywords(question)
    
    try:
        articles = []
        if ticker:
            print(f"Ticker identified: {ticker}.")
            today = datetime.date.today()
            fifteen_days_ago = today - datetime.timedelta(days=15)
            endpoint_url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": fifteen_days_ago.strftime("%Y-%m-%d"),
                "to": today.strftime("%Y-%m-%d"),
                "token": FINNHUB_API_KEY,
            }
            response = requests.get(endpoint_url, params=params)
            response.raise_for_status()
            articles = response.json()[:25]
        else:
            print("No specific ticker.")
            endpoint_url = "https://finnhub.io/api/v1/news"
            params = {"category": "general", "minId": 0, "token": FINNHUB_API_KEY}
            response = requests.get(endpoint_url, params=params)
            response.raise_for_status()
            articles = response.json()[:30]

        if not articles:
            yield "No recent news found via Finnhub."
            return

        documents_list = []
        news_df = pd.DataFrame(columns=["headline", "summary", "source", "url"])

        for article in articles:
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            source_name = article.get("source", "Unknown")
            url = article.get("url", "")

            full_text = f"Source: {source_name}\nHeadline: {headline}\nSummary: {summary}\n"
            news_df = pd.concat([news_df, pd.DataFrame([{
                "headline": headline, "summary": summary, "source": source_name, "url": url
            }])], ignore_index=True)

            if summary and headline:
                doc = Document(
                    text=full_text,
                    metadata={"source_title": headline, "url": url, "source_name": source_name},
                )
                documents_list.append(doc)

        if not documents_list:
            yield "Found articles, but content was empty."
            return

        # Store user message in memory before querying
        if session_id:
            memory_manager.add_message(session_id, "user", question)

        index = VectorStoreIndex.from_documents(documents_list)
        
        # Build query with conversation context
        conversation_context = ""
        if session_id:
            history_str = memory_manager.get_history_as_string(session_id)
            if history_str:
                conversation_context = f"\n\nPrevious conversation:\n{history_str}\n\n"
        
        enhanced_question = f"{conversation_context}Current question: {question}" if conversation_context else question
        
        # Async Query Engine
        query_engine = index.as_query_engine(
            response_mode="compact",
            similarity_top_k=5,
            streaming=True,
        )

        print("Synthesizing answer...")
        streaming_response = await query_engine.aquery(enhanced_question)
        
        full_response = ""
        async for text in streaming_response.async_response_gen():
            full_response += text
            yield text
            await asyncio.sleep(0)
        
        # Store assistant response in memory
        if session_id and full_response:
            memory_manager.add_message(session_id, "assistant", full_response)

    except Exception as e:
        print(f"Error in query_online: {e}")
        yield f"Error: {str(e)}"


async def query_document(
    question: str, 
    doc_path: str, 
    test: bool = False,
    session_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Document-based research and answer generation (Async Streaming) with conversation memory.
    
    Args:
        question: User's question
        doc_path: Path to the document
        test: Whether to run in test mode
        session_id: Optional session ID for conversation memory
    """
    if test:
        yield "Test document research response"
        return
        
    print(f"--- Querying document (Async): {doc_path} | Session: {session_id} ---")

    try:
        # Store user message in memory before querying
        if session_id:
            memory_manager.add_message(session_id, "user", question)
        
        reader = SimpleDirectoryReader(input_files=[doc_path])
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)

        # Build query with conversation context
        conversation_context = ""
        if session_id:
            history_str = memory_manager.get_history_as_string(session_id)
            if history_str:
                conversation_context = f"\n\nPrevious conversation:\n{history_str}\n\n"
        
        enhanced_question = f"{conversation_context}Current question: {question}" if conversation_context else question

        query_engine = index.as_query_engine(
            response_mode="compact",
            similarity_top_k=3,
            streaming=True
        )

        streaming_response = await query_engine.aquery(enhanced_question)
        
        full_response = ""
        async for text in streaming_response.async_response_gen():
            full_response += text
            yield text
            await asyncio.sleep(0)
        
        # Store assistant response in memory
        if session_id and full_response:
            memory_manager.add_message(session_id, "assistant", full_response)

    except Exception as e:
        print(f"Error in query_document: {e}")
        yield f"Error: {str(e)}"
import datetime
import os
import requests
import pandas as pd
from dotenv import load_dotenv
import asyncio

# --- LlamaIndex & Ollama Imports ---
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import AsyncGenerator  # Generator yerine AsyncGenerator

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

DEFAULT_QA_PROMPT_TMPL = (
    "You are a specialized financial analyst. Your task is to answer the user's question "
    "based *only* on the provided context information. Do not use any prior knowledge.\n"
    "---------------------\n"
    "CONTEXT:\n{context_str}\n"
    "---------------------\n"
    "QUESTION: {query_str}\n\n"
    "INSTRUCTIONS:\n"
    "1. Answer the question directly and concisely.\n"
    "2. Synthesize information from all provided sources to form a complete answer.\n"
    "3. **Crucially, clearly attribute claims to their sources** (e.g., 'According to Source Title X...', 'The Motley Fool article argues that...').\n"
    "4. If sources conflict, point out the disagreement (e.g., 'While Source A states..., Source B suggests...').\n"
    "5. If the context does not contain the answer, state that clearly.\n\n"
    "ANSWER (must be in English):\n"
)
qa_template = PromptTemplate(DEFAULT_QA_PROMPT_TMPL)

async def plain_chat(question: str, test: bool = False) -> AsyncGenerator[str, None]:
    """
    Plain chat with Ollama (Async Streaming)
    """
    if test:
        yield "Test response from Ollama"
        return
    
    print("--- Performing plain chat with Ollama (Async) ---")
    
    prompt = (
        "You are a helpful financial analyst. "
        "If the user asks a general question, have a normal conversation. "
        "If the user asks a financial question, answer it concisely. "
        "Never use hashtags or emojis.\n\n"
        f"User: {question}\nAnalyst:"
    )

    try:
        
        response_gen = await llm.astream_complete(prompt)
        async for chunk in response_gen:
            yield chunk.delta
            await asyncio.sleep(0) 

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

async def query_online(question: str, test: bool = False) -> AsyncGenerator[str, None]:
    """
    Online research and answer generation (Async Streaming).
    """
    if test:
        yield "Test online research response"
        return
        
    print("--- Performing online research (Async RAG) ---")

    ticker = extract_ticker_from_keywords(question)
    
    try:
        articles = []
        if ticker:
            print(f"Ticker identified: {ticker}.")
            today = datetime.date.today()
            three_days_ago = today - datetime.timedelta(days=15)
            endpoint_url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": three_days_ago.strftime("%Y-%m-%d"),
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

        index = VectorStoreIndex.from_documents(documents_list)
        
        # Async Query Engine
        query_engine = index.as_query_engine(
            response_mode="compact",
            similarity_top_k=5,
            streaming=True,
        )

        print("Synthesizing answer...")
        streaming_response = await query_engine.aquery(question)
        
        async for text in streaming_response.async_response_gen():
            yield text
            await asyncio.sleep(0) 

    except Exception as e:
        print(f"Error in query_online: {e}")
        yield f"Error: {str(e)}"

async def query_document(question: str, doc_path: str, test: bool = False) -> AsyncGenerator[str, None]:
    """
    Document-based research and answer generation (Async Streaming).
    """
    if test:
        yield "Test document research response"
        return
        
    print(f"--- Querying document (Async): {doc_path} ---")

    try:
        reader = SimpleDirectoryReader(input_files=[doc_path])
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)

        query_engine = index.as_query_engine(
            response_mode="compact",
            similarity_top_k=3,
            streaming=True
        )

        streaming_response = await query_engine.aquery(question)
        async for text in streaming_response.async_response_gen():
            yield text
            await asyncio.sleep(0)

    except Exception as e:
        print(f"Error in query_document: {e}")
        yield f"Error: {str(e)}"
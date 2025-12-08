import datetime
import os
import requests
import pandas as pd
from dotenv import load_dotenv

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

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def plain_chat(question: str, test: bool = False) -> str:
    """
    Normal sohbet fonksiyonu. Doğrudan Ollama'ya sorar.
    """
    if test:
        print("--- Performing plain chat (test mode) ---")
        return "Test response from Ollama"
    
    print("--- Performing plain chat with Ollama ---")
    
    prompt = (
        "You are a helpful financial analyst. "
        "If the user asks a general question, have a normal conversation. "
        "If the user asks a financial question, answer it concisely. "
        "Never use hashtags or emojis.\n\n"
        f"User: {question}\nAnalyst:"
    )

    try:
        response = llm.complete(prompt)
        return str(response)
    except Exception as e:
        print(f"Ollama Chat Error: {e}")
        return "I am having trouble connecting to my brain (Ollama). Please try again."

def generate_search_query(question: str) -> str:
    """
    Kullanıcı sorusunu arama motoru anahtar kelimelerine çevirir.
    """
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
    """
    Basit hisse senedi bulucu (Hardcoded).
    Bunu da istersen Ollama'ya sorabilirsin ama bu yöntem daha hızlıdır.
    """
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

    print("No specific ticker found in question.")
    return None

def query_online(question: str, test: bool = False) -> str:
    """
    Finnhub API'den haber çeker ve RAG yapar.
    """
    if test:
        print("--- Performing online research (test mode) ---")
        return "Test online research response"
        
    print("--- Performing online research (Finnhub Hybrid + Re-rank Strategy) ---")

    ticker = extract_ticker_from_keywords(question)
    
    try:
        articles = []
        if ticker:
            print(f"Ticker identified: {ticker}. Fetching company-specific news.")
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
            print("No specific ticker. Fetching general market news.")
            endpoint_url = "https://finnhub.io/api/v1/news"
            params = {"category": "general", "minId": 0, "token": FINNHUB_API_KEY}
            response = requests.get(endpoint_url, params=params)
            response.raise_for_status()
            articles = response.json()[:30]

        if not articles:
            return "No recent news found via Finnhub."

        print(f"Found {len(articles)} articles. Processing for RAG...")

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
            return "Found articles, but content was empty."

        # RAG
        index = VectorStoreIndex.from_documents(documents_list)
        query_engine = index.as_query_engine(
            response_mode="compact",
            similarity_top_k=5,
            # text_qa_template=qa_template,
        )

        print("Synthesizing answer from top 5 chunks using Ollama...")
        response = query_engine.query(question)
        
        news_df.to_csv("finnhub_retrieved_articles.csv", index=False)
        
        return str(response)

    except Exception as e:
        print(f"Error in query_online: {e}")
        return f"Error while doing online research: {str(e)}"

def query_document(question: str, doc_path: str, test: bool = False) -> str:
    """
    Yerel PDF/Dosya üzerinde RAG yapar.
    """
    if test:
        print("--- Performing document research (test mode) ---")
        return "Test document research response"
        
    print(f"--- Querying document with Ollama: {doc_path} ---")
    if not os.path.exists(doc_path):
        return "Error: Document not found."

    try:
        reader = SimpleDirectoryReader(input_files=[doc_path])
        docs = reader.load_data()
        
        index = VectorStoreIndex.from_documents(docs)

        query_engine = index.as_query_engine(
            response_mode="compact",
            similarity_top_k=3,
        )

        response = query_engine.query(question)
        return str(response)

    except Exception as e:
        print(f"Error in query_document: {e}")
        return "Error while processing the document."

# TEST CODE
if __name__ == "__main__":
    print("-----------------------------------")
    print("Test 1: Plain Chat (Ollama)")
    chat_resp = plain_chat("What is a 10-K report?")
    print(f"LLM Response:\n{chat_resp}\n")

    print("-----------------------------------")
    print("Test 2: Online RAG Query")
    online_question = "What is the market's reaction to NVIDIA's most recent product announcements?"
    rag_resp = query_online(online_question)
    print(f"Online RAG Response:\n{rag_resp}\n")
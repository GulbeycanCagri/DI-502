import os
import torch
import requests
import datetime
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate  

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")


# ---------------------------------------------------------------------------
# CUSTOM RAG PROMPT TEMPLATE
# ---------------------------------------------------------------------------

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
#qa_template = PromptTemplate(DEFAULT_QA_PROMPT_TMPL)
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_model():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # It is commented out to disable quantization for better performance.
    # We can re-enable it later if needed.
    
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Load in full bfloat16 precision
        device_map="auto",
        # quantization_config=quant_config, # <-- Quantization is disabled
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


model, tokenizer = get_model()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
tokenizer.chat_template = None

chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False,
)


Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1, "do_sample": True},
    device_map="auto",
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.chunk_size = 1024


# ---------------------------------------------------------------------------
#  PLAIN CHAT 
# ---------------------------------------------------------------------------

def plain_chat(question: str) -> str:
    """
    Handles conversational chat with manual stop sequence handling.
    """
    print("--- Performing plain chat ---")
    
    prompt = (
        "You are a helpful financial analyst. "
        "If the user asks a general question, have a normal conversation. "
        "If the user asks a financial question, answer it concisely. "
        "Never use hashtags or emojis.\n\n"
        f"User: {question}\nAnalyst:"
    )

    output = chat_pipeline(
        prompt,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
    )
    
    raw_response = output[0]["generated_text"]
    stop_sequences = ["\nUser:", "\nAnalyst:"]
    
    min_stop_index = len(raw_response)
    for seq in stop_sequences:
        stop_index = raw_response.find(seq)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
            
    clean_response = raw_response[:min_stop_index].strip()
    return clean_response


# ---------------------------------------------------------------------------
# KEYWORD GENERATOR
# ---------------------------------------------------------------------------

def generate_search_query(question: str) -> str:
    """
    Uses the LLM to convert a full question into search engine keywords.
    """
    print("--- Generating search keywords ---")
    prompt = (
        "You are a search query generator. "
        "Convert the following user question into 3-5 keywords for a financial news search engine. "
        "IMPORTANT: Only output the keywords. Do not add any other text, examples, formatting, or explanations.\n\n"
        f"Question: {question}\nKeywords:"
    )
    
    try:
        output = chat_pipeline(
            prompt,
            max_new_tokens=20,
            temperature=0.1,  
            do_sample=False,
        )
        
        raw_response = output[0]["generated_text"].strip()
        
        keywords = raw_response.split('\n')[0].strip().replace('"', '')
        
        print(f"Generated keywords: {keywords}")
        return keywords
    except Exception as e:
        print(f"Keyword generation failed: {e}. Falling back to raw question.")
     
        return question

# ---------------------------------------------------------------------------
# ONLINE RAG 
# ---------------------------------------------------------------------------
def extract_ticker_from_keywords(question: str) -> str | None:
    """
    Analyzes the user's question to find a relevant stock ticker.
    This is a simple implementation. A real product might use an LLM
    or a fuzzy search against a list of all tickers.
    """
    
    lowered_question = question.lower()
    
    # Simple hardcoded mapping
    # In a real app, we'd use a dictionary or a search.
    if "nvidia" in lowered_question:
        return "NVDA"
    if "apple" in lowered_question:
        return "AAPL"
    if "amd" in lowered_question or "advanced micro devices" in lowered_question:
        return "AMD"
    if "intel" in lowered_question:
        return "INTC"
    if "tesla" in lowered_question:
        return "TSLA"
    if "microsoft" in lowered_question:
        return "MSFT"
    
    # No specific ticker found
    print("No specific ticker found in question.")
    return None

def query_online(question: str) -> str:
    """
    Performs online research using the Finnhub.io API.
    - If a ticker is found, gets company-specific news.
    - If no ticker is found, gets general market news.
    - Uses LlamaIndex's vector search to "re-rank" and find
      the most relevant articles from a larger pool.
    """
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
                'symbol': ticker,
                'from': three_days_ago.strftime('%Y-%m-%d'),
                'to': today.strftime('%Y-%m-%d'),
                'token': FINNHUB_API_KEY
            }
            response = requests.get(endpoint_url, params=params)
            response.raise_for_status()
            articles = response.json()
            articles = articles[:25] 

        else:
            print("No specific ticker. Fetching general market news.")
            endpoint_url = "https://finnhub.io/api/v1/news"
            params = {
                'category': 'general',
                'minId': 0, # Required param, 0 means latest
                'token': FINNHUB_API_KEY
            }
            response = requests.get(endpoint_url, params=params)
            response.raise_for_status()
            articles = response.json()
            articles = articles[:30] 

        if not articles:

            return "No recent news found."

        print(f"Found {len(articles)} articles from Finnhub. Now processing for relevance.")

        # 4. Create Document List
        documents_list = []
        news_df = pd.DataFrame(columns=['headline', 'summary', 'source', 'url'])
        for article in articles:
            headline = article.get('headline', '')
            summary = article.get('summary', '')
            source_name = article.get('source', 'Unknown')
            
            full_text = (
                f"Source: {source_name}\n"
                f"Headline: {headline}\n"
                f"Summary: {summary}\n" 
            )
            
            news_df = pd.concat([news_df, 
                                 pd.DataFrame(
                                     [{ 'headline': 
                                         headline, 'summary': 
                                             summary, 'source': source_name, 'url': article.get('url', '') }])], 
                                ignore_index=True)
            
            
            
            if summary and headline: 
                doc = Document(
                    text=full_text,
                    metadata={
                        "source_title": headline,
                        "url": article.get('url', ''),
                        "source_name": source_name
                    }
                )
                documents_list.append(doc)

        if not documents_list:
            return "Found articles, but none had relevant content to process."

        index = VectorStoreIndex.from_documents(documents_list)
        query_engine = index.as_query_engine(
            response_mode="compact",
            #text_qa_template=qa_template,
            similarity_top_k=5

        )
        
        print(f"Synthesizing answer from top 5 most relevant chunks...")
        response = query_engine.query(question)
        news_df.to_csv("finnhub_retrieved_articles.csv", index=False)
        return str(response)

    except Exception as e:
        print(f"Error in query_online (Finnhub Strategy): {e}")
        return "Error while doing online research."
    
# ---------------------------------------------------------------------------
# DOCUMENT RAG (Upgraded)
# ---------------------------------------------------------------------------

def query_document(question: str, doc_path: str) -> str:
    """
    Performs RAG on a local document.
    """
    print(f"--- Querying document: {doc_path} ---")
    if not os.path.exists(doc_path):
        return "Error: Document not found."

    try:
        reader = SimpleDirectoryReader(input_files=[doc_path])
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)
        
        # Create Query Engine with our custom prompt
        query_engine = index.as_query_engine(
            response_mode="compact",
            #text_qa_template=qa_template  # <-- USE CUSTOM PROMPT
        )
        
        response = query_engine.query(question)
        return str(response)

    except Exception as e:
        print(f"Error in query_document: {e}")
        return "Error while processing the document."

# ---------------------------------------------------------------------------
# TEST CODE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    print("-----------------------------------")
    print("Test 1: Plain Chat")
    chat_resp = plain_chat("What is a 10-K report?")
    print(f"LLM Response:\n{chat_resp}\n")
    
    print("-----------------------------------")
    print("Test 2: Online RAG Query")
    
    # Use the complex question from before
    online_question = "What is the market's reaction to NVIDIA's most recent product announcements, and how are analysts adjusting their price targets?"
    rag_resp = query_online(online_question)
    print(f"Online RAG Response:\n{rag_resp}\n")
    
    # print("-----------------------------------")
    # print("Test 3: Document RAG (Create a PDF file to test this)")
    # # Example: Assumes you have a file named 'test_report.pdf'
    # doc_path = "test_report.pdf" 
    # if os.path.exists(doc_path):
    #     doc_question = "What was the total revenue mentioned in the report?"
    #     doc_resp = query_document(doc_question, doc_path)
    #     print(f"Document RAG Response:\n{doc_resp}\n")
    # else:
    #     print(f"Skipping Document RAG test, file not found: {doc_path}")
import datetime
import os
import time

import requests
import torch
import wandb
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
WANDB_PROJECT = "Economind"
WANDB_RUN_NAME = "llama3_base_model"

wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)


# MODEL LOADING
def get_model():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # quantization_config=quant_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


model, tokenizer = get_model()

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


#  PERFORMANCE LOGGER
def log_metrics(tag: str, start_time: float, response: str, prefix: str = "", **extra):
    """Logs performance metrics to W&B."""
    end_time = time.time()
    elapsed = end_time - start_time
    num_tokens = len(tokenizer.encode(response))
    throughput = num_tokens / elapsed if elapsed > 0 else 0
    gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)

    wandb.log(
        {
            "mode": tag,
            f"response_time_sec_{prefix}": elapsed,
            f"num_tokens_{prefix}": num_tokens,
            f"throughput_tokens_per_sec_{prefix}": throughput,
            f"gpu_memory_MB_{prefix}": gpu_mem,
            **extra,
        }
    )

    print(
        f"[{tag}] time={elapsed:.2f}s | tokens={num_tokens} | "
        f"throughput={throughput:.1f} tok/s | GPU={gpu_mem:.1f} MB"
    )


# PLAIN CHAT
def plain_chat(question: str) -> str:
    print("--- Performing plain chat ---")
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    prompt = (
        "You are a concise financial analyst. "
        "If the user asks a financial question, answer it clearly and shortly. "
        "Never use hashtags or emojis.\n\n"
        f"Question: {question}\nAnswer:"
    )

    output = chat_pipeline(
        prompt,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False,
        return_full_text=False,
    )

    response = output[0]["generated_text"].strip()
    log_metrics("plain_chat", start_time, response, prefix="plain_chat")
    return response


# ONLINE RAG (Finnhub API)
def extract_ticker_from_keywords(question: str) -> str | None:
    q = question.lower()
    for k, v in {
        "nvidia": "NVDA",
        "apple": "AAPL",
        "amd": "AMD",
        "intel": "INTC",
        "tesla": "TSLA",
        "microsoft": "MSFT",
    }.items():
        if k in q:
            return v
    return None


def query_online(question: str) -> str:
    print("--- Performing online RAG via Finnhub API ---")
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    ticker = extract_ticker_from_keywords(question)
    articles = []
    try:
        if ticker:
            today = datetime.date.today()
            past = today - datetime.timedelta(days=10)
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": past,
                "to": today,
                "token": FINNHUB_API_KEY,
            }
        else:
            url = "https://finnhub.io/api/v1/news"
            params = {"category": "general", "token": FINNHUB_API_KEY}

        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json()[:25]

        if not articles:
            return "No relevant news found."

        docs = []
        for a in articles:
            headline, summary = a.get("headline", ""), a.get("summary", "")
            if headline and summary:
                docs.append(Document(text=f"Headline: {headline}\nSummary: {summary}"))

        index = VectorStoreIndex.from_documents(docs)
        num_chunks = len(index.docstore.docs)
        engine = index.as_query_engine(response_mode="compact", similarity_top_k=5)
        rag_response = engine.query(question)

        log_metrics(
            "online_rag",
            start_time,
            str(rag_response),
            num_chunks=num_chunks,
            response_time_per_chunk=(time.time() - start_time) / max(num_chunks, 1),
            prefix="online_rag",
        )
        return str(rag_response)

    except Exception as e:
        print(f"Error in RAG: {e}")
        return "Error performing online research."


# DOCUMENT RAG (with Normalized Metrics)
def query_document(question: str, doc_path: str) -> str:
    print(f"--- Querying document: {doc_path} ---")
    if not os.path.exists(doc_path):
        return "Error: Document not found."

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    try:
        reader = SimpleDirectoryReader(input_files=[doc_path])
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)
        engine = index.as_query_engine(response_mode="compact")
        resp = engine.query(question)

        # Additional metadata
        num_chunks = len(index.docstore.docs)
        file_size_kb = os.path.getsize(doc_path) / 1024
        response_time = time.time() - start_time

        log_metrics(
            "document_rag",
            start_time,
            str(resp),
            num_chunks=num_chunks,
            document_size_kb=file_size_kb,
            response_time_per_chunk=response_time / max(num_chunks, 1),
            prefix="document_rag",
        )

        print(f"[RAG] num_chunks={num_chunks}, file_size={file_size_kb:.1f} KB")
        return str(resp)
    except Exception as e:
        print(f"Error in Document RAG: {e}")
        return "Error processing document."


if __name__ == "__main__":
    print("-----------------------------------")
    print("Test 1: Plain Chat")
    chat_resp = plain_chat("What is a 10-K report?")
    print(f"LLM Response:\n{chat_resp}\n")

    print("-----------------------------------")
    print("Test 2: Online RAG Query")
    online_question = (
        "What is the market's reaction to NVIDIA's most recent product announcements, "
        "and how are analysts adjusting their price targets?"
    )
    rag_resp = query_online(online_question)
    print(f"Online RAG Response:\n{rag_resp}\n")

    print("-----------------------------------")
    print("Test 3: Document RAG")
    example_doc = "apple_test.pdf"
    if os.path.exists(example_doc):
        doc_resp = query_document(
            "What are Apple’s main product categories as described in its 2024 Form 10-K report",
            example_doc,
        )
        print(f"Document RAG Response:\n{doc_resp}\n")
    else:
        print("No PDF found. Skipping Document RAG test.")

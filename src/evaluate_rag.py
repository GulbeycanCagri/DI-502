import os
import torch
import requests
import datetime
import time
import wandb
import pandas as pd
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from typing import List


import json
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance,
)
from langchain_google_genai import ChatGoogleGenerativeAI
import ragas
import asyncio
import grpc.aio
import re

load_dotenv()
WANDB_PROJECT = "Economind"
WANDB_RUN_NAME = "llama3_rag_evaluation_complex_contexts_not_quantized"


try:
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    hf_embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    ragas.llm = gemini_llm
    print("RAGAS Judge (Gemini) initialized.")
except Exception as e:
    print(f"Could not initialize Gemini. RAGAS evaluation will fail. Error: {e}")
    gemini_llm = None

wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)


# MODEL LOADING
def get_model():
    model_id = "meta-llama/Llama-2-7b-chat-hf"

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
        #quantization_config=quant_config,
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


def get_performance_metrics(tag: str, 
                start_time: float, 
                response: str, 
                prefix: str = "",
                **extra):

    end_time = time.time()
    elapsed = end_time - start_time
    num_tokens = len(tokenizer.encode(response))
    throughput = num_tokens / elapsed if elapsed > 0 else 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        gpu_mem = 0.0

    performance_data = {
        f"response_time_sec": elapsed,
        f"throughput_tokens_per_sec": throughput,
        f"gpu_memory_MB": gpu_mem,
    }

    return performance_data

def generate_llama_answer(question: str, contexts: List[str], sample_idx: int) -> str:
    """Runs the local Llama 3 instruct model on the retrieved contexts."""

    context_block = "\n\n".join(
        [f"[Context {idx}]\n{ctx.strip()}" for idx, ctx in enumerate(contexts, start=1)]
    )

    prompt = (
    "You are a diligent financial analyst. Answer using ONLY the provided SEC filing excerpts. "
    "If the answer is not in the context, say you cannot find it. Avoid hallucinations. "
    "**DO NOT** include the context labels (e.g., [Context 1], [Context 2]) in your final answer. " 
    f"{context_block}\n\nQuestion: {question}\nAnswer:"
)

    start_time = time.time()
    output = chat_pipeline(
        prompt,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
    )
    rag_answer = output[0]["generated_text"].strip()
    rag_answer = rag_answer = re.sub(r'\[Context\s+\d+\]', '', rag_answer).strip()
    performance_metrics= get_performance_metrics(
        tag="RAG_Generation",
        start_time=start_time,
        response=rag_answer,
        prefix=f"sample_{sample_idx}_"
    )
    return rag_answer, performance_metrics


def evaluate_rag_pipeline(evaluation_json_path: str):
    """
    Runs the RAG pipeline against a golden test set and evaluates
    it using RAGAS.
    """
  
    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    performance_data = {
        "response_time_sec": [],
        "throughput_tokens_per_sec": [],
        "gpu_memory_MB": [],
    }
    
    
    
    
    print("Running RAG pipeline to generate answers for evaluation...")


    for enum, golden_file in enumerate(os.listdir(evaluation_json_path)):
        if enum ==1:
            break
        golden_file_path = os.path.join(evaluation_json_path, golden_file)
        with open(golden_file_path, "r") as f:
            golden_dataset = json.load(f)
    
        # Re-build the RAG pipeline for each item in the test set
        for item in tqdm(golden_dataset):
            context = item.get("context")
            qa_pairs = item.get("qa_pairs", [])
            
            if not context or not qa_pairs:
                continue
                
            for qa in qa_pairs:
                question = qa.get("question")
                ground_truth_answer = qa.get("answer")
                
                if not question or not ground_truth_answer:
                    continue

                doc = Document(text=context)
                index = VectorStoreIndex.from_documents([doc])
                retriever = index.as_retriever(similarity_top_k=5)
                retrieved_nodes = retriever.retrieve(question)
                retrieved_contexts = [node.get_content() for node in retrieved_nodes]
                if not retrieved_contexts:
                    retrieved_contexts = [context]

                sample_idx = len(evaluation_data["question"]) + 1
                rag_answer, perf_metrics = generate_llama_answer(question, retrieved_contexts, sample_idx)

                # 4. Store the data for RAGAS
                evaluation_data["question"].append(question)
                evaluation_data["answer"].append(rag_answer)
                evaluation_data["contexts"].append(retrieved_contexts)
                evaluation_data["ground_truth"].append(ground_truth_answer)
                
                performance_data["response_time_sec"].append(perf_metrics["response_time_sec"])
                performance_data["throughput_tokens_per_sec"].append(perf_metrics["throughput_tokens_per_sec"])
                performance_data["gpu_memory_MB"].append(perf_metrics["gpu_memory_MB"])
                
                


    return evaluation_data, performance_data


if __name__ == "__main__":
    
    final_results_evaluation = {}
    final_results_performance = {}
    # The program should be started from the main directory of the project
    evaluation_data, results_performance = evaluate_rag_pipeline("data/datasets/test_SEC_10K_dataset")
    if not evaluation_data["question"]:
        raise ValueError("No evaluation samples were generated. Check the evaluation dataset.")
    
    dataset = Dataset.from_dict(evaluation_data)
    metrics_to_run = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextRelevance()
    ]

    results_evaluation = evaluate(
        dataset,
        metrics=metrics_to_run,
        llm=gemini_llm,
        embeddings=hf_embed_model
    )

    # return mean scores
    final_results_evaluation["faithfulness"] = sum(list(results_evaluation["faithfulness"])) / len(evaluation_data["question"])
    final_results_evaluation["answer_relevancy"] = sum(list(results_evaluation["answer_relevancy"])) / len(evaluation_data["question"])
    final_results_evaluation["context_relevance"] = sum(list(results_evaluation["nv_context_relevance"])) / len(evaluation_data["question"])
    
    #make nans to 0
    final_results_evaluation = {k: (0 if (isinstance(v, float) and (v != v)) else v) for k, v in final_results_evaluation.items()}
    
    
    # return std scores
    final_results_evaluation["faithfulness_std"] = torch.std(torch.tensor(results_evaluation["faithfulness"])).item()
    final_results_evaluation["answer_relevancy_std"] = torch.std(torch.tensor(results_evaluation["answer_relevancy"])).item()
    final_results_evaluation["context_relevance_std"] = torch.std(torch.tensor(results_evaluation["nv_context_relevance"])).item()

    #make nans to 0
    final_results_evaluation = {k: (0 if (isinstance(v, float) and (v != v)) else v) for k, v in final_results_evaluation.items()}


    final_results_performance["response_time_sec"] = sum(list(results_performance["response_time_sec"])) / len(evaluation_data["question"])
    final_results_performance["throughput_tokens_per_sec"] = sum(list(results_performance["throughput_tokens_per_sec"])) / len(evaluation_data["question"])
    final_results_performance["gpu_memory_MB"] = sum(list(results_performance["gpu_memory_MB"])) / len(evaluation_data["question"])

    #return std scores
    final_results_performance["response_time_sec_std"] = torch.std(torch.tensor(results_performance["response_time_sec"])).item()
    final_results_performance["throughput_tokens_per_sec_std"] = torch.std(torch.tensor(results_performance["throughput_tokens_per_sec"])).item()
    final_results_performance["gpu_memory_MB_std"] = torch.std(torch.tensor(results_performance["gpu_memory_MB"])).item()   

    wandb.log({"ragas": final_results_evaluation}) 
    wandb.log({"performance": final_results_performance})   

    
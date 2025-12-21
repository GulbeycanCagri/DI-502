import os
import torch
import time
import wandb
import argparse
import sys
import json
import re
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRelevance
import ragas

load_dotenv()

# Configuration
WANDB_PROJECT = "Economind"
WANDB_GROUP = "Embedder Performance Comparison"

# Define Embedding Models to test
EMBEDDING_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-large-v2"
]

# Fixed Prompt for Embedder Evaluation (using 'detailed_analyst' as standard)
FIXED_PROMPT = {
    "name": "detailed_analyst",
    "description": "Detailed financial analyst with comprehensive answers",
    "template": (
        "You are a senior financial analyst. Analyze the provided SEC filing excerpts comprehensively "
        "and provide detailed answers with relevant details and context. If information cannot be found, explicitly state this. "
        "Never fabricate information. Exclude any reference labels from your answer. "
        "{context_block}\n\nQuestion: {question}\nProvide a detailed answer:"
    )
}

# Initialize Gemini for RAGAS Judge
try:
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    ragas.llm = gemini_llm
    print("RAGAS Judge (Gemini) initialized.")
except Exception as e:
    print(f"Could not initialize Gemini. RAGAS evaluation will fail. Error: {e}")
    gemini_llm = None


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
        quantization_config=quant_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Load LLM once (global)
print("Loading LLM...")
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
Settings.chunk_size = 1024


def get_performance_metrics(start_time: float, response: str) -> Dict:
    """Calculate performance metrics for the model response."""
    end_time = time.time()
    elapsed = end_time - start_time
    num_tokens = len(tokenizer.encode(response))
    throughput = num_tokens / elapsed if elapsed > 0 else 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        gpu_mem = 0.0

    return {
        "response_time_sec": elapsed,
        "throughput_tokens_per_sec": throughput,
        "gpu_memory_MB": gpu_mem,
    }


def generate_llama_answer(question: str, contexts: List[str], prompt_template: str) -> tuple:
    """Generate answer using Llama 3 with a specific prompt template."""
    
    context_block = "\n\n".join(
        [f"[Context {idx}]\n{ctx.strip()}" for idx, ctx in enumerate(contexts, start=1)]
    )

    prompt = prompt_template.format(context_block=context_block, question=question)

    start_time = time.time()
    output = chat_pipeline(
        prompt,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
    )
    rag_answer = output[0]["generated_text"].strip()
    # Remove context reference labels
    rag_answer = re.sub(r'\[Context\s+\d+\]', '', rag_answer).strip()
    performance_metrics = get_performance_metrics(start_time, rag_answer)
    
    return rag_answer, performance_metrics, prompt


def evaluate_rag_pipeline_with_embedder(evaluation_json_path: str, prompt_config: Dict) -> tuple:
    """Run RAG pipeline with a specific embedder (configured globally) and collect evaluation data."""
    
    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "prompt": [],
    }
    
    performance_data = {
        "response_time_sec": [],
        "throughput_tokens_per_sec": [],
        "gpu_memory_MB": [],
    }

    print(f"\nRunning RAG pipeline...")

    for enum, golden_file in enumerate(os.listdir(evaluation_json_path)):
        if enum == 1:
            break
        golden_file_path = os.path.join(evaluation_json_path, golden_file)
        with open(golden_file_path, "r") as f:
            golden_dataset = json.load(f)

        # Process each item in the test set
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

                # Create index using the CURRENT Settings.embed_model
                doc = Document(text=context)
                index = VectorStoreIndex.from_documents([doc])
                retriever = index.as_retriever(similarity_top_k=5)
                retrieved_nodes = retriever.retrieve(question)
                retrieved_contexts = [node.get_content() for node in retrieved_nodes]
                if not retrieved_contexts:
                    retrieved_contexts = [context]

                rag_answer, perf_metrics, used_prompt = generate_llama_answer(
                    question, 
                    retrieved_contexts, 
                    prompt_config["template"]
                )

                # Store data for RAGAS evaluation
                evaluation_data["question"].append(question)
                evaluation_data["answer"].append(rag_answer)
                evaluation_data["contexts"].append(retrieved_contexts)
                evaluation_data["ground_truth"].append(ground_truth_answer)
                evaluation_data["prompt"].append(used_prompt)
                
                performance_data["response_time_sec"].append(perf_metrics["response_time_sec"])
                performance_data["throughput_tokens_per_sec"].append(perf_metrics["throughput_tokens_per_sec"])
                performance_data["gpu_memory_MB"].append(perf_metrics["gpu_memory_MB"])

    return evaluation_data, performance_data


def compute_ragas_evaluation(evaluation_data: Dict, ragas_embed_model) -> Dict:
    """Compute RAGAS metrics for the evaluation data using specific embedder."""
    
    if not evaluation_data["question"]:
        return None

    dataset = Dataset.from_dict({
        "question": evaluation_data["question"],
        "answer": evaluation_data["answer"],
        "contexts": evaluation_data["contexts"],
        "ground_truth": evaluation_data["ground_truth"],
    })
    
    metrics_to_run = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextRelevance()
    ]

    results = evaluate(
        dataset,
        metrics=metrics_to_run,
        llm=gemini_llm,
        embeddings=ragas_embed_model
    )

    # Compute mean scores
    final_results = {
        "faithfulness": sum(list(results["faithfulness"])) / len(evaluation_data["question"]),
        "answer_relevancy": sum(list(results["answer_relevancy"])) / len(evaluation_data["question"]),
        "context_relevance": sum(list(results["nv_context_relevance"])) / len(evaluation_data["question"]),
    }
    
    # Replace NaN with 0
    final_results = {k: (0 if (isinstance(v, float) and (v != v)) else v) for k, v in final_results.items()}
    
    # Compute standard deviations
    final_results["faithfulness_std"] = torch.std(torch.tensor(results["faithfulness"])).item()
    final_results["answer_relevancy_std"] = torch.std(torch.tensor(results["answer_relevancy"])).item()
    final_results["context_relevance_std"] = torch.std(torch.tensor(results["nv_context_relevance"])).item()
    
    # Replace NaN with 0
    final_results = {k: (0 if (isinstance(v, float) and (v != v)) else v) for k, v in final_results.items()}

    return final_results


def compute_performance_metrics(performance_data: Dict) -> Dict:
    """Compute mean and std performance metrics."""
    
    if not performance_data["response_time_sec"]:
        return None

    n_samples = len(performance_data["response_time_sec"])
    
    final_metrics = {
        "response_time_sec": sum(performance_data["response_time_sec"]) / n_samples,
        "throughput_tokens_per_sec": sum(performance_data["throughput_tokens_per_sec"]) / n_samples,
        "gpu_memory_MB": sum(performance_data["gpu_memory_MB"]) / n_samples,
    }

    final_metrics["response_time_sec_std"] = torch.std(torch.tensor(performance_data["response_time_sec"])).item()
    final_metrics["throughput_tokens_per_sec_std"] = torch.std(torch.tensor(performance_data["throughput_tokens_per_sec"])).item()
    final_metrics["gpu_memory_MB_std"] = torch.std(torch.tensor(performance_data["gpu_memory_MB"])).item()

    return final_metrics


def log_sample_data_to_wandb(evaluation_data: Dict, performance_data: Dict):
    """Log individual samples with their prompts, answers, and performance metrics to W&B table."""
    
    # Create a table with sample details
    table = wandb.Table(columns=[
        "Sample_ID", "Question", "Ground_Truth", "Model_Answer", 
        "Contexts", "Context_Count", "Prompt_Used", 
        "Response_Time_Sec", "Throughput_Tokens_Per_Sec"
    ])
    
    for idx in range(len(evaluation_data["question"])):
        # Join contexts with a separator
        contexts_str = "\n---\n".join(evaluation_data["contexts"][idx])
        
        table.add_data(
            idx + 1,
            evaluation_data["question"][idx],
            evaluation_data["ground_truth"][idx],
            evaluation_data["answer"][idx],
            contexts_str,
            len(evaluation_data["contexts"][idx]),
            evaluation_data["prompt"][idx][:100] + "..." if len(evaluation_data["prompt"][idx]) > 100 else evaluation_data["prompt"][idx],
            performance_data["response_time_sec"][idx] if idx < len(performance_data["response_time_sec"]) else 0,
            performance_data["throughput_tokens_per_sec"][idx] if idx < len(performance_data["throughput_tokens_per_sec"]) else 0
        )
    
    wandb.log({"samples_details": table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation for a specific embedder.")
    parser.add_argument("--embedder_idx", type=int, default=None, help="Index of the embedder to run (0-based).")
    args = parser.parse_args()

    # Determine which embedders to run
    embedders_to_run = EMBEDDING_MODELS
    if args.embedder_idx is not None:
        if 0 <= args.embedder_idx < len(EMBEDDING_MODELS):
            embedders_to_run = [EMBEDDING_MODELS[args.embedder_idx]]
            print(f"Running specific embedder index {args.embedder_idx}: {embedders_to_run[0]}")
        else:
            print(f"Error: Invalid embedder index {args.embedder_idx}. Must be between 0 and {len(EMBEDDING_MODELS)-1}.")
            sys.exit(1)

    all_results = {}
    
    for embedder_name in embedders_to_run:
        # Clean embedder name for W&B run name (replace slashes)
        safe_embedder_name = embedder_name.replace("/", "_")
        run_name = f"llama3_rag_eval_embedder_{safe_embedder_name}"
        
        # Initialize W&B run
        wandb.init(project=WANDB_PROJECT, 
                   group=WANDB_GROUP,
                   name=run_name, 
                   reinit=True)
        
        print(f"\n{'='*60}")
        print(f"Starting evaluation run for Embedder: {embedder_name}")
        print(f"{'='*60}")
        
        # 1. Configure LlamaIndex Embedder
        print(f"Initializing LlamaIndex embedding model: {embedder_name}")
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedder_name)
        
        # 2. Configure RAGAS Embedder (LangChain wrapper)
        print(f"Initializing RAGAS embedding model: {embedder_name}")
        ragas_embed_model = HuggingFaceEmbeddings(model_name=embedder_name)
        
        # Run evaluation pipeline
        evaluation_data, performance_data = evaluate_rag_pipeline_with_embedder(
            "data/datasets/test_SEC_10K_dataset",
            FIXED_PROMPT
        )
        
        if not evaluation_data["question"]:
            print(f"Warning: No evaluation samples generated for embedder: {embedder_name}")
            wandb.finish()
            continue
        
        # Compute RAGAS metrics
        ragas_results = compute_ragas_evaluation(evaluation_data, ragas_embed_model)
        perf_results = compute_performance_metrics(performance_data)
        
        # Store results
        all_results[embedder_name] = {
            "ragas": ragas_results,
            "performance": perf_results,
            "num_samples": len(evaluation_data["question"])
        }
        
        # Log to W&B
        if ragas_results:
            ragas_log = {f"ragas/{k}": v for k, v in ragas_results.items()}
            wandb.log(ragas_log)
            
        if perf_results:
            perf_log = {f"performance/{k}": v for k, v in perf_results.items()}
            wandb.log(perf_log)
        
        # Log configuration
        wandb.log({
            "embedding_model": embedder_name,
            "prompt_name": FIXED_PROMPT["name"],
            "num_evaluation_samples": len(evaluation_data["question"])
        })
        
        # Log sample details
        log_sample_data_to_wandb(evaluation_data, performance_data)
        
        # Finish W&B run
        wandb.finish()
        
        print(f"\nCompleted evaluation for embedder: {embedder_name}")
        if ragas_results:
            print(f"  - Faithfulness: {ragas_results['faithfulness']:.4f}")
            print(f"  - Answer Relevancy: {ragas_results['answer_relevancy']:.4f}")
            print(f"  - Context Relevance: {ragas_results['context_relevance']:.4f}")
    
    # Summary report
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for embedder_name, results in all_results.items():
        print(f"\n{embedder_name}:")
        print(f"  Samples: {results['num_samples']}")
        if results['ragas']:
            print(f"  Faithfulness: {results['ragas']['faithfulness']:.4f}")
            print(f"  Answer Relevancy: {results['ragas']['answer_relevancy']:.4f}")
            print(f"  Context Relevance: {results['ragas']['context_relevance']:.4f}")

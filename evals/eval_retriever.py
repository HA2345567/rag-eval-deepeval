from pathlib import Path
import os
import sys


# Support direct execution as well as `python -m evals.eval_retriever`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from evals.harness import load_goldens, print_summary, summarize_by_metric
from reranker import RERANK_MODEL, RerankingRetriever
from retriever import MODEL as EMBEDDING_MODEL


load_dotenv(PROJECT_ROOT / ".env")

GOLDEN_PATH = str(PROJECT_ROOT / "goldens" / "retriever_goldens.json")
JUDGE_MODEL = os.getenv("NVIDIA_JUDGE_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
THRESHOLD = 0.7

def run(retriever):
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is missing from .env")

    judge = OpenAIModel(
        model=JUDGE_MODEL,
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0,
    )
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    goldens = load_goldens(GOLDEN_PATH)
    
    test_cases=[]
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]
        
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    #4. Evaluate 
    
    result = evaluate(
        test_cases = test_cases,
        metrics=metrics,
       hyperparameters={
    "retriever": "reranked",
    "embedding_model": EMBEDDING_MODEL,
    "reranker_model": RERANK_MODEL,
    "generator_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "fetch_k": 10,
    "top_k": 5,
    "judge_model": judge.get_model_name(),
    "golden_set": GOLDEN_PATH,
},
    )
    
    return summarize_by_metric(result)

def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(RerankingRetriever())

if __name__ == "__main__":
    print_summary("retriever", run_local())

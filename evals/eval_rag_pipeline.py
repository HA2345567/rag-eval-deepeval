
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from evals.harness import load_goldens, summarize_by_metric
from rag_pipeline import RagPipeline

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"  # reuse the queries
JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b"
THRESHOLD = 0.7


def run(rag):
    goldens = load_goldens(GOLDEN_PATH)
    test_cases = []

    for g in goldens:
        result = rag.invoke(g["query"])

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],
                retrieval_context=result["context"],
            )
        )

    # 3. Three RAG metrics
    metrics = [
        ContextualRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True
        ),
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True
        )
    ]

    # 4. Evaluate
    result = evaluate(test_cases=test_cases, metrics=metrics)

    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run the evaluation."""

    return run(RagPipeline())


if __name__ == "__main__":
    print("rag_pipeline", run_local())

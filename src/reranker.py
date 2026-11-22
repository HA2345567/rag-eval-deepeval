
from langchain_nvidia_ai_endpoints import Model, NVIDIARerank, register_model

from langsmith import traceable
from dotenv import load_dotenv

from retriever import load_storage

load_dotenv()

RERANK_MODEL = "nvidia/nemotron-3.5-rerank-8b"

# This hosted endpoint is newer than the SDK's built-in model registry.
register_model(
    Model(
        id=RERANK_MODEL,
        model_type="ranking-vlm",
        client="NVIDIARerank",
        endpoint=(
            "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
            "nemotron-3.5-rerank-8b/reranking"
        ),
    )
)


class RerankingRetriever:
    def __init__(self, fetch_k=10, top_k=5):
        self.store = load_storage()
        self.reranker = NVIDIARerank(
            model=RERANK_MODEL,
            top_n=top_k
        )
        self.fetch_k = fetch_k
        self.top_k = top_k

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        # 1. OVER-RETRIEVE: fast vector similarity search
        candidates = self.store.similarity_search(
            query, k=self.fetch_k
        )

        if not candidates:
            return []

        # 2. RERANK: NVIDIA API reranks the documents
        ranked = self.reranker.compress_documents(
            documents=candidates,
            query=query
        )

        # 3. RETURN: top-k most relevant documents
        return ranked[:self.top_k]

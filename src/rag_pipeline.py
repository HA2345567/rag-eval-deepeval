
from langsmith import traceable
from src.generator import generate
from src.reranker import RerankingRetriever


class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5):
        self.retriever = RerankingRetriever(
            fetch_k=fetch_k,
            top_k=top_k
        )

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        # 1. RETRIEVE
        docs = self.retriever.invoke(query)

        # 2. EXTRACT CONTEXT
        context = [doc.page_content for doc in docs]

        # 3. GENERATE
        answer = generate(query, context)

        return {
            "query": query,
            "context": context,
            "answer": answer
        }


if __name__ == "__main__":
    rag = RagPipeline()

    result = rag.invoke(
        "Why do we need golden datasets for evaluation?"
    )

    print("QUERY:", result["query"])
    print("ANSWER:", result["answer"])

    for i, chunk in enumerate(result["context"]):
        print(f"\n--- CONTEXT CHUNK {i + 1} ---")
        print(chunk)

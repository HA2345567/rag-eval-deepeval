
import os
import re
import glob

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import Model, NVIDIAEmbeddings, register_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_store"
COLLECTION_NAME = "video_transcripts"
MODEL = "nvidia/nemotron-3.5-embed-8b"

# Declare the API type for this model, which is newer than the SDK registry.
register_model(
    Model(
        id=MODEL,
        model_type="embedding",
        client="NVIDIAEmbeddings",
        endpoint="https://integrate.api.nvidia.com/v1/embeddings",
    )
)


# Read transcripts and remove VTT timestamps
def load_transcripts():
    docs = []

    for path in glob.glob(os.path.join(DATA_DIR, "*.vtt")):
        lines = []

        with open(path, "r", encoding="utf-8-sig") as file:
            for line in file:
                line = line.strip()

                if not line or line == "WEBVTT" or "-->" in line:
                    continue

                if line.isdigit():
                    continue

                lines.append(line)

        text = " ".join(lines)

        match = re.search(
            r"Session[_\s-]*(\d+)",
            os.path.basename(path),
            re.IGNORECASE
        )

        if not match or not text.strip():
            continue

        session = match.group(1)

        docs.append(
            Document(
                page_content=text,
                metadata={"session": session}
            )
        )

    return docs


# Initialize NVIDIA embeddings
def get_embeddings():
    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise ValueError("NVIDIA_API_KEY is missing from .env")

    return NVIDIAEmbeddings(
        model=MODEL,
        nvidia_api_key=api_key
    )


# Build, embed and persist documents
def load_storage():

    embeddings = get_embeddings()

    db_file = os.path.join(DB_DIR, "chroma.sqlite3")

    if os.path.isfile(db_file):
        print("Loading existing ChromaDB...")

        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

    docs = load_transcripts()

    if not docs:
        raise ValueError("No transcripts found in data directory.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    print(f"Loaded {len(docs)} transcripts.")
    print(f"Created {len(chunks)} chunks.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR
    )

    print("Embeddings stored successfully!")

    return vector_store


# Build retriever
def build_retriever():

    vector_store = load_storage()

    return vector_store.as_retriever(
        search_kwargs={"k": 3}
    )


# Main execution
if __name__ == "__main__":

    retriever = build_retriever()

    query = "What is the main topic of the video?"

    results = retriever.invoke(query)

    print("\nRetrieved Results:\n")

    for r in results:
        print(
            f"[Session {r.metadata['session']}]: "
            f"{r.page_content[:150]}...\n"
        )

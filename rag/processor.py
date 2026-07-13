import fitz  # pymupdf
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ef = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
_client = chromadb.Client()


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def build_collection(chunks: list[str], collection_name: str = "papers") -> chromadb.Collection:
    try:
        _client.delete_collection(collection_name)
    except Exception:
        pass

    collection = _client.create_collection(
        name=collection_name,
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )

    if chunks:
        collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )

    return collection

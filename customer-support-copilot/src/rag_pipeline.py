"""
RAG policy retrieval pipeline for the Customer Support Copilot.

This module loads a FAISS index built from company policy documents and retrieves
relevant policy chunks for English or German support tickets.

Important:
- Policy documents are currently written in English.
- The FAISS index must be rebuilt with the same multilingual embedding model used here.
- Default index path: vectorstore/faiss_policy_index_multilingual
"""

import os
from functools import lru_cache
from typing import Dict, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTORSTORE_DIR = os.path.join(
    BASE_DIR,
    "vectorstore",
    "faiss_policy_index_multilingual",
)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the multilingual sentence-transformer embedding model once."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    """Load the multilingual FAISS vectorstore once, with a clear error if missing."""
    if not os.path.isdir(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"FAISS vectorstore not found at: {VECTORSTORE_DIR}\n"
            "Run notebooks/07_rag_knowledge_retrieval.ipynb first to rebuild and save "
            "the multilingual FAISS index."
        )

    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings=get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def retrieve_policy_context(ticket_text: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Retrieve relevant company policy context for English or German support tickets.

    Args:
        ticket_text: Customer ticket text in English or German.
        top_k: Number of policy chunks to retrieve. Default is 3 so borderline
            cases like battery/warranty can return both technical and warranty
            context when relevant.

    Returns:
        List of dictionaries with source filename and chunk content.
    """
    if not str(ticket_text).strip():
        return []

    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(str(ticket_text), k=top_k)

    results: List[Dict[str, str]] = []
    for doc in docs:
        results.append(
            {
                "source": os.path.basename(doc.metadata.get("source", "unknown_source")),
                "content": doc.page_content,
            }
        )

    return results


if __name__ == "__main__":
    sample_queries = [
        "My product arrived damaged and I want a refund.",
        "Ich habe ein beschädigtes Produkt erhalten und möchte eine Rückerstattung.",
    ]

    for query in sample_queries:
        print("=" * 100)
        print("Query:", query)
        for item in retrieve_policy_context(query, top_k=3):
            print("Source:", item["source"])
            print(item["content"][:400])

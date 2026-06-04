import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTORSTORE_DIR = os.path.join(
    BASE_DIR,
    "vectorstore",
    "faiss_policy_index"
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_policy_vectorstore():
    """
    Loads the saved FAISS policy vectorstore.
    This is used during prediction/retrieval instead of rebuilding the index.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def retrieve_policy_context(ticket_text, top_k=2):
    """
    Retrieves the most relevant policy chunks for a given customer ticket.

    Parameters:
        ticket_text (str): Customer support ticket text.
        top_k (int): Number of policy chunks to retrieve.

    Returns:
        list: Retrieved policy chunks with source and content.
    """

    vectorstore = load_policy_vectorstore()

    retrieved_docs = vectorstore.similarity_search(
        ticket_text,
        k=top_k
    )

    results = []

    for doc in retrieved_docs:
        source_file = os.path.basename(doc.metadata.get("source", ""))

        results.append({
            "source": source_file,
            "content": doc.page_content
        })

    return results


if __name__ == "__main__":
    sample_ticket = "My product arrived damaged and I want a refund."

    retrieved_context = retrieve_policy_context(
        sample_ticket,
        top_k=2
    )

    print("Ticket:", sample_ticket)

    for i, context in enumerate(retrieved_context, 1):
        print("=" * 80)
        print(f"Result {i}")
        print("Source:", context["source"])
        print(context["content"])
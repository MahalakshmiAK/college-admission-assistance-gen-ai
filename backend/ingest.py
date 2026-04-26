import json
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from rag_engine import build_embedding_model, VECTOR_STORE_DIR

# ✅ SINGLE DATASET PATH
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "enhanced_college_dataset.json"


def load_json_data():
    """
    Reads the single JSON file and converts it into LangChain Document objects.
    """
    documents = []

    if not DATA_PATH.exists():
        print(f"❌ File not found: {DATA_PATH}")
        return documents

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # Combine fields into meaningful text
        content = f"""
        College: {item.get('college', '')}
        Category: {item.get('category', '')}
        Title: {item.get('title', '')}
        Content: {item.get('content', '')}
        """

        metadata = {
            "college": item.get("college", "General"),
            "category": item.get("category", ""),
            "title": item.get("title", "Untitled"),
            "doc_id": item.get("doc_id", "")
        }

        documents.append(
            Document(
                page_content=content.strip(),
                metadata=metadata
            )
        )

    return documents


def run_ingestion():
    """
    Load -> Vectorize -> Save
    """
    print("📂 Loading dataset...")
    raw_docs = load_json_data()

    if not raw_docs:
        print("❌ No data found.")
        return

    print(f"🧠 Encoding {len(raw_docs)} documents...")

    embedding_model = build_embedding_model()

    vector_db = Chroma.from_documents(
        documents=raw_docs,
        embedding=embedding_model,
        persist_directory=str(VECTOR_STORE_DIR)
    )

    vector_db.persist()

    print(f"✅ Ingestion complete! Saved to: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    run_ingestion()
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vectorstore"

# ✅ SINGLE DATASET
DATA_PATH = DATA_DIR / "enhanced_college_dataset.json"


# ---------------- EMBEDDINGS ----------------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def build_embedding_model():
    return SentenceTransformerEmbeddings()


# ---------------- LOAD DATA ----------------
def load_documents():
    documents = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        text = f"""
        College: {item.get("college", "")}
        Category: {item.get("category", "")}
        Title: {item.get("title", "")}
        Content: {item.get("content", "")}
        """

        doc = Document(
            page_content=text.strip(),
            metadata={
                "college": item.get("college", ""),
                "category": item.get("category", ""),
                "title": item.get("title", ""),
                "doc_id": item.get("doc_id", "")
            }
        )
        documents.append(doc)

    return documents


# ---------------- BUILD VECTOR STORE ----------------
def create_vectorstore():
    docs = load_documents()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=build_embedding_model(),
        persist_directory=str(VECTOR_STORE_DIR)
    )

    vector_store.persist()
    return vector_store


# ---------------- RETRIEVAL ----------------
def _extract_college_from_query(query: str) -> str:
    query = query.lower()
    colleges = [
        "iit bombay", "iit delhi", "iit madras", "iit kanpur",
        "iit kharagpur", "iit roorkee", "iit guwahati",
        "iit hyderabad", "iit indore", "iit gandhinagar",
        "iit jodhpur", "iit mandi", "iit patna", "iit ropar",
        "iit bhubaneswar", "iit dhanbad", "iit palakkad",
        "iit dharwad", "nit tiruchirappalli"
    ]

    for c in colleges:
        if c in query:
            return c.title()

    return "General"


def hybrid_retrieve(vector_store, query, k=5):
    target_college = _extract_college_from_query(query)
    results = vector_store.similarity_search_with_score(query, k=k * 2)

    scored_results = []
    for doc, distance in results:
        score = 1.0 / (1.0 + distance)

        # Boost if same college
        if target_college != "General" and doc.metadata.get("college", "").lower() == target_college.lower():
            score *= 1.3

        scored_results.append((doc, score))

    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results[:k]


# ---------------- GENERATION ----------------
class GroqAnswerGenerator:
    def __init__(self, api_key: str):
        from langchain_groq import ChatGroq
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3
        )

    def generate(self, query: str, chunks: list[tuple[Document, float]]) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage

        context = "\n".join([
            f"[{d.metadata.get('college')}] {d.page_content}"
            for d, _ in chunks
        ])

        system_prompt = (
            "You are a College Admission Assistant.\n"
            "- Use bullet points\n"
            "- Highlight numbers in **bold**\n"
            "- Use relevant emojis\n"
            "- Answer ONLY from given context\n"
        )

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuery: {query}")
        ])

        return response.content


# ---------------- MAIN RAG CLASS ----------------
class CollegeAdmissionRAG:
    def __init__(self, groq_api_key: str = None):
        # Load existing OR create new vectorstore
        if VECTOR_STORE_DIR.exists():
            self.vector_store = Chroma(
                persist_directory=str(VECTOR_STORE_DIR),
                embedding_function=build_embedding_model()
            )
        else:
            self.vector_store = create_vectorstore()

        self.generator = GroqAnswerGenerator(groq_api_key) if groq_api_key else None

    def answer(self, query: str, k: int = 5):
        chunks = hybrid_retrieve(self.vector_store, query, k)

        answer = (
            self.generator.generate(query, chunks)
            if self.generator else "API Key Missing."
        )

        sources = [
            {
                "title": d.metadata.get("title"),
                "college": d.metadata.get("college"),
                "category": d.metadata.get("category"),
                "score": round(score, 2)
            }
            for d, score in chunks
        ]

        return {
            "answer": answer,
            "sources": sources
        }
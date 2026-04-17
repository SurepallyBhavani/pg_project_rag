from __future__ import annotations

from flask import Flask, jsonify, request  # type: ignore
import os
from pathlib import Path
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv  # type: ignore
import openai  # type: ignore

from config import DEFAULT_DATA_ROOT, PERSIST_DIRECTORY
from src.baseline_vector_store import BaselineVectorStore


load_dotenv()

app = Flask(__name__)
store = BaselineVectorStore(str(PERSIST_DIRECTORY))
SYSTEM_READY = False


def initialize_runtime() -> None:
    global SYSTEM_READY
    if PERSIST_DIRECTORY.exists() and store.initialize():
        SYSTEM_READY = True


def process_query(query: str) -> Dict[str, object]:
    if not SYSTEM_READY:
        return {
            "answer": "Baseline vector store is not available yet. Please run `python scripts/build_vector_store.py` first.",
            "sources": [],
            "confidence": None,
            "error": None,
            "retrieval_method": "Vector",
            "query_type": "baseline_vector_query",
        }

    results = store.search(query, k=4)
    if not results:
        return {
            "answer": "No relevant information was found in the baseline corpus for this query.",
            "sources": [],
            "confidence": None,
            "error": None,
            "retrieval_method": "Vector",
            "query_type": "baseline_vector_query",
        }

    context_parts: List[str] = []
    citations: List[str] = []
    snippets: List[str] = []
    scores: List[float] = []
    for doc, distance in results:
        citation = _format_citation(doc.metadata)
        citations.append(citation)
        scores.append(1.0 / (1.0 + float(distance)))
        context_parts.append(f"Source: {citation}\nContent:\n{doc.page_content}")
        snippets.append(doc.page_content[:350].strip())

    context = "\n\n".join(context_parts)
    answer = _try_llm_answer(query, context) or _extractive_answer(query, snippets)
    confidence = round(sum(scores[:3]) / max(min(len(scores), 3), 1), 2)

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(citations)),
        "confidence": confidence,
        "confidence_label": _confidence_label(confidence),
        "error": None,
        "retrieval_method": "Vector",
        "query_type": "baseline_vector_query",
    }


def _try_llm_answer(query: str, context: str) -> Optional[str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    if not api_key:
        return None
    try:
        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a standard vector-RAG academic assistant. "
                        "Answer only from the retrieved context. "
                        "Provide a clear explanatory answer without mentioning confidence."
                    ),
                },
                {"role": "system", "content": f"Context:\n{context}"},
                {"role": "user", "content": query},
            ],
        )
        return completion.choices[0].message.content
    except Exception:
        return None


def _extractive_answer(query: str, snippets: List[str]) -> str:
    query_terms = _tokenize(query)
    selected: List[str] = []
    for snippet in snippets:
        sentences = re.split(r"(?<=[.!?])\s+", snippet.replace("\n", " "))
        best_sentence = ""
        best_overlap = -1
        for sentence in sentences:
            overlap = len(query_terms & _tokenize(sentence))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()
        if best_sentence:
            selected.append(best_sentence)
    if not selected:
        return snippets[0] if snippets else "No grounded answer could be formed."
    return "\n\n".join(selected[:3])


def _format_citation(metadata: Dict[str, object]) -> str:
    subject = str(metadata.get("subject", "general")).upper()
    category = str(metadata.get("category", "material"))
    file_name = str(metadata.get("file_name", "source"))
    return f"[{subject} | {category} | {file_name}]"


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json() or {}
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided."}), 400
    response = process_query(query)
    return jsonify({"success": True, **response})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "EduAssist Baseline RAG",
        "ready": SYSTEM_READY,
        "data_root": str(DEFAULT_DATA_ROOT),
        "persist_directory": str(PERSIST_DIRECTORY),
    })


initialize_runtime()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

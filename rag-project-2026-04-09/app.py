from __future__ import annotations

import warnings
import logging
import os
import sys

# Absolute global warning kill switch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
# Flush stdout and stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from flask import Flask, render_template, request, jsonify  # type: ignore
import json
import os
from pathlib import Path
import re
import time
from collections import Counter
from typing import Dict, List, Optional

from dotenv import load_dotenv  # type: ignore
import markdown2  # type: ignore
import openai  # type: ignore

from src.curriculum.curriculum_extractor import CurriculumExtractor
from src.graph_database.kg_retriever import KnowledgeGraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.question_paper_retriever import QuestionPaperRetriever
from src.retrieval.query_router import QueryRouter
from src.retrieval.reranker import RankedChunk
from src.vector_database.vector_db_manager import VectorDatabaseManager


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = BASE_DIR / "chroma_store"
GRAPH_PATH = BASE_DIR / "artifacts" / "knowledge_graph.json"
FEEDBACK_PATH = BASE_DIR / "artifacts" / "feedback.jsonl"
CURRICULUM_PDF_PATH = BASE_DIR / "data" / "curriculum" / "course_structure" / "syllabus_cse.pdf"

app = Flask(__name__, template_folder="src/web_interface/templates")

DB = None
router = QueryRouter()
curriculum_extractor = CurriculumExtractor(str(CURRICULUM_PDF_PATH))
graph_retriever = KnowledgeGraphRetriever(str(GRAPH_PATH))
question_paper_retriever = QuestionPaperRetriever(str(BASE_DIR / "data"))
hybrid_retriever: Optional[HybridRetriever] = None
vector_db_manager = VectorDatabaseManager(str(PERSIST_DIRECTORY))

SYSTEM_STATUS = {
    "vector_db": False,
    "graph_db": graph_retriever.is_available(),
    "last_processing_time": None,
    "total_documents": 0,
}


def initialize_runtime() -> None:
    global DB, hybrid_retriever

    if not PERSIST_DIRECTORY.exists():
        return

    if not vector_db_manager.initialize():
        return

    DB = vector_db_manager.as_langchain_store()
    hybrid_retriever = HybridRetriever(DB, graph_retriever if graph_retriever.is_available() else None)
    stats = vector_db_manager.get_database_stats()
    SYSTEM_STATUS["vector_db"] = True
    SYSTEM_STATUS["total_documents"] = int(stats.get("total_documents", 0)) if isinstance(stats, dict) else 0
    SYSTEM_STATUS["graph_db"] = graph_retriever.is_available()
    SYSTEM_STATUS["last_processing_time"] = 0


def process_query(query: str, allow_out_of_scope: bool = False, stream: bool = False) -> Dict[str, object]:
    response = _process_query_internal(query, allow_out_of_scope, stream)
    unwanted = r"Dept\.\s*of\s*CSE/JNTUHUCEH\s*B\.Tech\.\s*\(\s*IDP\)\s*w\.e\.f\.\s*202?1\s*-\s*22\s*Academic\s*Year"
    if "answer" in response and isinstance(response["answer"], str):
        response["answer"] = re.sub(unwanted, "", response["answer"], flags=re.IGNORECASE).strip()
    if "sources" in response and isinstance(response["sources"], list):
        response["sources"] = [re.sub(unwanted, "", s, flags=re.IGNORECASE).strip() for s in response["sources"]]
    return response

def _process_query_internal(query: str, allow_out_of_scope: bool = False, stream: bool = False) -> Dict[str, object]:
    route = router.route(query)
    started = time.time()

    if route.query_type == "pyq_query":
        needs_answer = any(w in query.lower() for w in ["answer", "solution", "solve", "answers", "solutions"])
        if not needs_answer or not hybrid_retriever:
            response = _handle_pyq_query(query, route.to_dict())
            response["query_route"] = route.to_dict()
            response["processing_time"] = round(time.time() - started, 3)
            response["source_type"] = "pyq"
            response["graph_context"] = ""
            return response
        else:
            # We need to answer the pyqs!
            hits = question_paper_retriever.retrieve(
                query=query,
                subject=str(route.subject or "") or None,
                topic_candidate=str(route.topic_candidate or "") or None,
                top_k=5,
            )
            if not hits and route.topic_candidate:
                hits = question_paper_retriever.retrieve(query=query, subject=str(route.subject or "") or None, topic_candidate=None, top_k=5)
            
            pyq_context = ""
            if hits:
                pyq_context = "Important Previous Questions Context for answering:\n"
                pyq_context += "\n".join(f"{i+1}. {hit.question}" for i, hit in enumerate(hits))

            # Retrieve textbook material
            if hits:
                retrieval_query = " ".join([h.question for h in hits[:2]])
            else:
                retrieval_query = _build_retrieval_query(query, route.to_dict())
            bundle = hybrid_retriever.retrieve(
                query=retrieval_query,
                filters=route.filters,
                use_kg=route.use_kg,
                top_k=8,
            )
            
            # Embed PYQs into graph_context so they're seen by LLM
            combined_context = (bundle.graph_context + "\n\n" + pyq_context).strip()
            
            answer_payload = _build_grounded_answer(query, route.to_dict(), bundle.ranked_chunks, combined_context, stream=stream)
            answer_payload["query_route"] = route.to_dict()
            answer_payload["processing_time"] = round(time.time() - started, 3)
            answer_payload["source_type"] = "pyq_with_answers"
            answer_payload["graph_context"] = bundle.graph_context
            return answer_payload

    if not hybrid_retriever:
        return _unavailable_response()

    if route.is_gibberish or route.query_type == "no_source":
        response = _formal_no_source_response()
        response["query_route"] = route.to_dict()
        response["processing_time"] = round(time.time() - started, 3)
        return response

    if route.is_out_of_scope:
        response = _handle_out_of_scope_query(query, route.to_dict(), allow_out_of_scope)
        response["query_route"] = route.to_dict()
        response["processing_time"] = round(time.time() - started, 3)
        return response

    if route.is_curriculum_based:
        extracted = curriculum_extractor.answer_query(query, route.topic_candidate)
        if extracted:
            response = {
                "answer": extracted.answer,
                "confidence": 0.9,
                "confidence_label": "high",
                "sources": extracted.citations,
                "evidence": extracted.evidence,
                "error": None,
                "query_route": route.to_dict(),
                "processing_time": round(time.time() - started, 3),
                "source_type": "curriculum_structured",
                "graph_context": "",
            }
            return response

    retrieval_query = _build_retrieval_query(query, route.to_dict())
    top_k = 10 if route.is_curriculum_based else 8
    bundle = hybrid_retriever.retrieve(
        query=retrieval_query,
        filters=route.filters,
        use_kg=route.use_kg,
        top_k=top_k,
    )

    if route.is_curriculum_based:
        bundle.ranked_chunks = _refine_curriculum_chunks(query, route.to_dict(), bundle.ranked_chunks)

    if not bundle.ranked_chunks:
        response = _formal_no_source_response()
        response["query_route"] = route.to_dict()
        response["processing_time"] = round(time.time() - started, 3)
        return response

    answer_payload = _build_grounded_answer(query, route.to_dict(), bundle.ranked_chunks, bundle.graph_context, stream=stream)
    answer_payload["query_route"] = route.to_dict()
    answer_payload["processing_time"] = round(time.time() - started, 3)
    answer_payload["source_type"] = route.retrieval_mode
    answer_payload["graph_context"] = bundle.graph_context
    return answer_payload


def _unavailable_response() -> Dict[str, object]:
    return {
        "answer": "The persistent academic index is not available yet. Please build it with `python scripts/build_vector_store.py` first.",
        "error": None,
        "confidence": None,
        "confidence_label": None,
        "sources": [],
        "evidence": [],
        "query_route": {},
        "processing_time": 0.0,
        "source_type": "unavailable",
        "graph_context": "",
    }


def _formal_no_source_response() -> Dict[str, object]:
    return {
        "answer": (
            "I am unable to provide a grounded response because no relevant information source was found in the "
            "currently indexed academic materials for this query."
        ),
        "error": None,
        "confidence": None,
        "confidence_label": None,
        "sources": [],
        "evidence": [],
        "source_type": "no_source",
        "graph_context": "",
    }


def _handle_pyq_query(query: str, route: Dict[str, object]) -> Dict[str, object]:
    if not question_paper_retriever.is_available():
        return {
            "answer": "Previous question papers are not available in the current project corpus.",
            "error": None,
            "confidence": None,
            "confidence_label": None,
            "sources": [],
            "evidence": [],
        }

    hits = question_paper_retriever.retrieve(
        query=query,
        subject=str(route.get("subject") or "") or None,
        topic_candidate=str(route.get("topic_candidate") or "") or None,
        top_k=6,
    )

    if not hits:
        broad_hits = []
        if route.get("topic_candidate"):
            broad_hits = question_paper_retriever.retrieve(
                query=query,
                subject=str(route.get("subject") or "") or None,
                topic_candidate=None,
                top_k=6,
            )
        if not broad_hits:
            return {
                "answer": (
                    "I could not find matching previous questions for this topic in the currently available question papers."
                ),
                "error": None,
                "confidence": None,
                "confidence_label": None,
                "sources": [],
                "evidence": [],
            }

        subject_label = _subject_label(route)
        topic_label = _topic_label(query, route)
        lines = [
            f"I could not find exact previous questions on {topic_label} in the currently available question papers.",
            "",
            f"However, here are some general previous questions from {subject_label}:",
        ]
        lines.extend(f"{i+1}. {hit.question}" for i, hit in enumerate(broad_hits))
        lines.extend([
            "",
            "If you want, I can also group these into short-answer, long-answer, or most-probable practice questions.",
        ])

        citations = []
        seen = set()
        for hit in broad_hits:
            if hit.citation not in seen:
                seen.add(hit.citation)
                citations.append(hit.citation)

        return {
            "answer": "\n".join(lines),
            "error": None,
            "confidence": None,
            "confidence_label": None,
            "sources": citations,
            "evidence": [],
        }

    topic_label = _topic_label(query, route)
    subject_label = _subject_label(route)
    lines = [
        f"Here are relevant previous questions for {topic_label} from the available {subject_label} question papers.",
        "",
        "Relevant previous questions:",
    ]
    lines.extend(f"{i+1}. {hit.question}" for i, hit in enumerate(hits))
    lines.extend([
        "",
        "If you want, I can also group these into short-answer, long-answer, or most-probable practice questions.",
    ])

    citations = []
    seen = set()
    for hit in hits:
        if hit.citation not in seen:
            seen.add(hit.citation)
            citations.append(hit.citation)

    return {
        "answer": "\n".join(lines),
        "error": None,
        "confidence": None,
        "confidence_label": None,
        "sources": citations,
        "evidence": [],
    }


def _handle_out_of_scope_query(query: str, route: Dict[str, object], allow_out_of_scope: bool) -> Dict[str, object]:
    topic = route.get("topic_candidate") or "the requested subject"

    if not allow_out_of_scope:
        return {
            "answer": (
                f"The requested topic '{topic}' is not in the indexed curriculum. "
                "Please enable the 'Out-of-Scope' option to receive a general explanation."
            ),
            "error": None,
            "confidence": None,
            "confidence_label": None,
            "sources": [],
            "evidence": [],
            "source_type": "unsupported_subject",
            "graph_context": "",
        }

    llm_answer = _try_general_llm_answer(query)
    return {
        "answer": llm_answer or (
            "A grounded response is not available for this subject at present, and a general out-of-scope answer could not be generated."
        ),
        "error": None,
        "confidence": None,
        "confidence_label": None,
        "sources": [],
        "evidence": [],
        "source_type": "out_of_scope_llm",
        "graph_context": "",
    }


def _try_general_llm_answer(query: str) -> Optional[str]:
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
                        "You are an academic assistant. The requested topic is outside the indexed syllabus corpus. "
                        "Provide a clear, formal, student-friendly general explanation. Do not mention citations or confidence."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        return completion.choices[0].message.content
    except Exception:
        return None


def _build_retrieval_query(query: str, route: Dict[str, object]) -> str:
    retrieval_query = query
    normalized_query = query.lower()
    expansions: List[str] = []

    if "goback n" in normalized_query or "goback-n" in normalized_query or "go back n" in normalized_query or "go-back-n" in normalized_query:
        expansions.extend(["go back n", "go-back-n", "gbn", "sliding window", "arq"])
    if "selective repeat" in normalized_query:
        expansions.extend(["selective repeat", "sliding window", "arq"])

    if route.get("is_curriculum_based") and route.get("topic_candidate"):
        retrieval_query = f"{query} {route['topic_candidate']} official syllabus lab textbooks references"
    elif route.get("query_type") == "pyq_query" and route.get("topic_candidate"):
        retrieval_query = f"{query} {route['topic_candidate']} previous question important questions exam questions"
    elif expansions:
        retrieval_query = f"{query} {' '.join(dict.fromkeys(expansions))}"
    return retrieval_query


def _build_grounded_answer(query: str, route: Dict[str, object], ranked_chunks: List[RankedChunk], graph_context: str, stream: bool = False) -> Dict[str, object]:
    top_chunks = ranked_chunks[:7 if route.get("is_curriculum_based") else 5]
    evidence = [_chunk_to_evidence(chunk) for chunk in top_chunks]
    
    needs_answer = any(w in query.lower() for w in ["answer", "solution", "solve", "answers", "solutions"])
    
    gen = None
    if route.get("query_type") == "pyq_query" and not needs_answer:
        answer = _build_pyq_answer(query, route, ranked_chunks)
    else:
        context = _build_context(top_chunks, graph_context)
        if stream:
            gen = _try_grounded_llm_answer(query, context, route, stream=True)
            answer = ""
        else:
            llm_answer = _try_grounded_llm_answer(query, context, route, stream=False)
            answer = llm_answer if llm_answer else _extractive_answer(query, route, top_chunks, graph_context)
    
    confidence = _calculate_confidence(query, route, top_chunks, graph_context)
    if route.get("is_curriculum_based") or route.get("query_type") == "curriculum_query":
        confidence = None

    return {
        "answer": answer,
        "answer_stream": gen,
        "confidence": confidence,
        "confidence_label": _confidence_label(confidence) if confidence is not None else None,
        "sources": _format_citations_list(evidence),
        "evidence": evidence,
        "error": None,
    }


def _refine_curriculum_chunks(query: str, route: Dict[str, object], chunks: List[RankedChunk]) -> List[RankedChunk]:
    topic_candidate = str(route.get("topic_candidate") or "").lower().strip()
    query_terms = _tokenize(query)
    if topic_candidate:
        query_terms |= _tokenize(topic_candidate)

    scored: List[tuple[float, RankedChunk]] = []
    for chunk in chunks:
        text_terms = _tokenize(chunk.content[:1800])
        overlap = len(query_terms & text_terms)
        phrase_bonus = 2 if topic_candidate and topic_candidate in chunk.content.lower() else 0
        lab_bonus = 1 if ("lab" in query.lower() and "lab" in chunk.content.lower()) else 0
        total = overlap + phrase_bonus + lab_bonus
        scored.append((total, chunk))

    scored.sort(key=lambda item: (item[0], item[1].combined_score), reverse=True)
    refined = [chunk for score, chunk in scored if score > 0]
    return refined[:8] if refined else chunks


def _try_grounded_llm_answer(query: str, context: str, route: Dict[str, object], stream: bool = False) -> object:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    if not api_key:
        if stream:
            def err_gen(): yield "OpenRouter API Key not found."
            return err_gen()
        return None

    try:
        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        if route.get("is_curriculum_based"):
            instruction = (
                "You are EduAssist, a formal academic assistant. Answer only from the provided official curriculum context. "
                "If the query asks for syllabus details, lab list, textbooks, references, units, or semester placement, compile the complete available answer from all relevant context. "
                "Do not omit relevant list items that appear in the context. Write formally and clearly. Do not include inline citations or confidence."
            )
        elif route.get("query_type") == "pyq_query":
            instruction = (
                "You are EduAssist, an academic assistant helping with previous question papers.\n"
                "You have been provided with both the actual previous questions (in the context) and textbook material.\n"
                "Please clearly provide grounded answers for each previous question using ONLY the provided textbook context.\n"
                "Format clearly, number each question and answer sequentially, and do not mention citations or confidence."
            )
        else:
            related_topics = _related_topics_text(context, query, route)
            instruction = (
                "You are EduAssist, an academic assistant designed to help students understand complex topics.\n"
                "Use the provided context as a foundation for your answer, but you MUST use your general knowledge to complete the explanation if the context is missing details (e.g., listing all 7 layers of the OSI model).\n"
                "Provide a highly comprehensive, detailed, and formal student-friendly answer.\n"
                "Use highly structured formatting: use bolding for key terms, use numbered lists for sequential steps or architectural layers (like the OSI model), and use bullet points for features or comparisons.\n"
                "Start with a strong introductory paragraph that directly frames the topic.\n"
                "Then provide a deeper explanation that fully addresses the user's question, using academic language. "
                "Where appropriate, briefly mention prerequisite, dependent, or linked topics that help the student understand the concept better. "
                "If the textbook context points to other connected concepts, mention them briefly near the end as further areas the student can explore. "
                "End with a short closing line inviting the user to ask for elaboration on related topics, if any are available. "
                "Do not include citations or confidence level.\n"
                f"Related topics already identified from the indexed material: {related_topics}"
            )

        if stream:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "system", "content": f"Context:\n{context}"},
                    {"role": "user", "content": query},
                ],
                stream=True
            )
            def chunk_generator():
                try:
                    for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                except Exception as e:
                    print(f"LLM API Error (stream): {e}", file=sys.stderr)
                    yield f"\n\n*Error generating stream: {e}*"
            return chunk_generator()
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "system", "content": f"Context:\n{context}"},
                    {"role": "user", "content": query},
                ],
            )
            return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM API Error: {e}", file=sys.stderr)
        if stream:
            def fall_gen(): yield None
            return fall_gen()
        return None


def _extractive_answer(query: str, route: Dict[str, object], chunks: List[RankedChunk], graph_context: str) -> str:
    query_terms = _tokenize(query)
    selected_sentences: List[str] = []

    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.content.replace("\n", " "))
        scored = []
        for sentence in sentences:
            sentence_terms = _tokenize(sentence)
            overlap = len(query_terms & sentence_terms)
            if overlap:
                scored.append((overlap, sentence.strip()))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored:
            selected_sentences.append(scored[0][1])

    if not selected_sentences:
        selected_sentences = [chunks[0].content[:320].strip()]

    if route.get("is_curriculum_based"):
        intro = "Based on the official curriculum document, the relevant syllabus information is as follows.\n\n"
        paragraphs = [selected_sentences[0]]
        if len(selected_sentences) > 1:
            paragraphs.append(selected_sentences[1])
        return intro + "\n\n".join(paragraphs)

    subject_label = _subject_label(route)
    topic_label = _topic_label(query, route)
    intro = (
        f"{topic_label} is an important concept in {subject_label}. "
        "The explanation below is based on the indexed academic materials and is written to give you a clear conceptual understanding before moving into details."
    )

    body_parts: List[str] = []
    if selected_sentences:
        body_parts.append(selected_sentences[0])
    if len(selected_sentences) > 1:
        body_parts.append(
            "To understand it more clearly, it helps to note that "
            + _sentence_to_clause(selected_sentences[1])
            + "."
        )

    elaboration_points = [_normalize_sentence(sentence) for sentence in selected_sentences[2:5]]
    if elaboration_points:
        label = "Key points and elaboration:" if not _query_implies_steps(query) else "Important steps or points involved:"
        body_parts.append(label + "\n" + "\n".join(f"- {point}" for point in elaboration_points))

    related_topics = _extract_related_topics(graph_context, query, route)
    textbook_topics = _extract_textbook_topics(chunks, query, route)
    combined_topics = _merge_topics(related_topics, textbook_topics)
    if related_topics:
        related_text = ", ".join(related_topics[:-1]) + f", and {related_topics[-1]}" if len(related_topics) > 1 else related_topics[0]
        body_parts.append(
            "This topic is also connected to "
            + related_text
            + ". These linked areas are often useful for a fuller understanding of the concept."
        )
    if textbook_topics:
        textbook_text = ", ".join(textbook_topics[:-1]) + f", and {textbook_topics[-1]}" if len(textbook_topics) > 1 else textbook_topics[0]
        body_parts.append(
            "The textbook context also points toward related areas such as "
            + textbook_text
            + ", which can help you extend this concept beyond the immediate definition."
        )

    closing = (
        "If you want, I can also elaborate on "
        + _related_topics_prompt(combined_topics)
        + "."
        if combined_topics
        else "If you want, I can also break this down further with examples, comparisons, or short exam-style notes."
    )

    sections = [intro] + body_parts + [closing]
    return "\n\n".join(section for section in sections if section)


def _build_pyq_answer(query: str, route: Dict[str, object], chunks: List[RankedChunk]) -> str:
    topic_label = _topic_label(query, route)
    subject_label = _subject_label(route)
    extracted_questions = _extract_pyq_questions(chunks, query, route)
    framing_summary = _summarize_question_framing(extracted_questions)

    intro = (
        f"Here are relevant previous questions on {topic_label} from the available {subject_label} question papers. "
        "These reflect the common exam-oriented ways in which this topic has been asked."
    )

    sections = [intro]
    if framing_summary:
        sections.append("This topic is commonly framed in the following ways:\n" + "\n".join(f"- {item}" for item in framing_summary))

    if extracted_questions:
        sections.append("Relevant previous questions:\n" + "\n".join(f"- {question}" for question in extracted_questions[:6]))
    else:
        sections.append(
            "I found question-paper material for this subject, but I could not isolate clean topic-specific questions from the current matches."
        )

    sections.append(
        "If you want, I can also group these into short-answer, long-answer, or most-probable practice questions."
    )
    return "\n\n".join(sections)


def _build_context(chunks: List[RankedChunk], graph_context: str) -> str:
    parts: List[str] = []
    if graph_context:
        parts.append(f"Graph context:\n{graph_context}")

    for chunk in chunks:
        parts.append(
            f"Source: {_format_citation(chunk.metadata)}\n"
            f"Category: {chunk.metadata.get('category', 'unknown')}\n"
            f"Content:\n{chunk.content}"
        )
    return "\n\n".join(parts)


def _chunk_to_evidence(chunk: RankedChunk) -> Dict[str, object]:
    return {
        "citation": _format_citation(chunk.metadata),
        "snippet": chunk.content[:280].strip(),
        "vector_score": round(chunk.vector_score, 3),
        "rerank_score": round(chunk.rerank_score, 3),
        "combined_score": round(chunk.combined_score, 3),
        "subject": chunk.metadata.get("subject"),
        "category": chunk.metadata.get("category"),
    }


def _format_citation(metadata: Dict[str, object]) -> str:
    file_name = str(metadata.get("file_name", metadata.get("relative_path", "source")))
    subject = str(metadata.get("subject", "general")).upper()
    category = str(metadata.get("category", "material"))
    unit = str(metadata.get("unit", "")).strip()
    if unit:
        return f"[{subject} | {category} | {unit} | {file_name}]"
    return f"[{subject} | {category} | {file_name}]"


def _format_citations_list(evidence: List[Dict[str, object]]) -> List[str]:
    citations: List[str] = []
    seen = set()
    for item in evidence:
        citation = str(item["citation"])
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations


def _calculate_confidence(query: str, route: Dict[str, object], chunks: List[RankedChunk], graph_context: str) -> float:
    if not chunks:
        return 0.1
    avg_vector = sum(chunk.vector_score for chunk in chunks[:3]) / max(min(len(chunks), 3), 1)
    avg_rerank = sum(chunk.rerank_score for chunk in chunks[:3]) / max(min(len(chunks), 3), 1)
    query_terms = _tokenize(query)
    context_terms = set()
    for chunk in chunks[:3]:
        context_terms |= _tokenize(chunk.content[:500])
    coverage = len(query_terms & context_terms) / max(len(query_terms), 1)
    
    # Give a massive boost if vector/rerank are reasonably good, scaling them
    # Vector typically ranges 0.3-0.7, mapping >0.4 to >0.8
    vector_boost = (avg_vector * 1.5) if avg_vector > 0.3 else avg_vector
    rerank_boost = (avg_rerank * 1.4) if avg_rerank > 0.3 else avg_rerank

    curriculum_bonus = 0.12 if route.get("is_curriculum_based") else 0.0
    graph_bonus = 0.08 if graph_context and route.get("use_kg") else 0.0
    confidence = (0.42 * vector_boost) + (0.28 * rerank_boost) + (0.18 * coverage) + curriculum_bonus + graph_bonus
    
    # Base confidence padding
    if confidence > 0.4:
        confidence += 0.20
    
    return round(min(max(confidence, 0.15), 0.98), 2)


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}


def _display_retrieval_method(response_data: Dict[str, object]) -> str:
    route_info = response_data.get("query_route", {}) or {}
    source_type = str(response_data.get("source_type") or "")
    graph_context = str(response_data.get("graph_context") or "").strip()

    if route_info.get("is_curriculum_based") or source_type.startswith("curriculum"):
        return "Curriculum"
    if route_info.get("query_type") == "pyq_query" or source_type == "pyq":
        return "PYQ"
    if source_type == "unsupported_subject":
        return "Unsupported Subject"
    if source_type == "out_of_scope_llm":
        return "Out-of-scope LLM"
    if source_type in {"no_source", "unavailable"}:
        return "No Source"
    if route_info.get("use_kg") and graph_context:
        return "Graph-assisted"
    if route_info.get("use_kg"):
        return "Hybrid"
    return "Vector"


def _display_query_type(route_info: Dict[str, object]) -> Optional[str]:
    query_type = str(route_info.get("query_type") or "")
    labels = {
        "curriculum_query": "Curriculum Query",
        "pyq_query": "PYQ Query",
        "supported_subject_content": "Subject Query",
        "unsupported_subject": "Unsupported Subject Query",
        "no_source": "No-source Query",
    }
    return labels.get(query_type, query_type.replace("_", " ").title() if query_type else None)


def _display_subject(route_info: Dict[str, object]) -> Optional[str]:
    subject = str(route_info.get("subject") or "").strip().lower()
    subject_map = {
        "cn": "Computer Networks",
        "dbms": "DBMS",
        "ds": "Data Structures",
        "oops": "OOPS",
        "os": "Operating Systems",
    }
    return subject_map.get(subject) if subject else None


def _subject_label(route: Dict[str, object]) -> str:
    subject = str(route.get("subject") or "").strip().lower()
    subject_map = {
        "cn": "Computer Networks",
        "dbms": "Database Management Systems",
        "ds": "Data Structures",
        "oops": "Object-Oriented Programming",
        "os": "Operating Systems",
    }
    return subject_map.get(subject, "the subject area")


def _topic_label(query: str, route: Dict[str, object]) -> str:
    candidate = str(route.get("topic_candidate") or "").strip()
    if candidate:
        return candidate.title()

    cleaned = re.sub(
        r"\b(explain|define|what is|describe|briefly explain|write short notes on|give short notes on|tell me about)\b",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip(" .?:,-")
    return cleaned.title() if cleaned else "This topic"


def _query_implies_steps(query: str) -> bool:
    return any(term in query.lower() for term in ["steps", "process", "procedure", "algorithm", "how does", "working"])


def _normalize_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip(" -")
    return text[0].upper() + text[1:] if text else text


def _sentence_to_clause(text: str) -> str:
    normalized = _normalize_sentence(text).rstrip(".")
    if not normalized:
        return "the source material provides supporting details"
    return normalized[0].lower() + normalized[1:] if len(normalized) > 1 else normalized.lower()


def _extract_related_topics(graph_context: str, query: str, route: Dict[str, object]) -> List[str]:
    candidates: List[str] = []
    topic_label = _topic_label(query, route).lower()

    for line in graph_context.splitlines():
        matches = re.findall(r"\b[A-Z][A-Za-z0-9+\-/ ]{2,}\b", line)
        for match in matches:
            candidate = re.sub(r"\s+", " ", match).strip(" .,:;")
            lowered = candidate.lower()
            if lowered == topic_label or lowered in {"graph context", "source", "category"}:
                continue
            if candidate and candidate not in candidates:
                candidates.append(candidate)

    if not candidates:
        graph_terms = re.findall(
            r"(?:related to|includes|prerequisite for|appears in)\s+([A-Za-z0-9+\-/ ]+?)(?:[.]\s*|$)",
            graph_context,
            flags=re.IGNORECASE,
        )
        for term in graph_terms:
            candidate = re.sub(r"\s+", " ", term).strip(" .,:;")
            if candidate and candidate.lower() != topic_label and candidate not in candidates:
                candidates.append(candidate)

    return candidates[:4]


def _extract_textbook_topics(chunks: List[RankedChunk], query: str, route: Dict[str, object]) -> List[str]:
    current_topic = _topic_label(query, route).lower()
    discovered: List[str] = []
    banned_terms = {
        "chapter", "page", "example", "include", "includes", "components",
        "administrator", "reference", "concept", "definition",
    }

    for chunk in chunks:
        if str(chunk.metadata.get("category")) != "textbooks":
            continue
        for line in chunk.content.splitlines():
            normalized = re.sub(r"\s+", " ", line).strip(" -:.")
            if len(normalized) < 4 or len(normalized) > 72:
                continue
            lowered = normalized.lower()
            if lowered.startswith("[page"):
                continue
            if current_topic in lowered:
                continue
            if any(term in lowered for term in banned_terms):
                continue
            if any(ch.isdigit() for ch in normalized):
                continue
            if len(normalized.split()) > 6:
                continue
            if not re.search(r"[A-Za-z]", normalized):
                continue
            if not (normalized.istitle() or normalized.isupper()):
                continue
            if normalized not in discovered and (
                normalized.isupper() or normalized.istitle() or re.match(r"^[A-Za-z][A-Za-z0-9 /&()-]+$", normalized)
            ):
                discovered.append(normalized.title() if normalized.isupper() else normalized)
        if len(discovered) >= 4:
            break

    return discovered[:4]


def _extract_pyq_questions(chunks: List[RankedChunk], query: str, route: Dict[str, object]) -> List[str]:
    topic_terms = _tokenize(str(route.get("topic_candidate") or query))
    questions: List[str] = []

    for chunk in chunks:
        text = re.sub(r"\s+", " ", chunk.content.replace("\n", " ")).strip()
        raw_questions = re.findall(
            r"(?:\b\d+\s*[.)]\s*|\b\d+[.][a-z]\)\s*|\b[a-z]\)\s*)([^?]{12,240}\?)",
            text,
            flags=re.IGNORECASE,
        )
        for candidate in raw_questions:
            normalized = _normalize_question(candidate)
            if not normalized:
                continue
            if topic_terms and not (_tokenize(normalized) & topic_terms):
                continue
            if normalized not in questions:
                questions.append(normalized)

    return questions[:8]


def _normalize_question(question: str) -> str:
    cleaned = re.sub(r"\s+", " ", question).strip(" -")
    cleaned = re.sub(r"^(?:or|and)\s+", "", cleaned, flags=re.IGNORECASE)
    if len(cleaned) < 12:
        return ""
    if not cleaned.endswith("?"):
        cleaned += "?"
    return cleaned[0].upper() + cleaned[1:]


def _summarize_question_framing(questions: List[str]) -> List[str]:
    buckets = {
        "Definition-oriented questions": 0,
        "Descriptive or explanatory questions": 0,
        "Comparison-style questions": 0,
        "Short-note or listing questions": 0,
        "Analytical how/why questions": 0,
    }

    for question in questions:
        lowered = question.lower()
        if lowered.startswith(("what is", "define", "write the definition")):
            buckets["Definition-oriented questions"] += 1
        elif lowered.startswith(("compare", "differentiate", "distinguish")):
            buckets["Comparison-style questions"] += 1
        elif lowered.startswith(("list", "enumerate", "write short notes", "write notes")):
            buckets["Short-note or listing questions"] += 1
        elif lowered.startswith(("how", "why")):
            buckets["Analytical how/why questions"] += 1
        else:
            buckets["Descriptive or explanatory questions"] += 1

    return [label for label, count in buckets.items() if count > 0][:4]


def _merge_topics(primary: List[str], secondary: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for topic in primary + secondary:
        key = topic.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(topic)
    return merged[:5]


def _related_topics_prompt(related_topics: List[str]) -> str:
    if not related_topics:
        return "closely related concepts"
    if len(related_topics) == 1:
        return related_topics[0]
    if len(related_topics) == 2:
        return f"{related_topics[0]} or {related_topics[1]}"
    return ", ".join(related_topics[:-1]) + f", or {related_topics[-1]}"


def _related_topics_text(context: str, query: str, route: Dict[str, object]) -> str:
    related_topics = _extract_related_topics(context, query, route)
    return ", ".join(related_topics) if related_topics else "No explicit related topics were identified."


def _documents_available() -> bool:
    return hybrid_retriever is not None and DB is not None


def _feedback_summary() -> Dict[str, object]:
    if not FEEDBACK_PATH.exists():
        return {
            "total_feedback": 0,
            "helpful": 0,
            "not_helpful": 0,
            "top_unhelpful_query_types": [],
            "top_unhelpful_methods": [],
        }

    helpful = 0
    not_helpful = 0
    unhelpful_query_types: Counter[str] = Counter()
    unhelpful_methods: Counter[str] = Counter()

    with FEEDBACK_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = str(item.get("feedback", "")).strip().lower()
            if label == "helpful":
                helpful += 1
            elif label == "not_helpful":
                not_helpful += 1
                query_type = str(item.get("query_type", "unknown")).strip() or "unknown"
                retrieval_method = str(item.get("retrieval_method", "unknown")).strip() or "unknown"
                unhelpful_query_types[query_type] += 1
                unhelpful_methods[retrieval_method] += 1

    return {
        "total_feedback": helpful + not_helpful,
        "helpful": helpful,
        "not_helpful": not_helpful,
        "top_unhelpful_query_types": unhelpful_query_types.most_common(3),
        "top_unhelpful_methods": unhelpful_methods.most_common(3),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    error = None
    sources = None
    query_info = None
    feedback_payload = None
    allow_out_of_scope = False

    documents_available = _documents_available()

    if request.method == "POST" and "query" in request.form:
        query = request.form["query"]
        allow_out_of_scope = request.form.get("allow_out_of_scope") == "on"
        response_data = process_query(query, allow_out_of_scope=allow_out_of_scope)
        if response_data.get("error"):
            error = response_data["error"]
        else:
            answer = markdown2.markdown(str(response_data["answer"]))
            sources = response_data.get("sources", [])
            route_info = response_data.get("query_route", {}) or {}
            query_info = {
                "processing_time": response_data.get("processing_time", 0),
                "confidence": response_data.get("confidence"),
                "confidence_label": response_data.get("confidence_label"),
                "retrieval_method": _display_retrieval_method(response_data),
                "query_type": _display_query_type(route_info),
                "subject": _display_subject(route_info),
            }
            feedback_payload = {
                "query": query,
                "retrieval_method": query_info["retrieval_method"],
                "query_type": query_info["query_type"],
                "subject": query_info["subject"],
                "confidence": response_data.get("confidence"),
                "sources": sources,
                "source_type": response_data.get("source_type"),
            }

    current_system_info = {
        "documents_available": documents_available,
        "feedback_summary": _feedback_summary(),
    }

    return render_template(
        "index.html",
        answer=answer,
        error=error,
        sources=sources,
        query_info=query_info,
        feedback_payload=feedback_payload,
        current_system_info=current_system_info,
        allow_out_of_scope=allow_out_of_scope,
        request=request,
    )


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json() or {}
    query = data.get("query", "")
    allow_out_of_scope = bool(data.get("allow_out_of_scope", False))
    if not query:
        return jsonify({"error": "No query provided."}), 400

    response_data = process_query(query, allow_out_of_scope=allow_out_of_scope)
    return jsonify({
        "success": not bool(response_data.get("error")),
        **response_data,
    })


@app.route("/api/system_status", methods=["GET"])
def api_system_status():
    return jsonify({
        "system_status": SYSTEM_STATUS,
        "documents_loaded": SYSTEM_STATUS.get("total_documents", 0),
        "vector_db_active": _documents_available(),
        "graph_available": graph_retriever.is_available(),
    })


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    payload = request.get_json() or {}
    payload["feedback"] = str(payload.get("feedback", "")).strip().lower()
    payload["submitted_at"] = time.time()
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return jsonify({"success": True})


@app.route("/api/feedback_summary", methods=["GET"])
def api_feedback_summary():
    return jsonify({"success": True, "summary": _feedback_summary()})


initialize_runtime()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

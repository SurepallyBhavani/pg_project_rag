from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Dict, List, Optional


SUPPORTED_SUBJECTS: Dict[str, Dict[str, List[str]]] = {
    "cn": {
        "name": "Computer Networks",
        "aliases": ["cn", "computer network", "computer networks", "networking", "network protocol", "osi", "tcp"],
    },
    "dbms": {
        "name": "Database Management System",
        "aliases": ["dbms", "database management system", "database management systems", "database", "sql", "normalization", "er model", "transaction"],
    },
    "ds": {
        "name": "Data Structures",
        "aliases": ["ds", "data structure", "data structures", "stack", "queue", "tree", "graph", "linked list", "sorting", "searching"],
    },
    "oops": {
        "name": "Object Oriented Programming",
        "aliases": ["oops", "oop", "object oriented programming", "object-oriented programming", "inheritance", "polymorphism", "encapsulation", "abstraction"],
    },
    "os": {
        "name": "Operating System",
        "aliases": ["os", "operating system", "operating systems", "process", "thread", "deadlock", "paging", "memory management", "synchronization"],
    },
}

CURRICULUM_TERMS = [
    "syllabus", "lab", "labs", "textbook", "textbooks", "reference", "references",
    "course structure", "curriculum", "credits", "semester", "course code",
    "prerequisite", "prerequisites", "outcomes", "units", "unit wise", "scheme",
]
PYQ_TERMS = [
    "pyq", "pyqs", "previous question", "previous questions", "question paper",
    "question papers", "important questions", "exam questions", "past questions",
    "asked in exams", "asked in previous papers",
]
SUMMARY_TERMS = ["summary", "summarize", "short notes", "study notes", "revision"]
RELATIONSHIP_TERMS = [
    "difference", "compare", "comparison", "related", "relationship", "connect",
    "affect", "versus", "vs", "prerequisite", "prerequisites", "depends on",
    "dependency", "dependent", "linked", "link", "before", "study before",
    "learn before", "connected to",
]
CONTENT_VERBS = ["what is", "define", "explain", "describe", "how does", "give notes", "short notes"]
TOPIC_SUBJECT_HINTS: Dict[str, List[str]] = {
    "cn": [
        "go back n", "go-back-n", "goback n", "goback-n", "selective repeat",
        "stop and wait", "sliding window", "arq", "crc", "framing", "hamming code",
        "osi model", "tcp/ip", "csma", "routing algorithm",
    ],
    "dbms": [
        "normalization", "normalisation", "functional dependency", "bcnf", "3nf", "sql", "transaction",
        "rdbms", "relational model", "er model", "concurrency control", "indexing",
    ],
    "os": [
        "deadlock", "paging", "thrashing", "process synchronization", "process synchronisation", "banker's algorithm",
        "cpu scheduling", "semaphore", "mutex", "virtual memory", "segmentation",
    ],
    "oops": [
        "inheritance", "polymorphism", "encapsulation", "abstraction", "operator overloading",
        "method overriding", "dynamic binding", "message passing", "virtual function",
    ],
    "ds": [
        "linked list", "queue", "stack", "tree", "graph traversal", "binary search tree",
        "sorting", "searching", "hash table", "spanning tree", "avl tree",
    ],
}


@dataclass
class RouteDecision:
    query_type: str
    retrieval_mode: str
    subject: Optional[str]
    topic_candidate: Optional[str]
    filters: Dict[str, str]
    use_kg: bool
    use_curriculum_only: bool
    is_curriculum_based: bool
    is_subject_based: bool
    is_out_of_scope: bool
    is_gibberish: bool
    needs_summary: bool
    confidence: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class QueryRouter:
    def route(self, query: str) -> RouteDecision:
        query_lower = query.lower().strip()
        reasons: List[str] = []
        topic_candidate = self._extract_topic_candidate(query)
        supported_subject = self._detect_supported_subject(query_lower)
        curriculum_intent = self._is_curriculum_query(query_lower)
        pyq_intent = self._is_pyq_query(query_lower)
        needs_summary = any(term in query_lower for term in SUMMARY_TERMS)
        use_kg = any(term in query_lower for term in RELATIONSHIP_TERMS)
        gibberish = self._is_gibberish(query)

        if gibberish:
            reasons.append("The query appears unrelated to the indexed academic corpus.")
            return RouteDecision(
                query_type="no_source",
                retrieval_mode="none",
                subject=None,
                topic_candidate=topic_candidate,
                filters={},
                use_kg=False,
                use_curriculum_only=False,
                is_curriculum_based=False,
                is_subject_based=False,
                is_out_of_scope=False,
                is_gibberish=True,
                needs_summary=False,
                confidence=0.0,
                reasons=reasons,
            )

        if curriculum_intent:
            reasons.append("Detected curriculum or subject-syllabus intent.")
            if topic_candidate:
                reasons.append(f"Target course/topic identified as: {topic_candidate}")
            return RouteDecision(
                query_type="curriculum_query",
                retrieval_mode="curriculum_only",
                subject=None,
                topic_candidate=topic_candidate,
                filters={"document_group": "curriculum"},
                use_kg=False,
                use_curriculum_only=True,
                is_curriculum_based=True,
                is_subject_based=False,
                is_out_of_scope=False,
                is_gibberish=False,
                needs_summary=needs_summary,
                confidence=0.95,
                reasons=reasons,
            )

        if supported_subject and pyq_intent:
            reasons.append(f"Detected previous-question-paper query for supported subject: {supported_subject}")
            if topic_candidate:
                reasons.append(f"Topic focus identified as: {topic_candidate}")
            return RouteDecision(
                query_type="pyq_query",
                retrieval_mode="pyq",
                subject=supported_subject,
                topic_candidate=topic_candidate,
                filters={"subject": supported_subject, "category": "question_papers"},
                use_kg=False,
                use_curriculum_only=False,
                is_curriculum_based=False,
                is_subject_based=True,
                is_out_of_scope=False,
                is_gibberish=False,
                needs_summary=False,
                confidence=0.88,
                reasons=reasons,
            )

        if supported_subject:
            reasons.append(f"Detected supported subject content query: {supported_subject}")
            if needs_summary:
                reasons.append("Detected summarization intent.")
            if use_kg:
                reasons.append("Detected relationship/dependency wording.")

            return RouteDecision(
                query_type="supported_subject_content",
                retrieval_mode="hybrid" if use_kg else "vector",
                subject=supported_subject,
                topic_candidate=topic_candidate,
                filters={"subject": supported_subject},
                use_kg=use_kg,
                use_curriculum_only=False,
                is_curriculum_based=False,
                is_subject_based=True,
                is_out_of_scope=False,
                is_gibberish=False,
                needs_summary=needs_summary,
                confidence=0.85 if not use_kg else 0.9,
                reasons=reasons,
            )

        if self._looks_like_academic_subject_query(query_lower, topic_candidate):
            reasons.append("Detected an academic subject/course query outside the currently supported five subjects.")
            if topic_candidate:
                reasons.append(f"Unsupported course/topic candidate: {topic_candidate}")
            return RouteDecision(
                query_type="unsupported_subject",
                retrieval_mode="none",
                subject=None,
                topic_candidate=topic_candidate,
                filters={},
                use_kg=False,
                use_curriculum_only=False,
                is_curriculum_based=False,
                is_subject_based=False,
                is_out_of_scope=True,
                is_gibberish=False,
                needs_summary=needs_summary,
                confidence=0.0,
                reasons=reasons,
            )

        reasons.append("No supported subject or curriculum evidence routing rule matched the query.")
        return RouteDecision(
            query_type="no_source",
            retrieval_mode="none",
            subject=None,
            topic_candidate=topic_candidate,
            filters={},
            use_kg=False,
            use_curriculum_only=False,
            is_curriculum_based=False,
            is_subject_based=False,
            is_out_of_scope=False,
            is_gibberish=False,
            needs_summary=False,
            confidence=0.0,
            reasons=reasons,
        )

    def _detect_supported_subject(self, query_lower: str) -> Optional[str]:
        for slug, info in SUPPORTED_SUBJECTS.items():
            for alias in info["aliases"]:
                if re.search(rf"\b{re.escape(alias)}\b", query_lower):
                    return slug
        for slug, hints in TOPIC_SUBJECT_HINTS.items():
            for hint in hints:
                if hint in query_lower:
                    return slug
        return None

    def _is_curriculum_query(self, query_lower: str) -> bool:
        return any(term in query_lower for term in CURRICULUM_TERMS)

    def _is_pyq_query(self, query_lower: str) -> bool:
        return any(term in query_lower for term in PYQ_TERMS)

    def _extract_topic_candidate(self, query: str) -> Optional[str]:
        # Strip conversational adverbs that throw off the regex greedy match
        clean_query = re.sub(
            r"\b(more clearly|in detail|briefly|with examples?|with an example|please|can you)\b", 
            "", 
            query, 
            flags=re.IGNORECASE
        ).strip()
        
        patterns = [
            r"(?:pyqs?|previous questions?|important questions?|exam questions?)\s+(?:on|for|from)\s+(.+?)(?:\?|$)",
            r"(?:questions?|question papers?)\s+(?:on|for)\s+(.+?)(?:\?|$)",
            r"(?:syllabus|lab|labs|textbook|textbooks|references?)\s+(?:of|for)\s+(.+?)(?:\?|$)",
            r"(?:labs?|textbooks?|references?)\s+under\s+(.+?)(?:\?|$)",
            r"(?:what is|define|explain|describe|give notes on|short notes on)\s+(.+?)(?:\?|$)",
            r"(.+?)\s+syllabus(?:\?|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, clean_query, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" .?")
                if candidate:
                    return candidate
        return None

    def _looks_like_academic_subject_query(self, query_lower: str, topic_candidate: Optional[str]) -> bool:
        if topic_candidate and len(topic_candidate.split()) >= 1:
            return True
        return any(verb in query_lower for verb in CONTENT_VERBS)

    def _is_gibberish(self, query: str) -> bool:
        stripped = query.strip()
        if len(stripped) < 3:
            return True
        alpha_chars = sum(ch.isalpha() for ch in stripped)
        if alpha_chars == 0:
            return True
        weird_ratio = sum(not (ch.isalnum() or ch.isspace() or ch in "?.,:-_/") for ch in stripped) / max(len(stripped), 1)
        return weird_ratio > 0.35

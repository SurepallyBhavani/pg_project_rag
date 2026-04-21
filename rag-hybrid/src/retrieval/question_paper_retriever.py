from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional

from pypdf import PdfReader  # type: ignore


@dataclass
class QuestionPaperHit:
    question: str
    citation: str
    subject: str
    score: float


class QuestionPaperRetriever:
    """
    Retrieve previous-year questions, preferring the syllabus-topic index 
    (`data/topic_question_index.json`) for accurate topic-based retrieval.
    Falls back to the legacy keyword-scanning approach if index is absent.
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        # Legacy flat record list (used as fallback)
        self.records: List[QuestionPaperHit] = []
        # New: topic index {subject -> {UNIT-X -> {title, keywords, questions:[]}}}
        self.topic_index: Dict[str, Dict] = {}
        self._load_topic_index()
        self._load_legacy()

    # ──────────────────────────────────────────────────────────────────────
    # Load helpers
    # ──────────────────────────────────────────────────────────────────────
    def _load_topic_index(self) -> None:
        index_path = self.data_root / "topic_question_index.json"
        if not index_path.exists():
            return
        try:
            with open(index_path, encoding="utf-8") as f:
                self.topic_index = json.load(f)
        except Exception:
            self.topic_index = {}

    def _load_legacy(self) -> None:
        """Load flat records from PDFs (used when topic index gives no results)."""
        self.records.clear()
        subject_root = self.data_root / "subjects"
        if not subject_root.exists():
            return

        for pdf_path in sorted(subject_root.rglob("question_papers/*.pdf")):
            subject = self._subject_from_path(pdf_path)
            citation = f"[{subject.upper()} | question_papers | {pdf_path.name}]"
            text = self._extract_pdf_text(pdf_path)
            for question in self._extract_questions(text):
                self.records.append(
                    QuestionPaperHit(
                        question=question,
                        citation=citation,
                        subject=subject,
                        score=0.0,
                    )
                )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def is_available(self) -> bool:
        return bool(self.topic_index) or bool(self.records)

    def retrieve(
        self,
        query: str,
        subject: Optional[str],
        topic_candidate: Optional[str],
        top_k: int = 6,
    ) -> List[QuestionPaperHit]:
        """
        Retrieve questions relevant to *topic_candidate* (and *subject*).
        Tries the syllabus-topic index first, then falls back to legacy approach.
        """
        if self.topic_index and subject and topic_candidate:
            hits = self._retrieve_from_index(subject, topic_candidate, top_k)
            if hits:
                return hits

        # If no topic candidate or index gave nothing, try index with query terms
        if self.topic_index and subject and not topic_candidate:
            hits = self._retrieve_from_index(subject, query, top_k)
            if hits:
                return hits

        # Legacy fallback
        return self._retrieve_legacy(query, subject, topic_candidate, top_k)

    # ──────────────────────────────────────────────────────────────────────
    # Index-based retrieval
    # ──────────────────────────────────────────────────────────────────────
    def _retrieve_from_index(
        self,
        subject: str,
        topic_phrase: str,
        top_k: int,
    ) -> List[QuestionPaperHit]:
        subj_data = self.topic_index.get(subject)
        if not subj_data:
            return []

        unit_topics = subj_data.get("unit_topics", {})
        topic_norm  = self._normalise(topic_phrase)
        topic_tokens = set(re.findall(r"[a-z]{3,}", topic_norm))

        # Step 1: score every unit's title + keywords against the topic phrase
        unit_scores: List[tuple] = []
        for unit_key, unit_data in unit_topics.items():
            score = 0.0
            title_norm = self._normalise(unit_data.get("title", ""))
            keywords   = [kw.lower() for kw in unit_data.get("keywords", [])]

            # Exact phrase match in title → very strong signal
            if topic_norm and topic_norm in title_norm:
                score += 5.0
            # Token overlap with title
            title_tokens = set(re.findall(r"[a-z]{3,}", title_norm))
            score += len(topic_tokens & title_tokens) * 2.0
            # Keyword matches (longer keywords worth more)
            for kw in keywords:
                if kw in topic_norm or topic_norm in kw:
                    score += len(kw.split()) ** 1.2
                elif all(tok in kw for tok in topic_tokens if len(tok) >= 4):
                    score += 1.0

            if score > 0:
                unit_scores.append((score, unit_key, unit_data))

        if not unit_scores:
            # No unit matched – try gathering from all units using topic tokens
            return self._collect_from_all_units(
                unit_topics, topic_tokens, topic_norm, subject, top_k
            )

        unit_scores.sort(key=lambda x: x[0], reverse=True)

        # Step 2: collect questions from the best-matching units, scored by
        #         how well each question mentions the topic phrase
        hits: List[QuestionPaperHit] = []
        seen: set = set()
        for _, unit_key, unit_data in unit_scores[:2]:  # top-2 units
            for qd in unit_data.get("questions", []):
                q   = qd["question"]
                src = qd.get("source", "question_papers")
                lq  = q.lower()
                if lq in seen:
                    continue
                seen.add(lq)

                # Score this specific question against the user's topic
                q_score = self._score_question_vs_topic(q, topic_norm, topic_tokens)
                citation = f"[{subject.upper()} | question_papers | {src}]"
                hits.append(
                    QuestionPaperHit(
                        question=q,
                        citation=citation,
                        subject=subject,
                        score=q_score,
                    )
                )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def _collect_from_all_units(
        self,
        unit_topics: Dict,
        topic_tokens: set,
        topic_norm: str,
        subject: str,
        top_k: int,
    ) -> List[QuestionPaperHit]:
        """Scan every question in every unit for topic-token overlap."""
        hits: List[QuestionPaperHit] = []
        seen: set = set()
        for unit_data in unit_topics.values():
            for qd in unit_data.get("questions", []):
                q   = qd["question"]
                lq  = q.lower()
                if lq in seen:
                    continue
                sc = self._score_question_vs_topic(q, topic_norm, topic_tokens)
                if sc > 0:
                    seen.add(lq)
                    citation = f"[{subject.upper()} | question_papers | {qd.get('source', 'question_papers')}]"
                    hits.append(QuestionPaperHit(
                        question=q,
                        citation=citation,
                        subject=subject,
                        score=sc,
                    ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def _score_question_vs_topic(
        self,
        question: str,
        topic_norm: str,
        topic_tokens: set,
    ) -> float:
        ql = question.lower()
        score = 0.0
        # Exact topic phrase match in question - best signal
        if topic_norm and topic_norm in ql:
            score += 4.0
        # Token overlap
        q_tokens = set(re.findall(r"[a-z]{3,}", ql))
        overlap   = topic_tokens & q_tokens
        score    += len(overlap) * 1.5
        # Small quality bonus
        if 20 <= len(question) <= 200:
            score += 0.3
        return score

    # ──────────────────────────────────────────────────────────────────────
    # Legacy keyword-scan retrieval (unchanged from original)
    # ──────────────────────────────────────────────────────────────────────
    def _retrieve_legacy(
        self,
        query: str,
        subject: Optional[str],
        topic_candidate: Optional[str],
        top_k: int,
    ) -> List[QuestionPaperHit]:
        query_terms   = self._tokenize(topic_candidate or query)
        strict_terms  = self._topic_focus_terms(topic_candidate)
        strict_mode   = bool(topic_candidate and strict_terms)
        filtered      = [r for r in self.records if not subject or r.subject == subject]

        scored: List[QuestionPaperHit] = []
        for record in filtered:
            q_terms = self._tokenize(record.question)
            overlap = len(query_terms & q_terms)
            if query_terms and overlap == 0 and topic_candidate:
                continue
            if strict_mode and not self._contains_topic_phrase(record.question, topic_candidate or "") \
                    and not self._contains_all_focus_terms(record.question, strict_terms):
                continue
            score = overlap / max(len(query_terms), 1)
            if topic_candidate and self._contains_topic_phrase(record.question, topic_candidate):
                score += 0.75
            if self._looks_clean(record.question):
                score += 0.2
            if not topic_candidate:
                score += 0.2
            scored.append(QuestionPaperHit(
                question=record.question,
                citation=record.citation,
                subject=record.subject,
                score=score,
            ))

        scored.sort(key=lambda item: (item.score, len(item.question)), reverse=True)

        deduped: List[QuestionPaperHit] = []
        seen: set = set()
        for hit in scored:
            key = hit.question.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
            if len(deduped) >= top_k:
                break

        return deduped

    # ──────────────────────────────────────────────────────────────────────
    # PDF parsing (legacy)
    # ──────────────────────────────────────────────────────────────────────
    def _extract_pdf_text(self, path: Path) -> str:
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "") for page in reader.pages]
            return "\n".join(pages)
        except Exception:
            return ""

    def _extract_questions(self, text: str) -> List[str]:
        normalized = text.replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n+", "\n", normalized)
        blocks = re.split(r"(?:^|\n)\s*(?:\d+\s*[.)]|[a-zA-Z]\))\s*", normalized, flags=re.MULTILINE)
        questions: List[str] = []
        seen: set = set()
        for block in blocks:
            for question in self._extract_questions_from_block(block):
                lowered = question.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                questions.append(question)
        return questions

    def _extract_questions_from_block(self, block: str) -> List[str]:
        cleaned_block = re.sub(r"\s+", " ", block).strip(" -")
        if len(cleaned_block) < 12:
            return []

        fragments = re.split(r"(?:(?:\b[a-zA-Z]\)|\\bOR\\b)|\[ *\d+\+?\d* *\])", cleaned_block, flags=re.IGNORECASE)
        questions: List[str] = []
        for fragment in fragments:
            candidate = fragment.strip(" -")
            if len(candidate) < 12:
                continue
            qm_matches = re.findall(r"[^?]{8,180}\?", candidate)
            if qm_matches:
                for match in qm_matches:
                    question = self._clean_question(match)
                    if question:
                        questions.append(question)
                continue
            imperative_match = re.match(
                r"(?i)\b(explain|define|compare|differentiate|distinguish|discuss|describe|what is|write short notes on|list|enumerate|give an example of|draw and explain)\b.{8,180}",
                candidate,
            )
            if imperative_match:
                question = self._clean_question(imperative_match.group(0))
                if question:
                    questions.append(question)
        return questions

    def _clean_question(self, question: str) -> str:
        cleaned = re.sub(r"\s+", " ", question).strip(" -")
        cleaned = re.sub(r"^(?:part\s*[ab]|or)\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bR1[0-9]\b", "", cleaned, flags=re.IGNORECASE).strip()
        if len(cleaned) < 12:
            return ""
        if not cleaned.endswith("?") and self._is_imperative_question(cleaned):
            cleaned += "?"
        if not cleaned.endswith("?"):
            return ""
        return cleaned[0].upper() + cleaned[1:]

    def _is_imperative_question(self, text: str) -> bool:
        return bool(
            re.match(
                r"(?i)\b(explain|define|compare|differentiate|distinguish|discuss|describe|what is|write short notes on|list|enumerate|give an example of|draw and explain)\b",
                text,
            )
        )

    # ──────────────────────────────────────────────────────────────────────
    # Utility methods
    # ──────────────────────────────────────────────────────────────────────
    def _normalise(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _contains_topic_phrase(self, question: str, topic_candidate: str) -> bool:
        normalized_question = re.sub(r"\s+", " ", question.lower())
        normalized_topic = re.sub(r"\s+", " ", topic_candidate.lower())
        return normalized_topic in normalized_question

    def _topic_focus_terms(self, topic_candidate: Optional[str]) -> List[str]:
        if not topic_candidate:
            return []
        stop_terms = {"show", "give", "important", "previous", "questions", "question", "paper", "papers", "in", "of", "for", "on"}
        raw_terms = [
            token for token in re.findall(r"[a-zA-Z0-9_]+", topic_candidate.lower())
            if len(token) >= 3 and token not in stop_terms
        ]
        return raw_terms[:3]

    def _contains_focus_terms(self, question: str, focus_terms: List[str]) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in focus_terms)

    def _contains_all_focus_terms(self, question: str, focus_terms: List[str]) -> bool:
        lowered = question.lower()
        return all(term in lowered for term in focus_terms)

    def _looks_clean(self, question: str) -> bool:
        if len(question) > 180:
            return False
        if question.count("?") != 1:
            return False
        noisy_markers = ["part", "marks", "time:", "max. marks", "jawaharlal", "university"]
        lowered = question.lower()
        return not any(marker in lowered for marker in noisy_markers)

    def _subject_from_path(self, path: Path) -> str:
        parts = list(path.parts)
        if "subjects" in parts:
            index = parts.index("subjects")
            if len(parts) > index + 1:
                return parts[index + 1]
        return "unknown"

    def _tokenize(self, text: str) -> set:
        return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}

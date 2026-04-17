from __future__ import annotations

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
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.records: List[QuestionPaperHit] = []
        self._load()

    def _load(self) -> None:
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

    def is_available(self) -> bool:
        return bool(self.records)

    def retrieve(self, query: str, subject: Optional[str], topic_candidate: Optional[str], top_k: int = 6) -> List[QuestionPaperHit]:
        query_terms = self._tokenize(topic_candidate or query)
        strict_terms = self._topic_focus_terms(topic_candidate)
        strict_topic_mode = bool(topic_candidate and strict_terms)
        subject_filtered = [record for record in self.records if not subject or record.subject == subject]

        scored: List[QuestionPaperHit] = []
        for record in subject_filtered:
            question_terms = self._tokenize(record.question)
            overlap = len(query_terms & question_terms)
            if query_terms and overlap == 0 and topic_candidate:
                continue
            if strict_topic_mode and not self._contains_topic_phrase(record.question, topic_candidate or "") and not self._contains_all_focus_terms(record.question, strict_terms):
                continue
            score = overlap / max(len(query_terms), 1)
            if topic_candidate and self._contains_topic_phrase(record.question, topic_candidate):
                score += 0.75
            if self._looks_clean(record.question):
                score += 0.2
            if not topic_candidate:
                score += 0.2
            scored.append(
                QuestionPaperHit(
                    question=record.question,
                    citation=record.citation,
                    subject=record.subject,
                    score=score,
                )
            )

        scored.sort(key=lambda item: (item.score, len(item.question)), reverse=True)

        deduped: List[QuestionPaperHit] = []
        seen = set()
        for hit in scored:
            key = hit.question.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
            if len(deduped) >= top_k:
                break

        return deduped

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
        seen = set()
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

        fragments = re.split(r"(?:(?:\b[a-zA-Z]\)|\bOR\b)|\[ *\d+\+?\d* *\])", cleaned_block, flags=re.IGNORECASE)
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
            if len(token) > 3 and token not in stop_terms
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

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}

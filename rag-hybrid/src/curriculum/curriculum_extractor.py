from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional

from pypdf import PdfReader  # type: ignore


SECTION_HEADERS = [
    "Prerequisites",
    "Objectives",
    "Outcomes",
    "UNIT - I",
    "UNIT - II",
    "UNIT - III",
    "UNIT - IV",
    "UNIT - V",
    "Textbook",
    "Textbooks",
    "References",
    "List of Experiments",
]


@dataclass
class CurriculumHit:
    answer: str
    citations: List[str]
    evidence: List[Dict[str, object]]


class CurriculumExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.pages: List[str] = []
        self.course_pages: List[Dict[str, object]] = []
        self.course_structure_pages: List[Dict[str, object]] = []
        if self.pdf_path.exists():
            self._load()

    def _load(self) -> None:
        reader = PdfReader(str(self.pdf_path))
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            self.pages.append(text)
            if "S. No." in text and "Course Title" in text:
                self.course_structure_pages.append({"page": idx, "text": text})
            if any(header in text for header in ("Prerequisites", "Objectives", "Outcomes")):
                title = self._extract_course_title(text)
                if title:
                    self.course_pages.append({"page": idx, "title": title, "text": text})

        self._expand_course_page_spans()

    def _expand_course_page_spans(self) -> None:
        if not self.course_pages:
            return

        sorted_pages = sorted(self.course_pages, key=lambda item: int(item["page"]))
        expanded: List[Dict[str, object]] = []

        for index, item in enumerate(sorted_pages):
            start_page = int(item["page"])
            next_start_page = int(sorted_pages[index + 1]["page"]) if index + 1 < len(sorted_pages) else len(self.pages) + 1
            combined_parts: List[str] = [str(item["text"])]

            for page_num in range(start_page + 1, next_start_page):
                page_text = self.pages[page_num - 1]
                if self._looks_like_new_course_page(page_text):
                    break
                combined_parts.append(page_text)

            expanded.append({
                "page": start_page,
                "title": item["title"],
                "text": "\n\n".join(part for part in combined_parts if part.strip()),
            })

        self.course_pages = expanded

    def answer_query(self, query: str, topic_candidate: Optional[str]) -> Optional[CurriculumHit]:
        if not self.pages:
            return None

        target = topic_candidate
        if not target:
            # Fallback: Try to detect subject in query
            from src.retrieval.query_router import QueryRouter
            router = QueryRouter()
            detected_slug = router._detect_supported_subject(query.lower())
            if detected_slug:
                from src.retrieval.query_router import SUPPORTED_SUBJECTS
                target = SUPPORTED_SUBJECTS[detected_slug]["name"]
            else:
                target = query
        
        course_page = self._find_best_course_page(target)
        query_lower = query.lower()
        wants_labs = bool(re.search(r"\blabs?\b", query_lower))
        wants_textbooks = bool(re.search(r"\btextbooks?\b|\breferences?\b", query_lower))
        wants_prereq = bool(re.search(r"\bprerequisites?\b", query_lower))
        wants_outcomes = bool(re.search(r"\boutcomes?\b", query_lower))
        wants_units = bool(re.search(r"\bunits?\b", query_lower))
        wants_syllabus = bool(re.search(r"\bsyllabus\b", query_lower)) or (not wants_labs and not wants_textbooks and not wants_prereq and not wants_outcomes and not wants_units)

        if wants_labs:
            labs_answer = self._build_labs_answer(target)
            if labs_answer:
                return labs_answer

        if course_page:
            answer = self._build_course_answer(course_page, wants_syllabus, wants_textbooks, wants_prereq, wants_outcomes, wants_units)
            if answer:
                citation = self._cite(course_page["page"])
                return CurriculumHit(
                    answer=answer,
                    citations=[citation],
                    evidence=[{"citation": citation, "snippet": course_page["text"][:350].strip()}],
                )

        return None

    def _find_best_course_page(self, target: str) -> Optional[Dict[str, object]]:
        from src.retrieval.query_router import SUPPORTED_SUBJECTS
        
        target_norm = self._normalize(target)
        
        # Check if target matches any subject aliases and use official name
        expanded_targets = [target_norm]
        for slug, info in SUPPORTED_SUBJECTS.items():
            if target_norm in [self._normalize(a) for a in info["aliases"]]:
                expanded_targets.append(self._normalize(info["name"]))
                break
        
        best = None
        best_score = 0
        for page in self.course_pages:
            title_norm = self._normalize(str(page["title"]))
            
            # Check against original and expanded targets
            for t in expanded_targets:
                score = self._match_score(t, title_norm)
                if score > best_score:
                    best = page
                    best_score = score
                    
        return best if best_score > 0 else None

    def _build_course_answer(
        self,
        page: Dict[str, object],
        wants_syllabus: bool,
        wants_textbooks: bool,
        wants_prereq: bool,
        wants_outcomes: bool,
        wants_units: bool,
    ) -> str:
        text = str(page["text"])
        title = str(page["title"])
        semester_line = self._find_semester_line(text)
        ltpc_line = self._extract_ltpc_line(text)

        parts: List[str] = []
        if wants_syllabus:
            units = self._extract_units(text)
            parts.append(f"The official syllabus structure available for {title} is as follows.")
            if semester_line:
                parts.append(f"Semester: {semester_line}")
            if ltpc_line:
                parts.append(f"L-T-P-C: {ltpc_line}")
            if units:
                parts.append("Units:\n" + units)
            return "\n\n".join(parts)

        if wants_prereq:
            prerequisites = self._extract_section(text, "Prerequisites", ["Objectives"])
            if prerequisites:
                return f"The official prerequisites listed for {title} are:\n\n{prerequisites}"

        if wants_outcomes:
            outcomes = self._extract_section(text, "Outcomes", ["UNIT - I"])
            if outcomes:
                return f"The official course outcomes listed for {title} are:\n\n{outcomes}"

        if wants_units:
            units = self._extract_units(text)
            if units:
                return f"The syllabus units listed for {title} are:\n\n{units}"

        if wants_textbooks:
            textbook = self._extract_section(text, "Textbook", ["References"]) or self._extract_section(text, "Textbooks", ["References"])
            references = self._extract_section(text, "References", [])
            parts = [f"The officially listed learning resources for {title} are as follows."]
            if textbook:
                parts.append("Textbook:\n" + textbook)
            if references:
                parts.append("References:\n" + references)
            return "\n\n".join(parts)

        return ""

    def _build_labs_answer(self, target: str) -> Optional[CurriculumHit]:
        lab_page = self._find_best_course_page(f"{target} lab")
        if lab_page:
            text = str(lab_page["text"])
            title = str(lab_page["title"])
            objectives = self._extract_section(text, "Objectives", ["Outcomes"])
            outcomes = self._extract_section(text, "Outcomes", ["List of Experiments", "Textbook", "Textbooks"])
            experiments = self._extract_section(text, "List of Experiments", ["Textbook", "Textbooks", "References"])
            textbook = self._extract_section(text, "Textbook", ["References"]) or self._extract_section(text, "Textbooks", ["References"])
            references = self._extract_section(text, "References", [])

            parts = [f"The official lab syllabus available for {title} is as follows."]
            if objectives:
                parts.append("Objectives:\n" + objectives)
            if outcomes:
                parts.append("Outcomes:\n" + outcomes)
            if experiments:
                parts.append("List of Experiments:\n" + experiments)
            if textbook:
                parts.append("Textbook:\n" + textbook)
            if references:
                parts.append("References:\n" + references)

            citation = self._cite(lab_page["page"])
            return CurriculumHit(
                answer="\n\n".join(parts),
                citations=[citation],
                evidence=[{"citation": citation, "snippet": text[:350].strip()}],
            )

        structure_match = self._find_course_structure_page(target)
        if not structure_match:
            return None

        page = structure_match["page"]
        semester = structure_match["semester"]
        course_title = structure_match["course_title"]
        labs = structure_match["labs"]
        if not labs:
            return None

        answer = (
            f"According to the official curriculum, the labs listed in the same semester as {course_title} "
            f"({semester}) are:\n\n" + "\n".join(f"- {lab}" for lab in labs)
        )
        citation = self._cite(page)
        return CurriculumHit(
            answer=answer,
            citations=[citation],
            evidence=[{"citation": citation, "snippet": structure_match["text"][:350].strip()}],
        )

    def _find_course_structure_page(self, target: str) -> Optional[Dict[str, object]]:
        target_norm = self._normalize(target)
        for item in self.course_structure_pages:
            text = str(item["text"])
            if target_norm and target_norm in self._normalize(text):
                segment = self._extract_matching_semester_segment(text, target_norm)
                semester = self._extract_nearest_semester(segment or text)
                rows = self._extract_course_rows(segment or text)
                labs = [row for row in rows if "lab" in row.lower()]
                return {
                    "page": item["page"],
                    "text": segment or text,
                    "semester": semester or "the corresponding semester",
                    "course_title": target,
                    "labs": labs,
                }
        return None

    def _extract_matching_semester_segment(self, text: str, target_norm: str) -> str:
        blocks = re.split(r"(?=(?:[IVX]+\s+YEAR.*?SEMESTER))", text, flags=re.IGNORECASE | re.DOTALL)
        for block in blocks:
            if target_norm and target_norm in self._normalize(block):
                return block.strip()
        return text

    def _extract_course_rows(self, text: str) -> List[str]:
        rows: List[str] = []
        for line in text.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip()
            if re.match(r"^\d+\s+[A-Z*].+", cleaned):
                rows.append(cleaned)
        return rows

    def _extract_nearest_semester(self, text: str) -> str:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            if "SEMESTER" in line.upper():
                return line
        return ""

    def _extract_course_title(self, text: str) -> Optional[str]:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
        for idx, line in enumerate(lines[:8]):
            if self._looks_like_title(line):
                return line
        return None

    def _looks_like_new_course_page(self, text: str) -> bool:
        title = self._extract_course_title(text)
        return bool(title and any(header in text for header in ("Prerequisites", "Objectives", "Outcomes")))

    def _looks_like_title(self, line: str) -> bool:
        if len(line) < 5 or len(line) > 70:
            return False
        if any(token in line for token in ["Dept.", "Academic Year", "B.Tech", "Semester", "L T P C", "Objectives", "Outcomes"]):
            return False
        letters = [ch for ch in line if ch.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        return uppercase_ratio > 0.75

    def _extract_section(self, text: str, start_header: str, end_headers: List[str]) -> str:
        pattern = re.escape(start_header)
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return ""
        start = match.end()
        end = len(text)
        for header in end_headers:
            next_match = re.search(re.escape(header), text[start:], re.IGNORECASE)
            if next_match:
                end = min(end, start + next_match.start())
        section = text[start:end].strip()
        return self._clean_block(section)

    def _extract_units(self, text: str) -> str:
        # Flexible regex for UNIT headers (hyphen, colon, space, or none)
        unit_matches = list(re.finditer(r"UNIT\s*[:\-\s]?\s*([IVX]+)", text, re.IGNORECASE))
        if not unit_matches:
            return ""
        blocks: List[str] = []
        for index, match in enumerate(unit_matches):
            start = match.start()
            end = unit_matches[index + 1].start() if index + 1 < len(unit_matches) else len(text)
            block = text[start:end].strip()
            if "Textbook" in block:
                block = block.split("Textbook", 1)[0].strip()
            blocks.append(self._format_unit_block(block))
        return "\n\n".join(block for block in blocks if block)

    def _clean_block(self, block: str) -> str:
        lines = [re.sub(r"\s+", " ", line).strip() for line in block.splitlines() if line.strip()]
        return "\n".join(lines)

    def _find_semester_line(self, text: str) -> str:
        for line in text.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip()
            if "Semester" in cleaned or "SEMESTER" in cleaned:
                return cleaned
        return ""

    def _extract_ltpc_line(self, text: str) -> str:
        match = re.search(r"L\s*T\s*P\s*C\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}-{match.group(4)}"
        return ""

    def _format_unit_block(self, block: str) -> str:
        cleaned = self._clean_block(block)
        lines = cleaned.splitlines()
        if not lines:
            return ""
        heading = lines[0].replace("UNIT -", "UNIT ").strip()
        body = " ".join(line.strip() for line in lines[1:]).strip()
        body = re.sub(r"\s+", " ", body)
        return f"{heading}: {body}" if body else heading

    def _cite(self, page: int) -> str:
        return f"[CURRICULUM | course_structure | syllabus_cse.pdf | Page {page}]"

    def _normalize(self, text: str) -> str:
        lowered = text.lower()
        lowered = lowered.replace("&", " and ")
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _match_score(self, left: str, right: str) -> int:
        if not left or not right:
            return 0
        if left == right:
            return 100
        if left in right or right in left:
            return 70
        left_terms = set(left.split())
        right_terms = set(right.split())
        overlap = left_terms & right_terms
        if overlap:
            return len(overlap) * 10
        return 0

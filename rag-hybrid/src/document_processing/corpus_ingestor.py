"""
Corpus ingestion utilities for the reorganized EduAssist dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile
from typing import Dict, Iterable, List

from pypdf import PdfReader  # type: ignore


SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt"}
SKIP_FOLDERS = {"__pycache__"}
OCR_PROCESSED_FOLDER = "ocr_processed"
OCR_PENDING_FOLDER = "ocr_pending"


@dataclass
class CorpusDocument:
    text: str
    metadata: Dict[str, str]


class CorpusIngestor:
    """Load first-pass ingestible files from the curated corpus."""

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)

    def collect_documents(self) -> List[CorpusDocument]:
        selected_documents: Dict[tuple[str, str, str], CorpusDocument] = {}

        for file_path in sorted(self._iter_supported_files()):
            text = self._extract_text(file_path)
            cleaned_text = self._normalize_text(text)
            if len(cleaned_text) < 100:
                continue

            metadata = self._build_metadata(file_path)
            dedupe_key = (
                metadata.get("subject", ""),
                metadata.get("category", ""),
                metadata.get("original_name", metadata.get("file_name", "")),
            )
            candidate = CorpusDocument(
                text=cleaned_text,
                metadata=metadata,
            )
            existing = selected_documents.get(dedupe_key)
            if existing and self._source_priority(existing.metadata) >= self._source_priority(metadata):
                continue
            selected_documents[dedupe_key] = candidate

        return list(selected_documents.values())

    def _iter_supported_files(self) -> Iterable[Path]:
        for path in self.data_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if any(part in SKIP_FOLDERS for part in path.parts):
                continue
            yield path

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf_text(path)
        if suffix == ".pptx":
            return self._extract_pptx_text(path)
        if suffix == ".docx":
            return self._extract_docx_text(path)
        if suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        return ""

    def _extract_pdf_text(self, path: Path) -> str:
        reader = PdfReader(str(path))
        pages: List[str] = []

        for index, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(f"[Page {index}]\n{page_text}")

        return "\n\n".join(pages)

    def _extract_docx_text(self, path: Path) -> str:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
        return self._strip_xml(xml)

    def _extract_pptx_text(self, path: Path) -> str:
        slides: List[str] = []
        with zipfile.ZipFile(path) as archive:
            slide_names = sorted(
                name for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            )
            for slide_index, slide_name in enumerate(slide_names, start=1):
                xml = archive.read(slide_name).decode("utf-8", errors="ignore")
                slide_text = self._strip_xml(xml)
                if slide_text.strip():
                    slides.append(f"[Slide {slide_index}]\n{slide_text}")

        return "\n\n".join(slides)

    def _strip_xml(self, xml: str) -> str:
        return re.sub(r"<[^>]+>", " ", xml)

    def _normalize_text(self, text: str) -> str:
        normalized_lines: List[str] = []
        seen = set()

        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if len(line) < 2:
                continue

            lowered = line.lower()
            if lowered.startswith("[page ") or lowered.startswith("[slide "):
                normalized_lines.append(line)
                continue

            if lowered in seen:
                continue
            seen.add(lowered)
            normalized_lines.append(line)

        return "\n".join(normalized_lines).strip()

    def _build_metadata(self, path: Path) -> Dict[str, str]:
        relative_path = path.relative_to(self.data_root)
        parts = list(relative_path.parts)
        document_group = parts[0] if parts else "unknown"
        if document_group == "curriculum":
            subject = "curriculum"
            category = parts[1] if len(parts) > 1 else "course_structure"
            is_ocr_processed = "false"
            original_name = path.name
        else:
            subject = parts[1] if len(parts) > 1 else "unknown"
            category, is_ocr_processed, original_name = self._resolve_subject_category(parts, path)

        return {
            "source": str(path),
            "relative_path": str(relative_path).replace("\\", "/"),
            "file_name": path.name,
            "original_name": original_name,
            "file_type": path.suffix.lower().lstrip("."),
            "document_group": document_group,
            "subject": subject,
            "category": category,
            "unit": self._infer_unit(parts),
            "is_subject_syllabus": "true" if "syllabus" in parts else "false",
            "is_ocr_processed": is_ocr_processed,
            "is_ocr_pending": "true" if OCR_PENDING_FOLDER in parts else "false",
            "ocr_status": self._ocr_status(parts),
        }

    def _infer_unit(self, parts: List[str]) -> str:
        for part in parts:
            if part.lower().startswith("unit"):
                return part
        return ""

    def _resolve_subject_category(self, parts: List[str], path: Path) -> tuple[str, str, str]:
        if len(parts) > 2 and parts[2] == OCR_PROCESSED_FOLDER:
            if len(parts) > 3:
                return parts[3], "true", path.name
            inferred = self._infer_category_from_name(path.name)
            return inferred, "true", path.name

        if len(parts) > 2 and parts[2] == OCR_PENDING_FOLDER:
            inferred = self._infer_category_from_name(path.name)
            return inferred, "false", path.name

        category = parts[2] if len(parts) > 2 else "root"
        return category, "false", path.name

    def _ocr_status(self, parts: List[str]) -> str:
        if OCR_PROCESSED_FOLDER in parts:
            return "processed"
        if OCR_PENDING_FOLDER in parts:
            return "pending"
        return "not_applicable"

    def _source_priority(self, metadata: Dict[str, str]) -> int:
        if metadata.get("ocr_status") == "processed":
            return 3
        if metadata.get("ocr_status") == "pending":
            return 2
        return 1

    def _infer_category_from_name(self, file_name: str) -> str:
        lowered = file_name.lower()
        if any(term in lowered for term in ["syllabus", "curriculum"]):
            return "syllabus"
        if any(term in lowered for term in ["slide", "ppt"]):
            return "slides"
        if any(term in lowered for term in ["textbook", "book", "reference"]):
            return "textbooks"
        if any(term in lowered for term in ["assign", "question paper", "qp"]):
            return "assignments"
        if any(term in lowered for term in ["program", "lab", "record", "code"]):
            return "programs"
        return "notes"

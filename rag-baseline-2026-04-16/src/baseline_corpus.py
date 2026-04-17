from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile
from typing import Dict, Iterable, List

from pypdf import PdfReader  # type: ignore


SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt"}
SKIP_FOLDERS = {"ocr_pending", "__pycache__"}


@dataclass
class BaselineDocument:
    text: str
    metadata: Dict[str, str]


class BaselineCorpusLoader:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def collect_documents(self) -> List[BaselineDocument]:
        documents: List[BaselineDocument] = []
        for file_path in sorted(self._iter_supported_files()):
            text = self._extract_text(file_path)
            cleaned = self._normalize_text(text)
            if len(cleaned) < 100:
                continue
            documents.append(
                BaselineDocument(
                    text=cleaned,
                    metadata=self._build_metadata(file_path),
                )
            )
        return documents

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
        return {
            "source": str(path),
            "relative_path": str(relative_path).replace("\\", "/"),
            "file_name": path.name,
            "subject": parts[1] if len(parts) > 1 and parts[0] == "subjects" else "curriculum",
            "category": parts[2] if len(parts) > 2 and parts[0] == "subjects" else "course_structure",
        }

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt"}


def main() -> int:
    data_root = PROJECT_ROOT / "data" / "subjects"
    pending_files = sorted(
        path for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and "ocr_pending" in path.parts
    )

    if not pending_files:
        print("No OCR-pending files were found.")
        return 0

    grouped = defaultdict(list)
    processed_total = 0

    for path in pending_files:
        subject = _subject_from_path(path)
        relative_pending = path.relative_to(data_root / subject / "ocr_pending")
        processed_path = data_root / subject / "ocr_processed" / relative_pending
        grouped[subject].append((path, processed_path, processed_path.exists()))
        if processed_path.exists():
            processed_total += 1

    print("OCR Audit Report")
    print("================")
    print(f"Total pending source files: {len(pending_files)}")
    print(f"Matching OCR-processed files: {processed_total}")
    print("")

    for subject in sorted(grouped):
        print(f"{subject.upper()}")
        for pending_path, processed_path, exists in grouped[subject]:
            status = "READY" if exists else "PENDING"
            print(f"  - {status}: {pending_path.name}")
            print(f"    raw:      {pending_path}")
            print(f"    expected: {processed_path}")
        print("")

    return 0


def _subject_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "subjects" in parts:
        index = parts.index("subjects")
        if len(parts) > index + 1:
            return parts[index + 1]
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())

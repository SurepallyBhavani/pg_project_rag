from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "subjects"
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy OCR-cleaned files into the matching ocr_processed folders."
    )
    parser.add_argument(
        "source_root",
        help="Folder containing OCR-cleaned files to import.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing file in ocr_processed if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without copying or moving files.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    if not source_root.exists() or not source_root.is_dir():
        print(f"Source folder not found: {source_root}")
        return 1

    pending_map = _build_pending_map()
    if not pending_map:
        print("No OCR-pending files were found in the project.")
        return 0

    source_files = [
        path for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not source_files:
        print(f"No supported OCR-cleaned files were found under: {source_root}")
        return 1

    matches = 0
    skipped = 0

    print("OCR Import Plan")
    print("===============")
    for source_path in sorted(source_files):
        normalized_name = _normalize_name(source_path.name)
        target_path = pending_map.get(normalized_name)
        if not target_path:
            print(f"SKIP   {source_path.name}")
            print("       reason: no matching OCR-pending file name in the project")
            skipped += 1
            continue

        action = "COPY"
        if args.move:
            action = "MOVE"

        if target_path.exists() and not args.overwrite:
            print(f"SKIP   {source_path.name}")
            print(f"       reason: target already exists at {target_path}")
            skipped += 1
            continue

        print(f"{action:<6} {source_path.name}")
        print(f"       source: {source_path}")
        print(f"       target: {target_path}")
        matches += 1

        if args.dry_run:
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if args.move:
            if target_path.exists() and args.overwrite:
                target_path.unlink()
            shutil.move(str(source_path), str(target_path))
        else:
            shutil.copy2(str(source_path), str(target_path))

    print("")
    print(f"Matched files: {matches}")
    print(f"Skipped files: {skipped}")
    print("Next step: run `python scripts/build_vector_store.py` to include the new OCR-cleaned files.")
    return 0


def _build_pending_map() -> dict[str, Path]:
    pending_map: dict[str, Path] = {}
    for pending_path in DATA_ROOT.rglob("*"):
        if not pending_path.is_file():
            continue
        if pending_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if "ocr_pending" not in pending_path.parts:
            continue

        subject = _subject_from_path(pending_path)
        relative_pending = pending_path.relative_to(DATA_ROOT / subject / "ocr_pending")
        processed_path = DATA_ROOT / subject / "ocr_processed" / relative_pending
        pending_map[_normalize_name(pending_path.name)] = processed_path
    return pending_map


def _subject_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "subjects" in parts:
        index = parts.index("subjects")
        if len(parts) > index + 1:
            return parts[index + 1]
    raise ValueError(f"Unable to determine subject from path: {path}")


def _normalize_name(file_name: str) -> str:
    stem = Path(file_name).stem.lower()
    normalized = "".join(ch for ch in stem if ch.isalnum())
    return f"{normalized}{Path(file_name).suffix.lower()}"


if __name__ == "__main__":
    raise SystemExit(main())

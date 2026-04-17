from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (BASE_DIR.parent / "rag-project-2026-04-09" / "data").resolve()
PERSIST_DIRECTORY = BASE_DIR / "chroma_store_baseline"

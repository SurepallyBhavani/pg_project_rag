from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (BASE_DIR.parent / "rag-hybrid" / "data").resolve()
PERSIST_DIRECTORY = BASE_DIR / "chroma_store_baseline"

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_DATA_ROOT, PERSIST_DIRECTORY
from src.baseline_corpus import BaselineCorpusLoader
from src.baseline_vector_store import BaselineVectorStore


def main() -> int:
    loader = BaselineCorpusLoader(str(DEFAULT_DATA_ROOT))
    documents = loader.collect_documents()
    if not documents:
        print("No baseline documents found.")
        return 1

    store = BaselineVectorStore(str(PERSIST_DIRECTORY))
    texts = [doc.text for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    print(f"Preparing baseline index for {len(documents)} source documents...")
    if not store.rebuild(texts, metadatas):
        print("Baseline vector store build failed.")
        return 1

    print(f"Baseline vector store created at: {PERSIST_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

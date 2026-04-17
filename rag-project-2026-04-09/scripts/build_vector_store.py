from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_processing.corpus_ingestor import CorpusIngestor
from src.graph_database.kg_builder import KnowledgeGraphBuilder
from src.vector_database.vector_db_manager import VectorDatabaseManager


def main() -> int:
    data_root = PROJECT_ROOT / "data"
    persist_directory = PROJECT_ROOT / "chroma_store"
    graph_path = PROJECT_ROOT / "artifacts" / "knowledge_graph.json"

    ingestor = CorpusIngestor(str(data_root))
    documents = ingestor.collect_documents()
    if not documents:
        print("No ingestible documents were found.")
        return 1

    manager = VectorDatabaseManager(str(persist_directory))
    texts = [doc.text for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    print(f"Preparing to index {len(documents)} source documents...")
    if not manager.replace_documents(texts, metadatas):
        print("Vector store build failed.")
        return 1

    graph_builder = KnowledgeGraphBuilder()
    graph_builder.build(documents, str(graph_path))

    print("Vector store build completed.")
    print(manager.get_database_stats())
    print(f"Knowledge graph written to {graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.retrieval.reranker import HeuristicReranker, RankedChunk


@dataclass
class RetrievalBundle:
    ranked_chunks: List[RankedChunk]
    graph_context: str


class HybridRetriever:
    def __init__(self, vector_store, graph_retriever=None):
        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
        self.reranker = HeuristicReranker()

    def retrieve(self, query: str, filters: Optional[Dict[str, str]] = None, use_kg: bool = False, top_k: int = 5) -> RetrievalBundle:
        normalized_filters = filters or {}
        vector_candidates = self._vector_search(query, normalized_filters, top_k=40)
        reranked_chunks = self.reranker.rerank(query, vector_candidates, top_k=max(top_k * 3, 20))
        ranked_chunks = self._select_balanced_chunks(reranked_chunks, normalized_filters, top_k)
        graph_context = ""
        if use_kg and self.graph_retriever:
            graph_context = self.graph_retriever.query_to_context(query, subject=normalized_filters.get("subject"))
        return RetrievalBundle(ranked_chunks=ranked_chunks, graph_context=graph_context)

    def _vector_search(self, query: str, filters: Dict[str, str], top_k: int) -> List[RankedChunk]:
        chroma_filter = self._build_chroma_filter(filters)
        raw_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=chroma_filter,
        )

        ranked: List[RankedChunk] = []
        for doc, distance in raw_results:
            vector_score = 1.0 / (1.0 + float(distance))
            ranked.append(
                RankedChunk(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    vector_score=vector_score,
                    rerank_score=0.0,
                    combined_score=vector_score,
                )
            )
        return ranked

    def _select_balanced_chunks(self, chunks: List[RankedChunk], filters: Dict[str, str], top_k: int) -> List[RankedChunk]:
        if filters.get("document_group") == "curriculum":
            return chunks[:top_k]

        if not filters.get("subject"):
            return chunks[:top_k]

        selected: List[RankedChunk] = []
        used_ids = set()

        def add_first_matching(categories: set[str], count: int = 1) -> None:
            added = 0
            for chunk in chunks:
                chunk_id = self._chunk_identity(chunk)
                if chunk_id in used_ids:
                    continue
                if str(chunk.metadata.get("category")) in categories:
                    selected.append(chunk)
                    used_ids.add(chunk_id)
                    added += 1
                    if added >= count:
                        return

        add_first_matching({"textbooks"}, count=3)
        add_first_matching({"notes", "slides"}, count=1)

        for chunk in chunks:
            chunk_id = self._chunk_identity(chunk)
            if chunk_id in used_ids:
                continue
            selected.append(chunk)
            used_ids.add(chunk_id)
            if len(selected) >= top_k:
                break

        selected.sort(key=lambda item: item.combined_score, reverse=True)
        return selected[:top_k]

    def _chunk_identity(self, chunk: RankedChunk) -> str:
        relative_path = str(chunk.metadata.get("relative_path", ""))
        content_key = chunk.content[:120]
        return f"{relative_path}|{content_key}"

    def _build_chroma_filter(self, filters: Dict[str, str]):
        if not filters:
            return None
        items = [{key: value} for key, value in filters.items() if value]
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        return {"$and": items}

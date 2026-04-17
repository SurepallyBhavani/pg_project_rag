from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, List


@dataclass
class RankedChunk:
    content: str
    metadata: Dict[str, object]
    vector_score: float
    rerank_score: float
    combined_score: float


class HeuristicReranker:
    """A lightweight reranker that works offline."""

    def rerank(self, query: str, chunks: List[RankedChunk], top_k: int = 5) -> List[RankedChunk]:
        query_terms = self._tokenize(query)
        normalized_query = self._normalize_text(query)

        ranked: List[RankedChunk] = []
        for chunk in chunks:
            chunk_terms = self._tokenize(chunk.content)
            normalized_chunk = self._normalize_text(chunk.content)
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            phrase_bonus = 0.2 if normalized_query and normalized_query in normalized_chunk else 0.0
            topic_bonus = self._topic_bonus(normalized_query, normalized_chunk)
            category = str(chunk.metadata.get("category"))
            syllabus_bonus = 0.1 if category == "syllabus" else 0.0
            
            if category == "textbooks":
                source_bonus = 0.80
            elif category in {"notes", "slides"}:
                source_bonus = 0.05
            else:
                source_bonus = 0.0
                
            rerank_score = min(1.0, overlap + phrase_bonus + topic_bonus + syllabus_bonus + source_bonus)
            combined = (0.50 * chunk.vector_score) + (0.50 * rerank_score) if category == "textbooks" else (0.65 * chunk.vector_score) + (0.35 * rerank_score)
            ranked.append(
                RankedChunk(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    vector_score=chunk.vector_score,
                    rerank_score=rerank_score,
                    combined_score=combined,
                )
            )

        ranked.sort(key=lambda item: item.combined_score, reverse=True)
        return ranked[:top_k]

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _topic_bonus(self, normalized_query: str, normalized_chunk: str) -> float:
        topic_patterns = {
            "go back n": ["go back n", "gbn"],
            "selective repeat": ["selective repeat"],
            "stop and wait": ["stop and wait"],
            "sliding window": ["sliding window"],
        }
        bonus = 0.0
        for _, variants in topic_patterns.items():
            if any(variant in normalized_query for variant in variants):
                if any(variant in normalized_chunk for variant in variants):
                    bonus += 0.25
        return min(0.3, bonus)

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple


class KnowledgeGraphRetriever:
    def __init__(self, graph_path: str):
        self.graph_path = Path(graph_path)
        self.graph = {"nodes": {}, "adjacency": {}}
        if self.graph_path.exists():
            self.graph = json.loads(self.graph_path.read_text(encoding="utf-8"))

    def is_available(self) -> bool:
        return bool(self.graph.get("nodes"))

    def query_to_context(self, query: str, subject: Optional[str] = None) -> str:
        if not self.is_available():
            return ""

        concepts = self._find_matching_nodes(query, subject)
        if not concepts:
            return ""

        lines: List[str] = []
        for node_id, payload in concepts[:3]:
            lines.extend(self._describe_node(node_id, payload))

        # Keep the graph context concise.
        deduped = []
        seen = set()
        for line in lines:
            key = line.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(line)
        return "\n".join(deduped[:8])

    def _find_matching_nodes(self, query: str, subject: Optional[str]) -> List[Tuple[str, Dict[str, str]]]:
        query_terms = self._tokenize(query)
        matches: List[Tuple[float, str, Dict[str, str]]] = []
        for node_id, payload in self.graph.get("nodes", {}).items():
            if payload.get("type") not in {"concept", "unit", "subject"}:
                continue
            if subject and payload.get("subject") not in {subject, None, ""} and payload.get("name") != subject:
                continue
            name_terms = self._tokenize(payload.get("name", ""))
            overlap = len(query_terms & name_terms)
            if overlap:
                matches.append((overlap / max(len(name_terms), 1), node_id, payload))

        matches.sort(key=lambda item: item[0], reverse=True)
        return [(node_id, payload) for _, node_id, payload in matches]

    def _describe_node(self, node_id: str, payload: Dict[str, str]) -> List[str]:
        lines: List[str] = []
        name = payload.get("name", "")
        neighbors = self.graph.get("adjacency", {}).get(node_id, [])
        for edge in neighbors[:5]:
            target_id = edge["target"]
            target = self.graph["nodes"].get(target_id, {})
            target_name = target.get("name", target_id)
            relation = edge["relation"]
            if relation == "prerequisite_of":
                lines.append(f"{name} is a prerequisite for {target_name}.")
            elif relation == "contains":
                lines.append(f"{name} includes {target_name}.")
            elif relation == "related_to":
                lines.append(f"{name} is related to {target_name}.")
            elif relation == "mentions":
                lines.append(f"{name} appears in {target_name}.")
        return lines

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}

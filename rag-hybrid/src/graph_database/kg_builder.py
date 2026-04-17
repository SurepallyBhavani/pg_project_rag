from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

from src.document_processing.corpus_ingestor import CorpusDocument


class KnowledgeGraphBuilder:
    """Build a lightweight concept graph from the curated corpus."""

    def build(self, documents: Iterable[CorpusDocument], output_path: str) -> Dict[str, object]:
        output = Path(output_path)
        graph = {
            "nodes": {},
            "adjacency": defaultdict(list),
        }

        subject_unit_map: Dict[str, set[str]] = defaultdict(set)

        for doc in documents:
            subject = doc.metadata.get("subject", "curriculum")
            category = doc.metadata.get("category", "unknown")
            unit = doc.metadata.get("unit", "")
            subject_node = f"subject::{subject}"
            self._add_node(graph, subject_node, {"type": "subject", "name": subject})

            if unit:
                unit_node = f"unit::{subject}::{unit.lower()}"
                self._add_node(graph, unit_node, {"type": "unit", "name": unit, "subject": subject})
                self._add_edge(graph, subject_node, unit_node, "has_unit")
                subject_unit_map[subject].add(unit)
            else:
                unit_node = None

            document_node = f"document::{doc.metadata.get('relative_path')}"
            self._add_node(
                graph,
                document_node,
                {
                    "type": "document",
                    "name": doc.metadata.get("file_name", ""),
                    "subject": subject,
                    "category": category,
                },
            )
            self._add_edge(graph, subject_node, document_node, "has_document")
            if unit_node:
                self._add_edge(graph, unit_node, document_node, "contains")

            concepts = self._extract_concepts(doc.text)[:20]
            previous_concept = None
            for concept in concepts:
                concept_node = f"concept::{subject}::{concept.lower()}"
                self._add_node(graph, concept_node, {"type": "concept", "name": concept, "subject": subject})
                self._add_edge(graph, subject_node, concept_node, "contains")
                self._add_edge(graph, document_node, concept_node, "mentions")
                if unit_node:
                    self._add_edge(graph, unit_node, concept_node, "contains")
                if previous_concept and previous_concept != concept_node:
                    self._add_edge(graph, previous_concept, concept_node, "related_to")
                previous_concept = concept_node

        for subject, units in subject_unit_map.items():
            ordered_units = sorted(units, key=self._unit_sort_key)
            for current, nxt in zip(ordered_units, ordered_units[1:]):
                self._add_edge(
                    graph,
                    f"unit::{subject}::{current.lower()}",
                    f"unit::{subject}::{nxt.lower()}",
                    "prerequisite_of",
                )

        serializable = {
            "nodes": graph["nodes"],
            "adjacency": dict(graph["adjacency"]),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return serializable

    def _extract_concepts(self, text: str) -> List[str]:
        concepts: List[str] = []
        seen = set()
        for raw_line in text.splitlines():
            line = raw_line.strip(" -:\t")
            if len(line) < 4 or len(line) > 80:
                continue
            if line.startswith("[Page") or line.startswith("[Slide"):
                continue
            if not re.search(r"[A-Za-z]", line):
                continue
            cleaned = re.sub(r"^[0-9.()]+\s*", "", line)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned.split()) > 7:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            if cleaned.isupper() or cleaned.istitle() or re.match(r"^[A-Za-z][A-Za-z0-9 /&-]+$", cleaned):
                seen.add(lowered)
                concepts.append(cleaned)
        return concepts

    def _add_node(self, graph: Dict[str, object], node_id: str, payload: Dict[str, str]) -> None:
        graph["nodes"][node_id] = payload

    def _add_edge(self, graph: Dict[str, object], source: str, target: str, relation: str) -> None:
        edge = {"target": target, "relation": relation}
        if edge not in graph["adjacency"][source]:
            graph["adjacency"][source].append(edge)

    def _unit_sort_key(self, unit_name: str) -> Tuple[int, str]:
        match = re.search(r"(\d+)", unit_name)
        if match:
            return int(match.group(1)), unit_name
        return 999, unit_name

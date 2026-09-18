# pg_project_rag

A syllabus-grounded Retrieval-Augmented Generation system for academic Q&A, built end-to-end solo — corpus ingestion, hybrid retrieval, LLM-based generation, an evaluation harness, and the frontend.

## Contents

- **[`rag-hybrid/`](rag-hybrid/)** — the main system. Hybrid vector + knowledge-graph retrieval, intent-based query routing, reranking, and grounded generation via LLM APIs. See [`rag-hybrid/README.md`](rag-hybrid/README.md) for the full architecture, prompting approach, and setup instructions.
- **`rag-baseline/`** — a simpler vector-only RAG baseline, used as a comparison point for evaluating the main system's retrieval and reranking design.
- **`comparison-benchmark/`** — evaluation harness and scripts comparing the main system against the baseline and a generic (non-retrieval) LLM.

## Quick start

See [`rag-hybrid/README.md`](rag-hybrid/README.md) — that's the project to run.

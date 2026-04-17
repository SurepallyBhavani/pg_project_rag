# EduAssist Baseline RAG

This is a separate baseline system for research comparison.

It is intentionally simpler than the enhanced EduAssist system:
- no curriculum extractor
- no knowledge graph retrieval
- no hybrid routing
- no dedicated PYQ path
- no structured curriculum handling

It follows a standard vector-RAG flow:
1. ingest documents
2. chunk text
3. build embeddings
4. store in Chroma
5. retrieve top-k chunks
6. generate grounded answer from retrieved context

## Isolation

This baseline is fully separate from:
- [rag-project-2026-04-09](C:/Users/Administrator/Desktop/rag%20trial%202%20success/rag-project-2026-04-09)

It writes only to its own folder:
- `chroma_store_baseline/`

By default it reads the shared corpus from:
- `../rag-project-2026-04-09/data`

## Build

```bash
python scripts/build_vector_store.py
```

## Run

```bash
python app.py
```

Default URL:
- `http://127.0.0.1:5002`

## API

POST `/api/ask`

JSON body:

```json
{
  "query": "What is normalization in DBMS?"
}
```

## Purpose

Use this only as a standard vector-RAG baseline for comparison with:
- general LLM baseline
- enhanced EduAssist

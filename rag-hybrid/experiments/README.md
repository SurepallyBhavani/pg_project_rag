# Experiments (superseded, not used by the live app)

These modules were early design iterations that were superseded and are **not imported anywhere in `app.py`** or any live script. Kept for reference only.

- `document_processing/pdf_loader.py` — first prototype: raw OpenAI API + GPT-4, no OpenRouter. Superseded by `src/document_processing/corpus_ingestor.py`.
- `document_processing/document_processor.py` — spaCy/NLTK-based named-entity and relationship extraction (typed patterns like `CAUSES`, `DEFINED_AS`, `SIMILAR_TO`) for building graph embeddings. Superseded by the much simpler heading/line-heuristic concept extraction in `src/graph_database/kg_builder.py`.
- `graph_database/graph_db_manager.py` — Neo4j-backed graph store with a NetworkX fallback. Superseded by the flat JSON adjacency-list graph (`artifacts/knowledge_graph.json`) read by `src/graph_database/kg_retriever.py`.
- `graph_database/graph_query_processor.py` — query layer for `graph_db_manager.py`; unused for the same reason.
- `query_processing/query_classifier.py` — TF-IDF + Naive Bayes classifier meant to route simple (vector) vs. complex (graph) queries. Superseded by the plain keyword/regex routing in `src/retrieval/query_router.py`.
- `models/query_classifier_model.pkl` — trained artifact for the classifier above.
- `settings.py` — an early `Config` class referencing Neo4j and a DeepSeek API key. The live app reads its own env vars directly and never imports this file.

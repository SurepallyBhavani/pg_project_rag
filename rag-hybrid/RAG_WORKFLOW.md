# EduAssist RAG Workflow Explanation

This document provides a systematic breakdown of the Retrieval-Augmented Generation (RAG) workflow used in this project. It is divided into two main phases: the offline setup (Data Ingestion) and the runtime processing (Query Pipeline).

---

## Phase 1: Offline Setup (Data Ingestion)
*This is the process of generating your academic index, which runs before your server starts when setting up the project.*

**Goal:** Convert raw academic PDFs (syllabi, notes, past year questions) into readable, searchable chunks, and store them efficiently so the system can retrieve them later without reading the raw files.

1. **Document Loading & Extraction:**
   - **What Happens:** The system scans the `data/` folder and loads the raw text out of your PDFs, cleaning out noise. For syllabus documents, it extracts specific structures (units, textbook references, lab exercises). 
   - **Tools/Tech:** `PyPDF` (for loading PDFs), `spacy` and `textblob` (for natural language parsing and cleaning).
   - **Python Files:** `src/document_processing/pdf_loader.py`, `src/document_processing/corpus_ingestor.py`, and `src/document_processing/document_processor.py`.

2. **Chunking & Embedding (The Core of RAG):**
   - **What Happens:** The cleaned text is broken down into semantic "chunks" (paragraphs of text) so they easily fit into a search context. These chunks are passed through an embedding model which turns the text into dense mathematical vectors (arrays of floating-point numbers). This represents the "meaning" of the text.
   - **Tools/Tech:** `sentence-transformers` running on `PyTorch`. 
   - **Python Files:** Triggered by `scripts/build_vector_store.py`.

3. **Database Population:**
   - **What Happens:** The vectors (embeddings) and metadata (file names, units, subjects) are permanently saved into a Vector Database. Simultaneously, a knowledge graph is generated to map relationship patterns between topics.
   - **Tools/Tech:** `ChromaDB` (Vector DB) and `NetworkX`/`Neo4j` (Graph DB).
   - **Python Files:** `src/vector_database/vector_db_manager.py` (saves to `chroma_store/`), and `src/graph_database/kg_retriever.py` (reads `artifacts/knowledge_graph.json`).

---

## Phase 2: Runtime Processing (Query Pipeline)
*This happens entirely in the background after you press "Submit" on the web interface (`app.py`).*

**Goal:** Understand the user's intent, quickly find the most accurate academic evidence, and intelligently construct an answer without hallucinating.

1. **Query Routing & Classification:**
   - **What Happens:** When the query arrives, the system doesn't immediately search. First, it classifies the query. Is it asking about a syllabus? Is it a Past Year Question (PYQ)? Is it completely out-of-scope? It understands the intent and routes the query appropriately.
   - **Tools/Tech:** `scikit-learn` or regex-based query classification algorithms.
   - **Python Files:** `app.py` (`process_query` function), `src/retrieval/query_router.py`, and `src/query_processing/query_classifier.py`.

2. **Hybrid Retrieval (Vector + Graph Search):**
   - **What Happens:** Based on the route, the system goes into the databases. It takes the user's query, embeds it into a vector, and pulls the most similar chunks from ChromaDB. Simultaneously, it looks up the topic in the Knowledge Graph to find associated overarching nodes.
     - **Exceptions:** If the router classified it as a PYQ, it pulls exclusively from previous papers. If it's a syllabus request, it targets the `curriculum_extractor`. 
   - **Tools/Tech:** `ChromaDB`, `NetworkX`.
   - **Python Files:** `src/retrieval/hybrid_retriever.py`, `src/retrieval/question_paper_retriever.py`, and `src/graph_database/kg_retriever.py`.

3. **Reranking:**
   - **What Happens:** The system pulls more chunks than it needs (often 8 or 10). The Reranker evaluates these extracted chunks heavily focusing on keyword overlap, syllabus relevance, and vector similarity to re-order them so the best, most correct chunk is placed at rank #1.
   - **Python Files:** `src/retrieval/reranker.py` and `app.py` (`_refine_curriculum_chunks`).

4. **Generation & Grounding (Strictly Answering):**
   - **What Happens:** `app.py` aggregates the top reranked chunks into one large "context" payload. It sends this context—alongside your query and a very strict "do not hallucinate" prompt—to the Large Language Model. The LLM reads only the context provided to answer the query.
   - **Fallback:** If offline or the LLM fails, the system executes an `_extractive_answer` where it literally glues together the most relevant sentences directly from your textbook chunks.
   - **Tools/Tech:** `LangChain` and the `OpenRouter API` (using models like `gpt-4o-mini`).
   - **Python Files:** `app.py` (specifically `_try_grounded_llm_answer` and `_build_grounded_answer`).

5. **Confidence Scoring & Returning (Final Mile):**
   - **What Happens:** The script runs `_calculate_confidence()` mapping mathematical similarity metrics to a Human readable "Confidence Label" (High/Medium/Low). It bundles the answer, the confidence score, and the exact Source Citations.
   - **Tools/Tech:** `Flask` (to send the JSON bundle to the frontend).
   - **Python Files:** `app.py`.

---

## Technical Details

### Retrieval & Chunk Limits
The number of chunks retrieved and selected depends dynamically on the *type* of query, directed by the router in `app.py`.

* **Standard Academic Queries / PYQs:**
  * **Retrieved:** The system fetches the top **8** chunks from Chroma.
  * **Selected to LLM:** It selects the top **5** best chunks after reranking.
* **Curriculum / Syllabus Queries:**
  * **Retrieved:** The system widens the net and retrieves the top **10** chunks.
  * **Selected to LLM:** It selects the top **7** chunks. (Syllabus details are often fragmented across multiple list items, requiring broader context).

*Code reference: `app.py` Line 190 and 425*

### The Confidence Calculation Formula
The confidence score is not a random guess by the LLM. It is a strictly calculated mathematical formula located in **`app.py`** under the **`_calculate_confidence()`** function (Lines 708–732). The system only evaluates the **Top 3 chunks** to ensure the score reflects the highest-quality evidence.

1. **Averaging Base Scores:** Averages the raw `vector_score` (similarity distance) and the `rerank_score` (keyword relevance) of the Top 3 chunks.
2. **Lexical Coverage:** Calculates an "overlap" percentage. Checks how many crucial words from your query actually appear in the first 500 characters of those Top 3 chunks.
3. **Activation Boosts:** If an average score is "reasonably good" (> 0.3), the system artificially "boosts" it:
   * Vector > 0.3 gets multiplied by **1.5**
   * Rerank > 0.3 gets multiplied by **1.4**
4. **Weighted Formula:** 
   `Base Score = (42% of Vector Boost) + (28% of Rerank Boost) + (18% of Coverage)`
5. **Contextual Bonuses:**
   * **+0.12** (+12%) if the query relies on official curriculum records.
   * **+0.08** (+8%) if Knowledge Graph successfully assisted the retrieval.
6. **Padding & Capping:** Uncapped bonuses exceeding `0.40` get an automatic `+0.20` padding. Finally, the total confidence is mathematically restricted to never fall below **15% (0.15)** and never exceed **98% (0.98)**.

# EduAssist — Syllabus-Grounded Hybrid RAG Academic Assistant

A Retrieval-Augmented Generation assistant that answers student questions strictly from an indexed set of official syllabi, textbooks, lecture notes, and previous-year question papers — rather than a general-purpose LLM's parametric knowledge. Every answer is grounded in cited source passages, with an explicit "I don't have that in the indexed material" fallback instead of a guess.

Built end-to-end solo: corpus ingestion, retrieval, generation, evaluation harness, and the frontend. Covers five subjects — Computer Networks (CN), DBMS, Operating Systems (OS), OOP (OOPS), and Data Structures (DS).

## The problem this was scoped to solve

General-purpose LLMs answer confidently from broad parametric knowledge, which is exactly the wrong behavior for exam prep — a technically-plausible answer that doesn't match the specific syllabus, textbook edition, or professor's phrasing is worse than no answer. The scope I set for myself: retrieval must be constrained to the indexed corpus, every answer must cite what it's grounded in, and the system must be able to say "not in the material" rather than fabricate — and I needed a way to actually measure whether that was working, not just eyeball a few responses.

## Architecture — the RAG pipeline

Each query goes through:

1. **Query routing** (`src/retrieval/query_router.py`) — keyword/regex-based classification into one of five routes: `curriculum_query`, `pyq_query`, `supported_subject_content`, `unsupported_subject`, `no_source`/gibberish. A `TOPIC_SUBJECT_HINTS` dictionary maps specific terms (e.g. "deadlock", "paging" → OS) to a subject even when the query never names the course.
2. **Retrieval**, which differs by route:
   - **Curriculum queries** bypass the vector store entirely — `src/curriculum/curriculum_extractor.py` parses the syllabus PDF directly, locating a course's page span by section headers (`Prerequisites`, `UNIT - I`, etc.).
   - **PYQ queries** are answered from a precomputed topic index (`data/topic_question_index.json`, unit → keywords → questions) scored by phrase/keyword overlap, with a regex-based PDF fallback — also not vector search.
   - **Everything else** goes through `src/retrieval/hybrid_retriever.py`: ChromaDB vector search (cosine similarity over `all-MiniLM-L6-v2` embeddings) → `HeuristicReranker` (token overlap, phrase match, syllabus-topic bonuses, and a source-authority weight that favors textbook passages over supplementary slides/notes) → a balanced-selection step that reserves slots for both textbook and supplementary-source chunks so the LLM never sees five near-duplicate high-scoring passages → final top-K sent to the LLM. Relationship-style queries ("difference between X and Y", "prerequisite of X") additionally pull context from a lightweight JSON knowledge graph (`src/graph_database/kg_retriever.py`).
3. **Generation and metadata-scoped context** (`app.py`) — the chunk metadata schema (`subject`, `category`, `unit`, `file_name`) doubles as a lightweight catalog: retrieval is filtered by it (e.g. a PYQ query is scoped to `category=question_papers` only), the same way metadata tags scope what a query engine is allowed to touch in a real catalog system.

## LLM APIs & prompting techniques

- **API integration:** OpenRouter (`openai/gpt-4o-mini` by default, configurable via env var), called directly through the `openai` Python client — not through a LangChain chain, so retrieval-to-generation orchestration is hand-written.
- **Intent-conditioned system prompts:** three distinct instruction profiles depending on the query route — curriculum queries get an instruction to compile syllabus items exhaustively; PYQ queries get an instruction to answer each extracted exam question sequentially and numbered; subject-content queries get an instruction for structured, cited explanation.
- **Grounding / hallucination suppression:** every profile carries an explicit constraint — answer only from the provided context, and state plainly when the context doesn't contain the answer, rather than filling the gap with general knowledge.
- **Query rewriting before retrieval:** queries are expanded with subject-specific terminology and topic synonyms before they hit the vector store (e.g. "go back n" → "go-back-n, GBN, sliding window, ARQ"), closing the vocabulary gap between how students phrase questions and how source material is written.
- **Graceful degradation:** if the LLM API is unavailable, the system falls back to extractive answering — selecting the highest-overlap sentences directly from retrieved chunks — so a grounded (if less fluent) answer is still possible offline.

## Evaluation approach

Manually inspecting responses doesn't scale and doesn't catch regressions, so this project has two separate, purpose-built evaluation tools rather than relying on spot-checks:

- **`scripts/run_evaluation.py`** runs a labeled query set through the live pipeline and checks route classification, subject detection, and retrieval-method accuracy against expected labels — a regression check for the routing/retrieval logic itself.
- **A three-way comparative benchmark** (in the sibling `comparison-benchmark/` folder) scores this system against a vanilla vector-only RAG baseline and an ungrounded generic LLM, on a hand-curated gold query set spanning normal academic questions, comparison/relationship questions, and deliberately adversarial or off-topic queries (testing refusal and scope behavior, not just correctness). Scoring combines extractive match, retrieval recall against target citations, and LLM-judged faithfulness/relevance into a single weighted score, with the full breakdown (not just a headline number) written out per run.

## Design decisions worth knowing

- **Reranking is a hand-tuned heuristic, not a trained cross-encoder** — a multi-factor score (token overlap, phrase match, topic-pattern bonuses, source-authority weighting) chosen for being fully offline and debuggable at this corpus size; a trained cross-encoder is the natural next step if the corpus grows.
- **The knowledge graph is a flat JSON adjacency list, not a graph database** — an earlier iteration used Neo4j with a NetworkX fallback and a spaCy/NLTK entity-and-relationship extractor; both were replaced by a much simpler heading-heuristic concept graph once it became clear the added infrastructure wasn't earning its complexity for a corpus this size. Similarly, an earlier TF-IDF + Naive Bayes query classifier was replaced by plain keyword routing. All three earlier attempts are kept in `experiments/` for reference, not deleted, since they represent real design exploration.
- **Curriculum and PYQ retrieval deliberately bypass the vector store** — both are structured, discrete data (a syllabus table, a bank of exam questions), and a curated index with keyword scoring is both cheaper and more precise than forcing them through semantic search built for prose.

## Tech stack

- **Retrieval:** ChromaDB (persistent, local), `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim), LangChain (chunking + vectorstore interface)
- **Generation:** OpenRouter → `gpt-4o-mini`
- **Backend:** Flask, REST endpoints (`/api/ask`, `/api/feedback`, `/api/feedback_summary`, `/api/system_status`)
- **Ingestion:** `pypdf`, hand-rolled PPTX/DOCX XML extraction
- **Feedback loop:** 👍/👎 on every answer, logged and aggregated to surface which query types or subjects need retrieval tuning

## Project structure

```
app.py                     # Flask app, routing dispatch, generation
data/
  curriculum/course_structure/   # syllabus PDF for CurriculumExtractor
  subjects/<cn|dbms|os|oops|ds>/<category>/   # textbooks, notes, slides, question_papers, syllabus
  topic_question_index.json      # precomputed PYQ topic index
src/
  curriculum/               # syllabus PDF parser
  document_processing/      # corpus_ingestor.py
  evaluation/                # evaluation_runner.py
  graph_database/            # kg_builder.py + kg_retriever.py
  retrieval/                  # query_router.py, hybrid_retriever.py, reranker.py, question_paper_retriever.py
  vector_database/            # vector_db_manager.py
  web_interface/templates/    # Flask/Jinja2 frontend
scripts/                    # build_vector_store.py, run_evaluation.py, analyze_feedback.py
experiments/                # earlier design iterations, superseded — see experiments/README.md
artifacts/                  # knowledge_graph.json, feedback.jsonl, evaluation runs
```

## Setup

```bash
# from the rag-hybrid/ directory
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini   # optional, this is the default
```

Add source material under `data/subjects/<subject>/<category>/` (category ∈ `textbooks`, `notes`, `slides`, `question_papers`, `syllabus`) and the course-structure syllabus PDF under `data/curriculum/course_structure/`.

Build the index (do this once, and again any time `data/` changes):

```bash
python scripts/build_vector_store.py
```

Run the app:

```bash
python app.py
# -> http://localhost:5001
```

## Known limitations / honest next steps

- The reranker's scoring weights are hand-tuned constants, not learned or grid-searched.
- The knowledge graph's relational edges reflect positional adjacency and shared-unit membership, not verified semantic relationships between concepts.
- Curriculum and PYQ extraction are tightly coupled to this specific syllabus PDF's formatting conventions, not a general-purpose parser.
- Multimodal content (diagrams, circuit figures) isn't indexed — only extractable text.

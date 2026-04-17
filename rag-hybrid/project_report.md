# A Syllabus-Grounded Hybrid Retrieval-Augmented Generation (RAG) Academic Assistant

## TABLE OF CONTENTS
* [List of Figures](#list-of-figures)
* [ABSTRACT](#abstract)
* [CHAPTER 1 INTRODUCTION](#chapter-1-introduction)
  * [1.1 Introduction](#11-introduction)
  * [1.2 Motivation](#12-motivation)
  * [1.3 Problem Definition](#13-problem-definition)
  * [1.4 Report Organisation](#14-report-organisation)
* [CHAPTER 2 LITERATURE SURVEY](#chapter-2-literature-survey)
* [CHAPTER 3 TECHNOLOGIES](#chapter-3-technologies)
  * [3.1 Technologies Used](#31-technologies-used)
  * [3.2 Libraries Used From Python](#32-libraries-used-from-python)
  * [3.3 Frameworks Used](#33-frameworks-used)
  * [3.4 Platforms Used](#34-platforms-used)
* [CHAPTER 4 IMPLEMENTATION](#chapter-4-implementation)
  * [4.1 System Architecture](#41-system-architecture)
  * [4.2 Data Ingestion & Preprocessing](#42-data-ingestion--preprocessing)
  * [4.3 Query Rewriting & Intent Routing](#43-query-rewriting--intent-routing)
  * [4.4 Hybrid Retrieval & Reranking](#44-hybrid-retrieval--reranking)
  * [4.5 Evidence-Supported Answer Generation](#45-evidence-supported-answer-generation)
  * [4.6 Summarisation Component](#46-summarisation-component)
  * [4.7 Recommendation Layer](#47-recommendation-layer)
  * [4.8 User Feedback Mechanism](#48-user-feedback-mechanism)
* [CHAPTER 5 RESULTS & EVALUATION](#chapter-5-results--evaluation)
  * [5.1 Three-Way Comparative Benchmark](#51-three-way-comparative-benchmark)
  * [5.2 Visual Results](#52-visual-results)
* [CHAPTER 6 CONCLUSION & FUTURE WORK](#chapter-6-conclusion--future-work)
  * [6.1 Conclusion](#61-conclusion)
  * [6.2 Future Work](#62-future-work)
* [CHAPTER 7 REFERENCES](#chapter-7-references)

---

## List of Figures
* **Figure 5.1:** Home Interface of the System with Live Feedback Widget
* **Figure 5.2:** Syllabus Query with Cited Evidence Snippets
* **Figure 5.3:** Summarised Study Notes Generated for Previous Year Questions (PYQs)
* **Figure 5.4:** Out-Of-Scope Query Rejection Handling
* **Figure 5.5:** Three-Way Benchmark — Unified Score Comparison (Proposed System vs. Baseline RAG vs. ChatGPT)
* **Figure 5.6:** Three-Way Benchmark — Detailed Metrics Breakdown

---

## ABSTRACT
The rapid growth of AI in education has enabled the development of intelligent systems that support students beyond the classroom. Existing chatbots and assistants often rely on generic web sources, which makes them prone to irrelevant information and hallucinated responses. A retrieval-augmented generation (RAG) based assistant that grounds responses strictly in the official syllabus and provides aligned learning resources can improve reliability, transparency, and trustworthiness in academic contexts.

The proposed system integrates hybrid retrieval, query rewriting, and cross-encoder reranking to accurately identify syllabus passages relevant to student queries. It generates responses with cited evidence snippets from the syllabus and, where applicable, extends coverage through other resources. A summarisation component further condenses retrieved content into concise, study-ready notes, thereby reducing information overload and enhancing accessibility.

Key contributions include a syllabus-aware retrieval framework, an evidence-supported answer generation pipeline, and a recommendation layer that enriches syllabus. This design bridges the gap between course logistics Q&A and resource recommendation, resulting in a syllabus-grounded and student-friendly assistant that strengthens exam preparation and self-study practices.

---

## CHAPTER 1 INTRODUCTION

### 1.1 Introduction
The rapid proliferation of Artificial Intelligence, particularly transformer-based Large Language Models (LLMs), has opened new possibilities for intelligent academic support systems. However, deploying such systems in an educational context introduces a critical challenge: general-purpose LLMs draw on broad parametric knowledge that is rarely aligned with a specific institution's prescribed syllabus. This mismatch results in responses that may be technically plausible but are irrelevant to, or inconsistent with, the official course material — precisely the scenario students rely upon most during exam preparation.

Retrieval-Augmented Generation (RAG) addresses this by constraining generation strictly within an external, curated document corpus. The system described in this report extends this paradigm into a purpose-built, syllabus-grounded academic assistant. Rather than querying open web sources or relying on the model's general knowledge, it grounds every response exclusively in the indexed official syllabus, textbooks, and course materials, providing cited evidence snippets alongside each answer to ensure full transparency.

### 1.2 Motivation
A recurring challenge for engineering students is the gap between the volume of available study material and the ability to quickly locate syllabus-relevant answers. Students must manually search through hundreds of pages of textbooks, cross-reference lecture slides, and sift through previous year question papers — a process that is both time-consuming and prone to information overload.

Existing general-purpose chatbots aggravate this problem by returning responses sourced from the open web, which frequently contain information outside the course scope, introduce hallucinated facts, or lack the structured depth required for academic study. The motivation for this project is to develop a system that eliminates these pitfalls by: (1) restricting retrieval exclusively to official syllabus-aligned materials, (2) rewriting queries to maximise retrieval precision, (3) reranking retrieved passages to surface the most authoritative content, and (4) condensing results into concise study-ready notes through a dedicated summarisation component.

### 1.3 Problem Definition
To design, implement, and deploy a syllabus-grounded academic assistant that: (1) rewrites and classifies student queries to determine retrieval intent; (2) retrieves syllabus passages from a structured vector database using hybrid retrieval; (3) reranks retrieved passages using cross-encoder-style scoring to prioritise authoritative syllabus and textbook content; (4) generates responses with cited evidence snippets to ensure transparency; (5) condenses retrieved material into concise, study-ready notes through a summarisation component; (6) recommends enriching resources through a dedicated recommendation layer; (7) gracefully rejects queries outside the indexed curriculum scope; and (8) continuously improves through a structured user feedback mechanism.

### 1.4 Report Organisation
**Chapter 2** reviews the academic literature on RAG systems, dense embeddings, reranking, summarisation, and educational recommendation systems. **Chapter 3** establishes the complete technological stack. **Chapter 4** details the eight implementation modules: data ingestion, query rewriting and intent routing, hybrid retrieval and reranking, evidence-supported answer generation, summarisation, recommendation layer, and user feedback. **Chapter 5** presents the three-way comparative benchmark with full quantitative metrics. **Chapter 6** concludes the project and proposes future directions. **Chapter 7** enumerates all references.

---

## CHAPTER 2 LITERATURE SURVEY

The development of this system draws on foundational advances across Retrieval-Augmented Generation, semantic embeddings, reranking models, text summarisation, and educational recommendation.

**2.1 Retrieval-Augmented Generation (RAG)**
Lewis et al. (2020) [1] introduced RAG as a mechanism that couples a dense retrieval component with a sequence-to-sequence generative model, demonstrating that grounding generation in a non-parametric external memory significantly improves factual accuracy. The proposed system applies this approach with a domain-specific twist: the retrieval corpus is restricted entirely to official syllabus documents, textbooks, and course materials, ensuring that responses remain curriculum-bounded rather than open-domain.

**2.2 Dense Embedding Representations**
Traditional sparse retrieval models such as TF-IDF suffer from the vocabulary mismatch problem — for example, failing to link the query term "normalise" to the document phrase "BCNF" or "Functional Dependency." Reimers and Gurevych (2019) [2] introduced Sentence-BERT (SBERT), which produces dense, semantically meaningful vector representations using siamese network fine-tuning. The system uses the lightweight `all-MiniLM-L6-v2` variant, producing 384-dimensional embeddings optimised for CPU-based deployment, enabling accurate syllabus passage retrieval even for semantically paraphrased student queries.

**2.3 Cross-Encoder Reranking**
Bi-encoder retrieval (as used in dense vector search) trades reranking precision for speed by encoding queries and documents independently. Cross-encoder models, by contrast, process the query and candidate passage jointly, allowing richer interaction features and more accurate relevance judgements. Nogueira and Cho (2019) [3] demonstrated that cross-encoder-style reranking applied after initial retrieval substantially improves passage ranking precision. The proposed system implements a category-aware multi-factor reranking module that scores each retrieved chunk using query-document token overlap, phrase-level matching, topic-pattern recognition, and source authority weights — emulating the precision of cross-encoder scoring within a CPU-efficient offline framework.

**2.4 Hallucination Reduction in Grounded Generation**
Shuster et al. (2021) [4] documented that constraining generation to retrieved reference contexts significantly reduces hallucination in conversational systems. The proposed system reinforces this at two levels: (a) context-level, through heuristic reranking that surfaces the highest-quality syllabus passages; and (b) prompt-level, through explicit system instructions that forbid the LLM from injecting information beyond the retrieved context.

**2.5 Summarisation for Academic Study**
Automatic text summarisation condenses lengthy retrieved passages into concise, study-ready content. Extractive summarisation selects the most relevant sentences from source documents [5], while abstractive approaches produce novel paraphrased summaries. The system employs both: an extractive fallback that scores and surfaces the highest-overlap sentences from retrieved chunks, and an LLM-based abstractive summariser for richer, structured condensation when an API connection is available.

**2.6 Educational Recommendation Systems**
Recommendation systems in educational contexts extend basic Q&A by surfacing linked topics, prerequisite concepts, and supplementary resources relevant to the student's query. The system implements a recommendation layer that mines the academic knowledge graph and retrieved textbook context to identify related concepts and suggest further reading directions alongside each generated answer.

---

## CHAPTER 3 TECHNOLOGIES

### 3.1 Technologies Used
* **Python (3.10+):** Employed as the primary backend language managing document ingestion, vector operations, query processing, and API orchestration.
* **HTML / CSS / JavaScript:** Forms the frontend user interface, with a clean academic design, live feedback statistics widget, animated analysis modal, and client-side Markdown rendering.

### 3.2 Libraries Used From Python
* **LangChain & LangChain-Core (v0.2.x):** Handles recursive text splitting, prompt engineering, and the orchestration chain linking retrievers to the generative model.
* **ChromaDB:** A persistent, local-first vector database that stores document chunks alongside rich metadata (subject, category, unit, file path), enabling filtered syllabus-aware retrieval.
* **Sentence-Transformers:** Provides the `all-MiniLM-L6-v2` dense embedding model, encoding both documents and queries into a shared 384-dimensional semantic space for nearest-neighbour retrieval.
* **PyPDF:** Parses raw text from academic PDF documents (textbooks, syllabi, question papers, slides) for ingestion into the vector store.
* **markdown2:** Converts LLM-generated structured Markdown responses into safely rendered HTML for display in the web interface.
* **Counter (collections):** Aggregates user feedback counts by query type, retrieval method, and subject within the feedback analysis pipeline to produce automated improvement recommendations.

### 3.3 Frameworks Used
* **Flask:** A lightweight WSGI framework providing the RESTful API layer (`/api/ask`, `/api/feedback`, `/api/feedback_summary`, `/api/system_status`) and serving the Jinja2-rendered frontend template.

### 3.4 Platforms Used
* **OpenRouter APIs:** Routes requests to OpenAI's `gpt-4o-mini` architecture for LLM-based answer generation and abstractive summarisation, allowing external inference without local GPU hardware.

---

## CHAPTER 4 IMPLEMENTATION

The system is composed of eight tightly integrated modules that together realise the syllabus-grounded retrieval, evidence-cited generation, summarisation, and recommendation pipeline described in the abstract.

### 4.1 System Architecture
The end-to-end pipeline executes the following stages per student query:

1. **Query Rewriting & Intent Routing** → rewrites and classifies the query; determines subject and retrieval mode
2. **Hybrid Vector Retrieval** → ChromaDB returns the top-K candidate syllabus passages (14 by default)
3. **Knowledge Graph Lookup** *(for relational queries)* → augments context with concept relationship data
4. **Cross-Encoder-Style Reranking** → re-scores and re-orders candidates by syllabus authority and query relevance
5. **Syllabus Refinement** *(for curriculum queries)* → applies term-overlap re-scoring to surface the most on-topic passages
6. **Evidence-Supported Answer Generation** → constructs a grounded response with cited source snippets
7. **Summarisation** → condenses retrieved content into concise, study-ready notes
8. **Recommendation Layer** → surfaces related topics and resources from the knowledge graph and indexed material
9. **User Feedback Collection** → captures satisfaction rating and logs it for analysis

All modules operate under a strict grounding contract: if the indexed syllabus does not contain relevant information, the system communicates this explicitly rather than fabricating a response.

### 4.2 Data Ingestion & Preprocessing
Academic PDFs — including official syllabi, prescribed textbooks, lecture slides, and previous year question papers — are parsed via PyPDF and split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` (chunk size: **1,000 characters**; overlap: **200 characters**). The overlap window preserves semantic continuity across paragraph boundaries.

Each chunk is encoded into a 384-dimensional vector by `all-MiniLM-L6-v2` and stored in ChromaDB alongside the following metadata fields:

* `subject` (cn / dbms / os / oops / ds)
* `category` (textbooks / slides / notes / question\_papers / syllabus)
* `file_name`, `relative_path`, `unit`

The `category` and `subject` fields are central to the syllabus-aware retrieval framework: they enable the reranker to apply differential authority weights and allow the router to filter retrieval to the correct document group (e.g., restricting a curriculum query to `category=syllabus` documents only).

### 4.3 Query Rewriting & Intent Routing
Before retrieval begins, every student query passes through a two-stage processing step:

**Query Rewriting:** The raw query is expanded with subject-specific terminology and topic synonyms to close the vocabulary gap between natural student phrasing and the precise language of the indexed syllabus. For example, a query about *"go back n"* is rewritten to include *"go-back-n, GBN, sliding window, ARQ"*, ensuring that the vector search surfaces the correct set of syllabus passages regardless of the abbreviation style used. This rewriting step is applied before retrieval and is tailored per detected subject, directly improving the precision of the hybrid retrieval stage.

**Intent Routing:** The `QueryRouter` classifies the rewritten query into one of five execution paths:

| Route Type | Trigger Condition | Retrieval Mode |
|---|---|---|
| `curriculum_query` | Terms: *syllabus, lab, textbooks, semester, units* | Syllabus-only filter |
| `pyq_query` | PYQ trigger terms + supported subject | Question-papers filter |
| `supported_subject_content` | Subject recognised via aliases or topic hints | Hybrid (KG) or Vector |
| `unsupported_subject` | Academic phrasing, no matching subject | Graceful rejection |
| `no_source` / `gibberish` | Off-domain or incoherent input | Hard rejection |

**Automatic Subject Detection:** Subject identification does not rely solely on the presence of a subject name in the query. A curated `TOPIC_SUBJECT_HINTS` dictionary maps specific technical topic terms (e.g., *"deadlock," "paging," "semaphore"* → OS; *"normalization," "BCNF," "functional dependency"* → DBMS; *"go-back-n," "CRC," "OSI model"* → CN) to the correct subject, enabling the router to classify queries that omit the subject name entirely. This is a key enhancement that allows students to ask natural, topic-focused questions without explicitly stating their course.

**Gibberish Detection:** A `_is_gibberish()` guard filters out inputs that are too short, contain no alphabetic characters, or carry an abnormally high ratio of special characters. This prevents the retrieval pipeline from being triggered on invalid inputs and ensures only well-formed academic queries reach the syllabus corpus.

Additional flags raised by the router:
* **`use_kg`:** Activated for relationship-oriented queries (*"difference," "compare," "prerequisite," "linked"*) to enable knowledge graph augmentation alongside standard vector retrieval.
* **`needs_summary`:** Flags summarisation intent (triggered by terms such as *"short notes," "summary," "revision"*) to activate the downstream summarisation component.
* **Topic candidate extraction:** A regex cascade extracts the specific concept (e.g., `"deadlock"` from `"explain deadlock in OS"`) used both in query expansion and in the cited evidence snippet display.

### 4.4 Hybrid Retrieval & Reranking
**Retrieval Query Expansion:** Before querying ChromaDB, the rewritten query string is further enriched with the detected topic candidate and curriculum-specific anchor terms (e.g., *"official syllabus lab textbooks references"* for curriculum queries, or *"previous question important questions exam questions"* for PYQ queries). This expansion ensures retrieval is aligned with the structural language of the indexed documents rather than the conversational phrasing of the student query.

**Hybrid Retrieval:** ChromaDB performs dense nearest-neighbour search in the 384-dimensional embedding space, optionally filtered by `subject` and `category` metadata — for example, restricting a PYQ query entirely to `category=question_papers` documents. For relationship-oriented queries where `use_kg=True`, a `GraphRetriever` module additionally traverses the pre-built academic knowledge graph (`artifacts/knowledge_graph.json`) to extract a neighbourhood of related concept nodes, serialising typed relationships (*prerequisite_for, related_to, appears_in*) into a structured `graph_context` string prepended to the retrieved chunks. This hybrid combination of dense vector search and structured graph traversal ensures that both topical and relational dimensions of the student query are covered.

**PYQ Double-Fetch Pipeline:** For queries classified as `pyq_query`, the system performs a two-phase retrieval rather than a single vector search. In the first phase, it retrieves raw examination question strings from the question-paper corpus. In the second phase, it uses the text of those extracted questions as the retrieval query against the textbook corpus, surfacing the precise conceptual passages that answer each exam question. This double-fetch mechanism ensures that both the relevant previous questions and their textbook-grounded answers are available for the generation stage.

**Cross-Encoder-Style Reranking:** The initial retrieval returns up to 14 candidate passages. These are then re-scored by the `HeuristicReranker` module, which evaluates each candidate against the query using a multi-factor scoring formula that emulates cross-encoder joint query-document interaction:

```
rerank_score = min(1.0,
    token_overlap       # Jaccard token overlap between query and passage
  + phrase_bonus        # +0.20 if the full query phrase appears verbatim
  + topic_bonus         # +0.25 per matched syllabus protocol variant
  + syllabus_bonus      # +0.10 for syllabus-category passages
  + source_bonus        # +0.80 for textbook passages | +0.05 for slides/notes
)
```

The **+0.80 source authority weight** for textbook passages is the defining design decision of the reranking stage. Since official textbooks represent the most authoritative statements of syllabus content, they are systematically elevated above supplementary slide notes, ensuring the evidence snippets cited in responses are drawn from primary, trustworthy sources.

The final combined score applies category-aware blending:
* **Textbooks:** `0.50 × vector_score + 0.50 × rerank_score`
* **Other categories:** `0.65 × vector_score + 0.35 × rerank_score`

**Curriculum Chunk Refinement:** For curriculum-mode queries, an additional post-reranking refinement step re-scores retrieved chunks by term overlap between the query (augmented with the topic candidate) and the chunk content, applying phrase-presence bonuses and lab-context bonuses. This extra stage ensures that curriculum queries targeting a specific subject's syllabus structure (e.g., *"labs under Computer Networks"*) receive strictly on-topic passages rather than tangentially related textbook content.

Together, these retrieval and reranking enhancements form the syllabus-aware retrieval framework described as a key contribution in the abstract.

### 4.5 Evidence-Supported Answer Generation
The top-ranked passages are assembled into a structured context window and submitted to the LLM (`gpt-4o-mini` via OpenRouter) alongside a routing-specific system instruction. Three instruction profiles are applied:

* **Curriculum queries:** The model is directed to compile all available syllabus items completely and without omission, directly serving the course logistics Q&A use case described in the abstract.
* **PYQ queries:** The model generates sequential, numbered answers aligned to each extracted examination question, directly grounded in retrieved textbook evidence.
* **Subject content queries:** The model produces a structured, formally written explanation using only the retrieved context — with bold key terms, numbered steps, and bullet-point features — accompanied by cited source references.

A strict hallucination-suppression instruction is embedded in all profiles:

> *"Answer strictly and exclusively using the provided context. Do NOT use your general knowledge to inject external facts or complete missing details. If the provided context does not contain the necessary information, state that explicitly."*

Each generated response is presented alongside a **Sources** citation card listing the exact syllabus passages and textbook sections that informed the answer, fulfilling the abstract's requirement for cited evidence snippets and ensuring full transparency to the student.

If the OpenRouter API is unavailable, the system falls back to an **extractive summarisation** engine that scores sentences from retrieved passages by token overlap with the query and surfaces the most relevant ones directly — maintaining grounded responses without requiring external API access.

### 4.6 Summarisation Component
A dedicated summarisation component addresses the challenge of information overload identified in the abstract. After the primary answer is generated or extracted, the system condenses the retrieved syllabus content into concise, study-ready notes through two complementary mechanisms:

* **Extractive summarisation:** The `_extractive_answer` engine ranks individual sentences from retrieved passages by query-term overlap and selects the highest-scoring sentences as a concise summary. This operates entirely offline and requires no API call.
* **Abstractive summarisation:** When the LLM generation endpoint is available and `needs_summary` is flagged, the LLM instruction profile is adjusted to produce a structured condensation of the retrieved content — using bullet points and short paragraphs — rather than a long-form explanation. This produces concise, exam-ready notes directly aligned with the syllabus passages retrieved.

The summarisation component is particularly valuable for queries such as *"give me short notes on paging"* or *"revision summary for DBMS normalization"*, where a student needs a quick, structured recap rather than a full elaboration.

### 4.7 Recommendation Layer
The recommendation layer enriches the student's learning journey by surfacing related syllabus topics and resources beyond the immediate query, as described in the abstract's key contributions. It operates through two complementary sub-components:

* **Knowledge graph recommendations:** Following answer generation, the `GraphRetriever` extracts concept neighbours from the academic knowledge graph that are linked to the queried topic. These are presented to the student as suggested further reading (e.g., after answering a query on *deadlock*, the system may recommend exploring *semaphores, mutual exclusion, and the Banker's Algorithm*).
* **Textbook topic discovery:** The `_extract_textbook_topics` module scans the retrieved textbook chunks for heading-level terms adjacent to the queried concept. These are surfaced as related areas the student can explore to deepen their understanding beyond the immediate answer.

Both components feed into a closing recommendation line appended to every response — for example: *"You may also explore Semaphores, Mutual Exclusion, and the Banker's Algorithm for a fuller understanding of this topic."* This recommendation layer bridges the gap between course logistics Q&A and resource recommendation noted in the abstract.

### 4.8 User Feedback Mechanism
The system incorporates a structured user feedback mechanism to enable continuous quality improvement aligned with the evolving needs of students:

**Collection:** After every answered query, the interface renders **👍 Helpful / 👎 Not Helpful** buttons. On click, the client submits a POST request to `/api/feedback` with the query, retrieval method, detected subject, query type, confidence score, and source type. The backend appends this as a JSON record to `artifacts/feedback.jsonl`.

**Analysis:** The `/api/feedback_summary` endpoint and the standalone `scripts/analyze_feedback.py` script aggregate the log to produce: total ratings, helpful/not-helpful counts and ratios, the top 3 query types and retrieval methods associated with not-helpful outcomes, and automated textual recommendations for course-correcting retrieval behaviour.

**Live Dashboard:** A clickable **📊 Feedback** widget in the application header displays running counts and opens an analysis modal with bar charts and recommendation cards, enabling the developer to monitor system quality without manual log inspection.

---

## CHAPTER 5 RESULTS & EVALUATION

### 5.1 Three-Way Comparative Benchmark
To rigorously validate the system's ability to ground responses in the official syllabus and outperform generic alternatives, a three-way comparative evaluation was conducted against:

* **Baseline RAG:** A vanilla ChromaDB vector retrieval system with no query rewriting, no reranking, and a generic generation prompt — representing a standard unoptimised RAG deployment.
* **ChatGPT (gpt-4o-mini via OpenRouter):** A general-purpose LLM with no access to the indexed syllabus corpus — representing the class of existing assistants that rely on generic web-sourced knowledge.

**Evaluation Dataset:** A 50-question synthetic gold dataset (`gold_dataset_50.json`) was constructed spanning all five subjects (CN, DBMS, OS, OOPS, DS) across four query types: concept explanation, comparison, PYQ-style, and curriculum/syllabus queries.

**Metrics Employed:**

| Metric | Description |
|---|---|
| **Extractive Match** | Token-level overlap between the system answer and the gold reference answer |
| **Recall** | Proportion of gold answer key-terms recovered in the system answer |
| **Faithfulness** | LLM-judged score (1–5) for factual consistency with the indexed syllabus material |
| **Relevance** | LLM-judged score (1–5) for topical alignment with the student query |
| **Unified Score** | Weighted combination: `0.35×Extractive + 0.35×Recall + 0.15×Faithfulness_norm + 0.15×Relevance_norm` |

> **Note:** ChatGPT Recall is marked N/A as it has no access to the gold syllabus corpus. Its Faithfulness score (5.00) reflects LLM self-confidence in general-knowledge outputs and is not comparable to syllabus-grounded faithfulness.

**Results:**

| System | Unified Score | Extractive Match | Recall | Faithfulness | Relevance |
|---|---|---|---|---|---|
| **Proposed System** | **80.2%** | **85.5%** | **80.6%** | 4.07 | **3.20** |
| Baseline RAG | 68.4% | 82.4% | 60.0% | 3.73 | 2.07 |
| ChatGPT | 42.0% | 26.7% | N/A | 5.00 | 0.73 |

**Key Findings:**
1. The proposed system outperforms Baseline RAG by **+11.8 percentage points** on the Unified Score, demonstrating that query rewriting and syllabus-authority reranking deliver measurable accuracy gains over unoptimised RAG.
2. The **Recall improvement of +20.6 points** (80.6% vs. 60.0%) confirms that the reranking stage surfaces more complete syllabus content, reducing the risk of students receiving partial or misleading answers.
3. ChatGPT achieves only **26.7% Extractive Match**, confirming that assistants relying on generic web-sourced knowledge cannot reliably reproduce curriculum-specific content expected by students.
4. The proposed system's **Faithfulness score of 4.07/5** validates the hallucination-suppression design: responses are both grounded in the syllabus and rated as factually consistent, directly addressing the reliability and trustworthiness goals stated in the abstract.

### 5.2 Visual Results

**Figure 5.1: Home Interface with Live Feedback Widget**
The application interface showing the system header, the live 📊 Feedback statistics widget (running helpful/not-helpful counts), the clean chat layout with Markdown-rendered answers, and the persistent query input.

**Figure 5.2: Syllabus Query with Cited Evidence Snippets**
A student query answered by the system. Highlights include the Query Analysis panel (retrieval method, confidence score, processing time), the structured syllabus-grounded answer, cited Sources card listing exact syllabus passages, and the 👍/👎 feedback buttons.

**Figure 5.3: Summarised Study Notes for PYQs**
Demonstrates the PYQ pipeline: the system surfaces relevant previous examination questions from the indexed corpus and generates concise, textbook-grounded answers below each question, functioning as a study-ready summary of exam-relevant content.

**Figure 5.4: Out-Of-Scope Query Rejection**
Validates the intent router's scope enforcement. Shows the formal rejection message returned when a student queries a topic outside the indexed curriculum, with an optional override to receive a general LLM response.

**Figure 5.5: Three-Way Benchmark — Unified Score**
Bar chart comparing the Proposed System (80.2%), Baseline RAG (68.4%), and ChatGPT (42.0%) on the Unified Score, demonstrating the system's superiority across all experimental conditions.

**Figure 5.6: Three-Way Benchmark — Detailed Metrics**
Multi-panel breakdown comparing Extractive Match, Recall, Faithfulness, and Relevance across all three systems, produced by the automated `generate_academic_graphs.py` benchmarking script.

---

## CHAPTER 6 CONCLUSION & FUTURE WORK

### 6.1 Conclusion
This project demonstrates that a syllabus-grounded hybrid RAG assistant can significantly improve the reliability, transparency, and trustworthiness of AI-based academic support — the core goals stated in the abstract. By grounding every response strictly in the official indexed syllabus and textbooks through a combination of query rewriting, hybrid retrieval, and cross-encoder-style reranking, the system achieves a Unified Score of **80.2%** — outperforming a standard Baseline RAG by **11.8 percentage points** and a general-purpose LLM (ChatGPT) by **38.2 percentage points** on a 50-question syllabus benchmark.

The summarisation component reduces information overload by condensing retrieved syllabus passages into concise study-ready notes, while the recommendation layer enriches student learning by surfacing linked concepts and related resources beyond the immediate query. Together, these components bridge the gap between course logistics Q&A and resource recommendation. The built-in user feedback mechanism transforms the system into a continuously improving platform, enabling data-driven refinement of retrieval and generation behaviour based on real student satisfaction signals.

### 6.2 Future Work
Several enhancements are planned to further strengthen the system's syllabus-grounded capabilities:

1. **True Cross-Encoder Reranking:** Replacing the current heuristic scoring with a learned cross-encoder model (such as `ms-marco-MiniLM-L-6-v2`) would provide jointly trained query-document relevance judgements, further improving the precision of syllabus passage selection beyond what rule-based weighting achieves.
2. **Adaptive Feedback-Driven Retrieval:** The feedback JSONL log can be used to automatically adjust source authority weights per subject — for example, increasing the textbook priority multiplier for subjects that consistently receive not-helpful ratings, creating a self-calibrating retrieval pipeline.
3. **Automated Knowledge Graph Construction:** Future iterations will auto-generate the concept relationship graph from the indexed syllabus corpus using Named Entity Recognition and dependency parsing, removing the need for manual curation and enabling the recommendation layer to scale across additional subjects and institutions.
4. **Multimodal Syllabus Understanding:** Expanding ingestion to include figures, tables, and diagrams from textbooks via vision-capable models would allow the system to answer queries about circuit diagrams, data structure illustrations, and algorithm flowcharts — types of syllabus content currently beyond the system's scope.
5. **Personalised Study Path Recommendation:** Extending the recommendation layer with student interaction history would enable the system to suggest personalised revision sequences, identifying gaps in a student's query history relative to the full syllabus coverage.

---

## CHAPTER 7 REFERENCES

[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, *et al.*, "Retrieval-augmented generation for knowledge-intensive nlp tasks," in *Advances in Neural Information Processing Systems* (NeurIPS), vol. 33, pp. 9459–9474, 2020. \
[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using siamese BERT-networks," in *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (EMNLP-IJCNLP), pp. 3982–3992, 2019. \
[3] R. Nogueira and K. Cho, "Passage re-ranking with BERT," *arXiv preprint arXiv:1901.04085*, 2019. \
[4] K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston, "Retrieval augmentation reduces hallucination in conversation," in *Findings of the Association for Computational Linguistics: EMNLP 2021*, pp. 3784–3803, 2021. \
[5] J. Goldstein, V. Mittal, J. Carbonell, and M. Kantrowitz, "Multi-document summarization by sentence extraction," in *NAACL-ANLP 2000 Workshop on Automatic Summarization*, pp. 40–48, 2000. \
[6] LangChain Documentation, "LangChain Core Abstractions," available online: *https://python.langchain.com/docs/core/* \
[7] Chroma Documentation, "Chroma: The Open Source AI-native open-source embedding database," available online: *https://docs.trychroma.com/* \
[8] OpenRouter API Official Specifications, available online: *https://openrouter.ai/docs*

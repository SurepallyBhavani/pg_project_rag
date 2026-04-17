# EduAssist: A Domain-Specific Hybrid Retrieval-Augmented Generation (RAG) Academic Assistant

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
  * [4.2 Data Ingestion & Preprocessing](#42-data-ingestion-and-preprocessing)
  * [4.3 Query Routing Engine](#43-query-routing-engine)
  * [4.4 Hybrid Retrieval & Heuristic Reranking](#44-hybrid-retrieval-and-heuristic-reranking)
  * [4.5 Grounded LLM Generation](#45-grounded-llm-generation)
* [CHAPTER 5 RESULTS](#chapter-5-results)
* [CHAPTER 6 CONCLUSION & FUTURE WORK](#chapter-6-conclusion--future-work)
  * [6.1 Conclusion](#61-conclusion)
  * [6.2 Future Work](#62-future-work)
* [CHAPTER 7 REFERENCES](#chapter-7-references)

---

## List of Figures
* **Figure 5.1:** Home Interface of EduAssist
* **Figure 5.2:** Execution of a Concept Query with Semantic Citations 
* **Figure 5.3:** Automated Solutions Generated for Previous Year Questions (PYQs)
* **Figure 5.4:** Out-Of-Scope Query Rejection Handling

---

## ABSTRACT
In modern academic environments, higher-education engineering students face an overwhelming influx of unstructured textual data distributed across textbooks, faculty presentation slides, legacy notes, and vast repositories of Previous Year Questions (PYQs). As general-purpose Large Language Models (LLMs) frequently hallucinate when presented with domain-specific or curriculum-bounded constraints, they cannot be reliably used for academic study independently without grounded mechanisms. 

To bridge this gap, this project introduces "EduAssist," a robust, hybrid Retrieval-Augmented Generation (RAG) assistant dedicated exclusively to computer science curricula (spanning core subjects such as Database Management Systems, Operating Systems, Computer Networks, Object-Oriented Programming, and Data Structures). By locally indexing approved academic textbooks and curriculum syllabi within a high-dimensional vector space (ChromaDB) and applying custom heuristic reranking algorithms, EduAssist retrieves mathematically aligned context before synthesizing answers via state-of-the-art LLMs. The result is a domain-restricted, hallucination-resistant academic assistant capable of resolving targeted previous questions dynamically, thereby significantly improving a student's learning efficiency.

---

## CHAPTER 1 INTRODUCTION

### 1.1 Introduction
The advent of Artificial Intelligence, specifically the deployment of transformer-based Large Language Models (LLMs) such as OpenAI's GPT-4 variants, has fundamentally restructured how humans retrieve and process information. Unlike traditional Information Retrieval (IR) models that depend on exact lexical matching (such as BM25), modern artificial intelligence handles natural language intrinsically to generate human-like contextual responses. However, while open-domain LLMs possess vast internal parametric memory regarding general world occurrences, their responses concerning specifically scoped academic curricula (such as exact definitions from prescribed engineering textbooks or tailored university syllabus configurations) are often ungrounded, generalized, or entirely hallucinated.

Retrieval-Augmented Generation (RAG) mitigates this limitation by strictly constraining the generative LLM within an external "non-parametric" database of trusted documents. EduAssist utilizes this paradigm to deploy a customized study tool for engineers.

### 1.2 Motivation
A significant challenge faced by engineering undergraduate and postgraduate students is the fragmentation of knowledge. Students are routinely required to triangulate concepts across lengthy textbooks containing thousands of pages, abbreviated class notes, and raw Previous Year Question (PYQ) papers to determine the depth and context necessary for examinations. 

Currently, students perform manual keyword searches across PDF documents using rudimentary text-matching tools, which utterly fail to capture semantic relevance. The primary motivation for this project is to leverage advanced Natural Language Processing (NLP) embedding models to comprehend the semantic meaning of student queries and map them programmatically to the definitive answers located deep within the prescribed computer science literature.

### 1.3 Problem Definition
To design, implement, and deploy an artificially intelligent academic assistant that evaluates student inputs, automatically determines the engineering subject context, queries a multidimensional vector database arrayed with verified academic materials, and leverages an LLM to generate long-form, highly detailed, text-book grounded explanations while intentionally rejecting inquiries outside of the localized domain constraint. 

### 1.4 Report Organisation
The remainder of this report is structured as follows. **Chapter 2** reviews the underlying academic literature encompassing RAG systems and semantic embeddings. **Chapter 3** establishes the specific technological stack and libraries leveraged during development. **Chapter 4** delves deeply into the underlying mathematical and programmatic implementation of the routing logic, vector indexing, and generative capabilities. **Chapter 5** presents structural placeholders and explanations for visual results. **Chapter 6** concludes the project scope and proposes future iterative enhancements. Ultimately, **Chapter 7** enumerates references and citations supporting the systemic methodology.

---

## CHAPTER 2 LITERATURE SURVEY

The development of the EduAssist platform relies heavily on foundational breakthroughs across multiple disciplines of Natural Language Processing and Machine Learning. 

**2.1 Retrieval-Augmented Generation (RAG)**
In their seminal 2020 paper, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* [1], Lewis et al. introduced RAG as a mechanism that integrates pre-trained sequence-to-sequence memory models with dense retrieval mechanisms. Their research demonstrated that coupling non-parametric external memory (a localized vector database) with parametric memory (the LLM's internal neural weights) radically enhances factual accuracy in long-form generation tasks. EduAssist applies this methodology to constraint generation specifically to undergraduate text structures.

**2.2 Dense Embedding Representations**
Traditional sparse retrieval models (such as TF-IDF) operate based on term frequency methodologies and suffer critically from the vocabulary mismatch problem (e.g., failing to link the query term "normalize" to the text string "BCNF" or "Functional Dependency"). Reimers and Gurevych (2019) introduced *Sentence-BERT* (SBERT) [2], showcasing that fine-tuning siamese and triplet network structures allows for the generation of semantically meaningful sentence embeddings. Thus, texts representing physically different string characters but identical conceptual meanings can exist near one another in a mathematical vector space. EduAssist fundamentally relies on the lightweight variation `all-MiniLM-L6-v2` resulting from this lineage.

**2.3 Evaluating RAG Factual Groundedness**
A critical concern in deploying conversational systems in academic environments is ungrounded hallucination, where a model fabricates a plausible but incorrect academic answer. Shuster et al. [3] documented mechanisms through which generation algorithms could be strictly bounded by reference contexts. Subsequent literature emphasizes that context length and algorithmic focus are paramount. As observed in modern applications, providing an LLM with excessively noisy data causes it to "lose" the answer in vast contexts. Therefore, EduAssist implements custom chunking and heuristic reranking (scoring verified textbooks higher than unverified slide sets) to maintain high-signal context streams for the generative endpoints.

---

## CHAPTER 3 TECHNOLOGIES

To facilitate advanced vector search and real-time inference within the EduAssist platform, a highly specialized, modern Python ecosystem is employed.

### 3.1 Technologies Used
* **Python (3.10+):** Employed globally as the primary backend logic layer to manage asynchronous I/O, OS integrations, and large data abstractions.
* **HTML/CSS/JavaScript:** Forms the frontend graphical user interface. The UI employs clean, formal vanilla styling that emphasizes clear academic readability (e.g., using subtle pastel metrics, deep contrasts, and structured markdown parsing) avoiding reliance on heavy frontend frameworks.

### 3.2 Libraries Used From Python
* **LangChain & LangChain-Core (v0.2.x):** The premier orchestration abstraction layer allowing complex sequential chains. It fundamentally handles recursive text splitting, prompt engineering paradigms, and the underlying linkage connecting the retrievers to the final generative chat models.
* **ChromaDB:** Serving as the persistent, local-first vector database. It utilizes an embedded SQLite engine to maintain relationships between document UUIDs, metadata properties, and multi-dimensional floats, executing nearest-neighbor algorithms rapidly.
* **Sentence-Transformers:** Developed by HuggingFace, this library operates the `all-MiniLM-L6-v2` dense embedding model, providing robust 384-dimensional representational vectors perfectly optimized for standard CPU execution.
* **PyPDF:** A robust document parsing engine leveraged explicitly to extract raw text variables sequentially from highly complex academic textbooks and slide structures.

### 3.3 Frameworks Used
* **Flask:** A lightweight Web Server Gateway Interface (WSGI) framework selected to build RESTful API structures interfacing between the asynchronous generative tasks and the user interface. 

### 3.4 Platforms Used
* **OpenRouter APIs:** To minimize hardware resource allocation dependency, the system utilizes OpenRouter endpoints to interface externally with OpenAI's `gpt-4o-mini` architecture for rapid processing. 

---

## CHAPTER 4 IMPLEMENTATION

The crux of the EduAssist functionality stems from five deeply connected sequential implementation modules responsible for digesting raw textbooks and converting them to mathematically queried answers.

### 4.1 System Architecture & Document Ingestion
The process begins with offline data processing. Academic PDFs are loaded into the operational stream via PyPDF arrays. Because LLMs inherently harbor strict token context limits, massive 500-page textbooks cannot be arbitrarily forwarded. EduAssist utilizes LangChain's `RecursiveCharacterTextSplitter`. Text is chunked with a constraint size of 1000 characters coupled with a 200-character overlap window. This overlap ensures semantic dependencies (a sentence wrapping over a paragraph) remain fully intact. The chunks are converted into 384-dimensional vectors by local lightweight infrastructure and lodged securely inside ChromaDB alongside origin metadata linking the vector back to the source file and page index.

### 4.2 Query Routing Engine
When an end-user queries the system, the text string is intercepted by the `QueryRouter`. The router implements advanced lexical intent algorithms.
1. It analyzes against a massive internally curated `TOPIC_SUBJECT_HINTS` dictionary to organically derive the underlying subject—effortlessly tying the query `"What is a deadlock and semaphore"` to the Operating Systems (OS) domain, negating the necessity for the user to select the subject explicitly.
2. It detects PYQ triggers. If terms like *"previous layout"*, *"questions"*, or *"exam format"* are parsed, the router categorizes the execution pipeline as a specific `pyq_query`.
3. It detects out-of-scope intent; resolving queries associated with geography or sports completely via immediate fallback rejections unless overridden.

### 4.3 Hybrid Retrieval and Heuristic Reranking
Simple vector proximity mathematically determines similar terms but lacks authoritative context logic. Therefore, a custom `HeuristicReranker` algorithm was heavily designed and implemented.
Once Chroma retrieves 14 loosely matching topological vector nodes, `HeuristicReranker` intervenes:
1. It evaluates simple token intersections to define lexical overlap.
2. It fundamentally enforces an artificial authoritative multiplier (`+0.20` weight bonus) specifically to chunks tagged under the category "textbook".
3. A sub-module explicitly balances the final payload list, forcefully prioritizing up to 3 textbooks before yielding to generic notes or slides, ensuring context depth. 

### 4.4 Resolving Previous Year Questions (PYQs) Deep Integrations
A distinct hurdle in the process involved addressing requests like *"Solve the previous questions on normalization."*
The implementation pipeline circumvents this by instituting double-RAG fetching paradigms:
1. It queries the vector base purely for the raw examination question strings related to the academic topic.
2. It dynamically intercepts those extracted questions, concatenates them locally, and uses the text of the actual examination questions to perform a *secondary* vector query specifically against the textbook database. 
Thus, the system reliably surfaces the precise conceptual nodes detailing the answer to the exact examination questions retrieved in phase one.

### 4.5 Grounded LLM Generation
Finally, the curated chunks, semantic vectors, and routed instructions aggregate into a tightly engineered LLM Instruction Prompt. The model is specifically constrained against brief summations and ordered to produce exhaustive, robust explanations. System safeguards instruct the LLM: *"Do NOT give short summaries; write extensive explanations, exploring all mechanisms, step-by-step processes, and features documented exclusively in the provided context."* This guarantees deep, high-fidelity academic output completely divorced from standard generalized AI platitudes.

---

## CHAPTER 5 RESULTS

*(Note: Appropriate visual UI screenshots highlighting the described behaviours will populate the following sections as detailed below).*

**Figure 5.1: The Home Interface**
A visual representation of the application interface highlighting the clean aesthetic presentation, user status bars, persistent input arrays, and responsive design integrations.

**Figure 5.2: Concept Query Execution**
This sub-section showcases the EduAssist bot answering a detailed topical engineering question. The visual highlights the analytical context pills (e.g., 'VECTOR SEARCH'), processing time latency values, confidence generation markers, and the meticulously drafted multi-paragraph conceptual explanation generated directly from textbooks.

**Figure 5.3: Automated Solutions Generated for PYQs**
Demonstrates the distinct pipeline logic implemented in Section 4.4, with the final visual representation correctly isolating exam-specific previous questions and successfully matching/generating comprehensive textbook answers below them sequentially.

**Figure 5.4: Out-Of-Scope Query Rejection Handling**
A visual demonstration validating the intent router logic. It highlights the formal error boundary UI response deployed when a user attempts to retrieve information regarding unstructured extra-curricular data isolated outside the approved curriculum boundary rules.

---

## CHAPTER 6 CONCLUSION & FUTURE WORK

### 6.1 Conclusion
The EduAssist project successfully demonstrates the vast potential of binding generalized AI reasoning engines with highly specific, localized mathematical vector constraints. By strictly controlling the context window and heuristically elevating authoritative textbook texts over conversational data, the architecture significantly mitigates catastrophic hallucination—creating an examination preparation tool that undergraduate and postgraduate students can trust explicitly. The inclusion of complex query routing intent and automated PYQ resolution mechanisms proves that RAG foundations can be radically evolved to accomplish highly complex multi-stage academic processes efficiently.

### 6.2 Future Work
While the core architecture maintains exceedingly robust retrieval accuracy, the system can be scaled comprehensively. 
1. **Multimodal RAG Pipelines:** Current capabilities are limited natively to unstructured text representations. An expansion towards `LLaVA` or vision-centric transformer blocks would allow the system to store, retrieve, and interpret architectural textbook diagrams, flowcharts, and empirical data tables.
2. **Automated Knowledge Graph Generation:** Advanced mapping to auto-generate intricate relation nodes dynamically across entire multi-college curriculums to construct an organic linked-academic universe without explicit mapping.
3. **Continuous Fine-tuning Validation:** Enacting cross-encoder natural language scoring checkpoints during retrieval processes to mathematically validate outputs on scale to boost inherent confidence intervals objectively.

---

## CHAPTER 7 REFERENCES

[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, *et al.*, "Retrieval-augmented generation for knowledge-intensive nlp tasks," in *Advances in Neural Information Processing Systems* (NeurIPS), vol. 33, pp. 9459-9474, 2020. \
[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using siamese BERT-networks," in *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (EMNLP-IJCNLP), pp. 3982-3992, 2019. \
[3] K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston, "Retrieval augmentation reduces hallucination in conversation," in *Findings of the Association for Computational Linguistics: EMNLP 2021*, pp. 3784-3803, 2021. \
[4] LangChain Documentation, "LangChain Core Abstractions," available online: *https://python.langchain.com/docs/core/* \
[5] Chroma Documentation, "Chroma: The Open Source AI-native open-source embedding database," available online: *https://docs.trychroma.com/* \
[6] OpenRouter API Official Specifications, available online: *https://openrouter.ai/docs*

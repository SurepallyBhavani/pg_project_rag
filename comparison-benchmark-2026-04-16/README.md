# EduAssist Comparison Benchmark

This benchmark is for comparing:

1. General LLM baseline (`ChatGPT`, `NotebookLM`, or both)
2. Standard vector-RAG baseline
3. Enhanced `EduAssist`

## Files

- `benchmark_queries_50.json`
  Final 50-query benchmark grouped by category.
- `external_outputs_template.csv`
  Template for collecting outputs from external systems.

## Suggested External Prompting Rule

Use the same query text exactly as written. Do not modify wording between systems.

For external systems:
- ask the query exactly as given
- save the final answer text
- do not manually improve or edit the answer
- record the model name and date separately if needed

## Recommended Comparison Protocol

Use two external baselines if possible:

1. `ChatGPT` as a general LLM baseline
2. `NotebookLM` as a document-grounded external baseline

### ChatGPT Baseline

Do **not** upload documents.

Use one fixed instruction at the start of the session:

`You are being evaluated against a syllabus-grounded academic assistant. Answer each query clearly and formally. If you are unsure, answer with your best general knowledge. Do not ask follow-up questions.`

Then ask each query exactly as written in `benchmark_queries_50.json`.

This gives a fair `general LLM` comparison.

### NotebookLM Baseline

Upload the same source family used by EduAssist:

- `syllabus_cse.pdf`
- notes for `DS`, `OOPS`, `OS`, `DBMS`, `CN`
- slides for `DS`, `OOPS`, `OS`, `DBMS`, `CN`
- textbooks for `DS`, `OOPS`, `OS`, `DBMS`, `CN`
- question papers for `DS`, `OOPS`, `OS`, `DBMS`, `CN`

Do **not** upload:

- `ocr_pending`
- generated reports
- evaluation artifacts

Use one fixed instruction:

`Answer only from the uploaded academic sources. If the answer is not supported by the sources, say that the source set does not provide enough information. Do not browse outside the uploaded files. Do not ask follow-up questions.`

Then ask each query exactly as written.

## What Data To Feed

### For ChatGPT

Feed only:
- the fixed instruction above
- each query text

Do **not** feed the corpus itself.

### For NotebookLM

Feed the corpus files themselves, not summaries:
- official curriculum PDF
- 5 supported subject notes / slides / textbooks / question papers

This makes NotebookLM the closest external document-grounded comparator.

## Query Groups

- `10` curriculum queries
- `15` subject explanation queries
- `10` graph / relationship queries
- `10` PYQ queries
- `5` unsupported / no-source queries

## External Output Collection

For each query, fill:
- `chatgpt_output`
- `notebooklm_output`
- optionally `notes`

You can leave one system blank if you only compare against one external baseline first.

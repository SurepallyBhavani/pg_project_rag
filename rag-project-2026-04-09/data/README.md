# EduAssist Corpus Layout

This project keeps the original app structure intact and only improves how the academic corpus is organized.

## Directory Layout

`curriculum/course_structure/`
- Shared official curriculum PDF used for course-structure and semester-mapping queries.

`subjects/<subject>/syllabus/`
- Subject syllabus source. For now this reuses the shared `syllabus_cse.pdf` because the per-subject syllabus is contained inside that official document.

`subjects/<subject>/notes/`
- Text-heavy notes, PDFs, DOCX files, and small text resources ready for ingestion.

`subjects/<subject>/slides/`
- Converted `.pptx` lecture slides ready for ingestion.

`subjects/<subject>/textbooks/`
- Textbooks or book chapters that already have usable text extraction.

`subjects/<subject>/programs/`
- Program sheets, records, and lab-style code notes.

`subjects/<subject>/assignments/`
- Assignment-style material.

`subjects/<subject>/question_papers/`
- Previous question papers used for exam-style query support, topic-wise PYQ retrieval, and question framing analysis.

`subjects/<subject>/ocr_pending/`
- Files awaiting full OCR cleanup. The current ingestion pipeline will still include them if readable text can already be extracted, so they can contribute as data sources immediately.

`subjects/<subject>/ocr_processed/`
- OCR-cleaned versions of files from `ocr_pending/`. These are safe to ingest and are preferred over the raw scanned copies.

`subjects/<subject>/misc/`
- Extra files that do not yet fit one of the main categories.

## Supported Subjects In This Prototype

- `cn` - Computer Networks
- `dbms` - Database Management System
- `ds` - Data Structures
- `oops` - Object Oriented Programming
- `os` - Operating System

## Ingestion Policy

First-pass ingestion should include:
- `curriculum/course_structure`
- every subject's `syllabus`
- every subject's `notes`
- every subject's `slides`
- every subject's `textbooks`
- every subject's `programs`
- every subject's `assignments`
- every subject's `question_papers`

`ocr_pending` is now included automatically when text extraction is usable.
If both `ocr_pending` and `ocr_processed` exist for the same source, the `ocr_processed` version is preferred.
You may ingest `ocr_processed` immediately after the cleaned files are added.

## Notes

- The current dataset is intentionally text-first. Diagram understanding can be added later as a multimodal enhancement.
- The original project outside this folder remains unchanged.
- Recommended OCR workflow:
  - keep the raw scanned file in `ocr_pending/`
  - save the OCR-cleaned copy under the matching subject in `ocr_processed/`
  - if useful, preserve category folders inside `ocr_processed/`, such as `ocr_processed/textbooks/` or `ocr_processed/notes/`
  - or use `python scripts/import_ocr_outputs.py <folder_with_cleaned_files> --dry-run` to preview matches, then rerun without `--dry-run` to copy them automatically
  - rerun `python scripts/build_vector_store.py` after adding OCR-cleaned files
- Use `python scripts/ocr_audit.py` to see which files are still pending OCR and where the cleaned copies should be placed.

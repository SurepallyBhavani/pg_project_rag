"""Scan syllabus PDF for ALL pages matching subject keywords to find the right course."""
import sys
sys.path.insert(0, '.')
from scripts.build_topic_index import _pdf_pages, _norm, SUBJECT_COURSE_KEYWORDS
from pathlib import Path

base = Path('.')
subjects_to_check = ['ds', 'oops']  # The problematic subjects

for subj in subjects_to_check:
    pdf_path = base / 'data' / 'subjects' / subj / 'syllabus' / 'syllabus_cse.pdf'
    pages = _pdf_pages(pdf_path)
    kws = SUBJECT_COURSE_KEYWORDS[subj]
    print(f"\n{'='*60}")
    print(f"Subject: {subj.upper()}, keywords: {kws}")
    print(f"Total pages: {len(pages)}")
    
    for idx, page in enumerate(pages):
        low = _norm(page)
        has_struct = 'objectives' in low or 'prerequisites' in low or 'outcomes' in low
        has_unit = 'unit' in low
        for kw in kws:
            if kw in low:
                marker = "** HAS STRUCT **" if has_struct else ""
                unit_marker = "HAS_UNIT" if has_unit else ""
                print(f"  Page {idx}: [{kw}] {marker} {unit_marker}")
                # Print first 300 chars of the page
                print(f"    {page[:300].strip()}")
                print()
                break

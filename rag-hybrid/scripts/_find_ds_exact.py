"""Find EXACT course titles for DS and OOPS in the syllabus."""
import sys
sys.path.insert(0, '.')
from scripts.build_topic_index import _pdf_pages, _norm
from pathlib import Path

pages = _pdf_pages(Path('data/subjects/ds/syllabus/syllabus_cse.pdf'))

# Look for pages that have Data Structures in their title area (first 200 chars)
ds_keywords = ['data structure', 'data structures through c', 'data structures using c', 
               'data structures lab', 'data structures and algorithms']
for idx, page in enumerate(pages):
    first_lines = _norm(page[:400])
    for kw in ds_keywords:
        if kw in first_lines:
            has_unit = 'unit' in _norm(page)
            has_struct = any(s in _norm(page) for s in ['objectives', 'prerequisites', 'outcomes'])
            print(f"Page {idx}: [{kw}] struct={has_struct} unit={has_unit}")
            print(f"  First ~200 chars: {page[:200].strip()}")
            print()

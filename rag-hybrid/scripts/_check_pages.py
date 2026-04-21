"""Check content on specific pages."""
import sys
sys.path.insert(0, '.')
from scripts.build_topic_index import _pdf_pages, _norm
from pathlib import Path

pages = _pdf_pages(Path('data/subjects/ds/syllabus/syllabus_cse.pdf'))

for idx in [9, 35, 36]:
    print(f"\n{'='*60}")
    print(f"PAGE {idx}:")
    print(pages[idx][:800])

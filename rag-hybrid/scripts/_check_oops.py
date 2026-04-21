"""Check OOPS pages 41-43 in syllabus."""
import sys
sys.path.insert(0, '.')
from scripts.build_topic_index import _pdf_pages
from pathlib import Path

pages = _pdf_pages(Path('data/subjects/oops/syllabus/syllabus_cse.pdf'))

for idx in [41, 42, 43]:
    print(f"\n{'='*60}")
    print(f"PAGE {idx}:")
    print(pages[idx][:1000])

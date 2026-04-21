import sys
sys.path.insert(0, '.')
from scripts.build_topic_index import _pdf_pages, _norm
from pathlib import Path
base = Path('.')
pages = _pdf_pages(base / 'data' / 'subjects' / 'ds' / 'syllabus' / 'syllabus_cse.pdf')
kws = ['data structures', 'data structure', 'linked list', 'sorting', 'trees', 'stack and queue', 'stacks']
for idx, page in enumerate(pages):
    low = _norm(page)
    has_struct = 'objectives' in low or 'prerequisites' in low or 'outcomes' in low
    for kw in kws:
        if kw in low and has_struct:
            print(f'Page {idx}: matched [{kw}]')
            print(page[:400])
            print('---')
            break

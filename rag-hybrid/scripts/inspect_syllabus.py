"""Inspect what unit topics the CurriculumExtractor can parse from each subject's syllabus PDF."""
from __future__ import annotations
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.curriculum.curriculum_extractor import CurriculumExtractor

BASE_DIR = Path(__file__).resolve().parent.parent
SUBJECTS = ["cn", "dbms", "ds", "oops", "os"]

for subject in SUBJECTS:
    syllabus_pdf = BASE_DIR / "data" / "subjects" / subject / "syllabus" / "syllabus_cse.pdf"
    if not syllabus_pdf.exists():
        print(f"\n[{subject.upper()}] No syllabus PDF found at {syllabus_pdf}")
        continue

    ext = CurriculumExtractor(str(syllabus_pdf))
    print(f"\n=== {subject.upper()} ===")
    for page in ext.course_pages:
        title = page["title"]
        text  = str(page["text"])
        # Extract unit blocks
        unit_matches = list(re.finditer(r"UNIT\s*-\s*[IVX]+", text, re.IGNORECASE))
        if not unit_matches:
            continue
        print(f"  Course: {title}")
        for i, m in enumerate(unit_matches):
            start = m.start()
            end   = unit_matches[i+1].start() if i+1 < len(unit_matches) else min(start+600, len(text))
            block = re.sub(r"\s+", " ", text[start:end]).strip()[:300]
            print(f"    {block}")
        print()

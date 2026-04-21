"""Quick diagnostic: show unit titles + question counts + unmatched questions."""
import json
from pathlib import Path

idx = json.load(open(Path(__file__).parent.parent / "data" / "topic_question_index.json", encoding="utf-8"))

for subj, data in idx.items():
    print(f"\n{'='*60}")
    print(f"Subject: {subj.upper()}")
    ut = data["unit_topics"]
    for u, udata in ut.items():
        print(f"  {u}: {udata['title'][:80]}  ({len(udata['questions'])} Qs)")
    um = data.get("unmatched", [])
    if um:
        print(f"  --- UNMATCHED ({len(um)}) ---")
        for q in um[:5]:
            print(f"    - {q['question'][:100]}")
        if len(um) > 5:
            print(f"    ... and {len(um)-5} more")

"""
build_topic_index.py
====================
Parse each subject's syllabus PDF to extract the 5 UNIT topics, then map
every question from the question papers to the closest topic(s). 

The key fix (v2): when locating the correct course inside the large combined
syllabus PDF, we now verify the **course title** appears in the first ~300
characters of each candidate page.  This prevents "data structures" (a generic
phrase) picking up "Programming for Problem Solving" (page 9) instead of the
actual "DATA STRUCTURES" course (page 35).

Writes:  data/topic_question_index.json

Run:  python scripts/build_topic_index.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Config: subject → phrases that uniquely ID the course title in the PDF
# ─────────────────────────────────────────────────────────────────────────────
SUBJECT_COURSE_KEYWORDS: Dict[str, List[str]] = {
    "cn":   ["computer networks", "data communications and networks"],
    "dbms": ["database management systems", "database management system"],
    "ds":   ["data structures", "data structure through c"],
    "oops": ["object oriented programming", "programming through c++",
             "oops through java", "object-oriented"],
    "os":   ["operating systems", "operating system"],
}

# Exact course titles as they appear in syllabus PDF headers.
# Used to disambiguate pages that merely *mention* a keyword vs. pages
# that are actually the course header page.
SUBJECT_COURSE_TITLES: Dict[str, List[str]] = {
    "cn":   ["computer networks"],
    "dbms": ["database management systems", "database management system"],
    "ds":   ["data structures"],
    "oops": ["object oriented programming"],
    "os":   ["operating systems", "operating system"],
}

# Fallback topic maps used when PDF-based extraction fails or returns
# incorrect results for a subject.  Keywords are carefully chosen to
# match what typically appears in JNTUH exam questions.
FALLBACK_TOPIC_MAPS: Dict[str, List[Dict]] = {
    "ds": [
        {"unit": "UNIT-I",  "title": "Introduction, Arrays and Linked Lists",
         "keywords": ["array", "linked list", "singly linked list", "doubly linked list",
                      "circular linked list", "introduction", "linear", "sequential",
                      "time complexity", "space complexity", "algorithms", "asymptotic",
                      "big oh", "notation", "abstract data type", "adt", "polynomial",
                      "sparse matrix", "representation"]},
        {"unit": "UNIT-II", "title": "Stacks and Queues",
         "keywords": ["stack", "queue", "deque", "priority queue", "circular queue",
                      "push", "pop", "enqueue", "dequeue", "infix", "postfix",
                      "prefix", "expression evaluation", "recursion", "tower of hanoi",
                      "balancing symbol", "conversion"]},
        {"unit": "UNIT-III","title": "Trees and Binary Trees",
         "keywords": ["tree", "binary tree", "bst", "binary search tree", "avl tree",
                      "b-tree", "b+ tree", "heap", "max heap", "min heap", "traversal",
                      "inorder", "preorder", "postorder", "threaded tree",
                      "expression tree", "red-black tree", "red black", "splay tree",
                      "tries", "trie", "skip list", "huffman"]},
        {"unit": "UNIT-IV", "title": "Graphs",
         "keywords": ["graph", "bfs", "dfs", "breadth first", "depth first",
                      "spanning tree", "shortest path", "dijkstra", "topological",
                      "adjacency", "directed", "undirected", "weighted", "kruskal",
                      "prim", "minimum spanning", "connected component", "biconnected",
                      "strongly connected"]},
        {"unit": "UNIT-V",  "title": "Searching, Sorting and Hashing",
         "keywords": ["sorting", "searching", "bubble sort", "insertion sort",
                      "selection sort", "merge sort", "quick sort", "heap sort",
                      "radix sort", "shell sort", "linear search", "binary search",
                      "hashing", "hash table", "hash function", "collision",
                      "open addressing", "chaining", "rehashing", "extendible hashing",
                      "linear probing", "quadratic probing"]},
    ],
    "oops": [
        {"unit": "UNIT-I",  "title": "Object Oriented Thinking and Basics of C++",
         "keywords": ["class", "object", "oop", "object oriented", "encapsulation",
                      "data hiding", "abstraction", "access specifier", "paradigm",
                      "structure", "constructor", "destructor", "default constructor",
                      "copy constructor", "parameterized constructor", "this pointer",
                      "inline function", "friend function", "static member",
                      "function overloading", "data member", "member function"]},
        {"unit": "UNIT-II", "title": "Classes and Data Abstraction",
         "keywords": ["class", "data abstraction", "access specifier", "public",
                      "private", "protected", "const member", "static data",
                      "static function", "proxy class", "nested class",
                      "scope resolution", "constructor", "destructor"]},
        {"unit": "UNIT-III","title": "Inheritance and Polymorphism",
         "keywords": ["inheritance", "polymorphism", "overloading", "overriding",
                      "virtual function", "single inheritance", "multiple inheritance",
                      "multilevel", "hierarchical", "hybrid", "dynamic binding",
                      "static binding", "pure virtual", "abstract class",
                      "virtual destructor", "base class", "derived class",
                      "method overriding", "operator overloading"]},
        {"unit": "UNIT-IV", "title": "I/O and File Handling",
         "keywords": ["exception", "exception handling", "try", "catch", "throw",
                      "file", "file handling", "stream", "i/o", "fstream",
                      "ifstream", "ofstream", "formatted", "unformatted",
                      "stream class", "hierarchy", "overloading operator",
                      "cin", "cout", "cerr"]},
        {"unit": "UNIT-V",  "title": "Exception Handling, Templates and Java",
         "keywords": ["stl", "standard template library", "vector", "map", "java",
                      "interface", "package", "applet", "thread", "swing",
                      "template", "function template", "class template", "generic",
                      "annotation", "scrollpane", "life cycle", "awt",
                      "exception specification", "stack unwinding", "rethrowing"]},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_pages(path: Path) -> List[str]:
    try:
        reader = PdfReader(str(path))
        return [(p.extract_text() or "") for p in reader.pages]
    except Exception as e:
        print(f"  [warn] could not read {path}: {e}")
        return []


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Find the page(s) belonging to the target course and extract unit topics
# ─────────────────────────────────────────────────────────────────────────────
def _subject_page_span(pages: List[str], subject: str) -> Tuple[int, int]:
    """
    Return (start_page_idx, end_page_idx) for the subject's course block.

    Strategy (v2):
      1. Find pages that have "Objectives"/"Prerequisites"/"Outcomes" AND a
         matching course keyword.
      2. Among those, **prefer** pages where the course title appears as an
         actual title in the first ~400 chars of the page (the header area).
         This prevents picking up e.g. "Programming for Problem Solving"
         (which mentions "data structures" in its body) over the actual
         "DATA STRUCTURES" course.
      3. If a title-verified page is found, use it.  Otherwise fall back to
         the first keyword-matched page.
    """
    keywords    = SUBJECT_COURSE_KEYWORDS[subject]
    title_names = SUBJECT_COURSE_TITLES.get(subject, keywords)

    # Phase 1: collect all candidate pages
    cand_pages: List[int] = []
    title_verified_pages: List[int] = []

    for idx, raw in enumerate(pages):
        low = _norm(raw)
        has_structure = ("objectives" in low or "prerequisites" in low
                         or "outcomes" in low)
        if not has_structure:
            continue

        keyword_match = False
        for kw in keywords:
            if kw in low:
                keyword_match = True
                break

        if not keyword_match:
            continue

        cand_pages.append(idx)

        # Check if the course title appears in the HEADER area (first ~400 chars)
        header = _norm(raw[:400])
        for title in title_names:
            if title in header:
                # Extra check: make sure this isn't a lab page or a different course
                # that merely has the title as a substring.  E.g. "DATA STRUCTURES LAB"
                # or "ADVANCED DATA STRUCTURES" should be lower priority.
                is_exact = True
                for noise in ["lab", "advanced", "elective"]:
                    if noise in header and noise not in title:
                        is_exact = False
                        break
                if is_exact:
                    title_verified_pages.append(idx)
                break

    # Phase 2: pick the best start page
    if title_verified_pages:
        start = title_verified_pages[0]   # prefer title-verified
    elif cand_pages:
        start = cand_pages[0]
    else:
        return (-1, -1)

    # End is either the start of the next *different* course or 6-8 pages later
    all_chosen = set(title_verified_pages or cand_pages)
    for idx in range(start + 1, min(start + 10, len(pages))):
        low = _norm(pages[idx])
        has_new_header = ("objectives" in low or "prerequisites" in low
                          or "outcomes" in low)
        if has_new_header and idx not in all_chosen:
            return (start, idx)

    return (start, min(start + 8, len(pages)))


def _extract_unit_topics(pages: List[str], subject: str) -> List[Dict]:
    start, end = _subject_page_span(pages, subject)
    if start == -1:
        return []

    # combine the page span into one text block
    combined = "\n\n".join(pages[start:end])

    # Split out unit blocks
    unit_pat = re.compile(r"UNIT\s*[-–]\s*([IVX]+)", re.IGNORECASE)
    matches  = list(unit_pat.finditer(combined))
    if not matches:
        return []

    topics: List[Dict] = []
    for i, m in enumerate(matches):
        s = m.start()
        e = matches[i + 1].start() if i + 1 < len(matches) else min(s + 800, len(combined))
        raw = re.sub(r"\s+", " ", combined[s:e]).strip()

        # Extract a short title (text before the first ':' or long pause)
        after = re.sub(r"UNIT\s*[-–]\s*[IVX]+\s*", "", raw, count=1, flags=re.IGNORECASE).strip()
        tm    = re.match(r"^(.{4,80}?)(?:\s*[:]|\s{2,})", after)
        title = tm.group(1).strip() if tm else after[:60].strip()
        title = re.sub(r"[^a-zA-Z0-9 &'()\-/]", "", title).strip()

        # Build a keyword set from the full block
        content   = after[:800]
        kw_set: set = set()
        # individual words 4+ chars
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9\-']{3,}", content):
            kw_set.add(tok.lower())
        # bigrams and trigrams
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-']{2,}", content)
        for size in (2, 3):
            for j in range(len(words) - size + 1):
                phrase = " ".join(w.lower() for w in words[j: j + size])
                if len(phrase) >= 5:
                    kw_set.add(phrase)

        topics.append({
            "unit":     f"UNIT-{m.group(1).upper()}",
            "title":    title,
            "keywords": sorted(kw_set),
        })

    return topics


def _validate_ds_topics(topics: List[Dict]) -> bool:
    """Check if the extracted DS topics look correct (contain data-structure terms)."""
    if not topics:
        return False
    all_titles = " ".join(t["title"].lower() for t in topics)
    all_kws    = " ".join(" ".join(t["keywords"]) for t in topics)
    combined   = all_titles + " " + all_kws
    # A correct DS syllabus should mention at least some of these
    markers = ["stack", "queue", "tree", "graph", "linked list", "sorting",
               "searching", "hashing", "traversal", "heap"]
    hits = sum(1 for m in markers if m in combined)
    return hits >= 3


def _validate_oops_topics(topics: List[Dict]) -> bool:
    """Check if the extracted OOPS topics look correct."""
    if not topics:
        return False
    all_titles = " ".join(t["title"].lower() for t in topics)
    all_kws    = " ".join(" ".join(t["keywords"]) for t in topics)
    combined   = all_titles + " " + all_kws
    markers = ["class", "object", "inheritance", "polymorphism", "virtual",
               "overloading", "exception", "constructor"]
    hits = sum(1 for m in markers if m in combined)
    return hits >= 3


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract questions from question-paper PDFs
# ─────────────────────────────────────────────────────────────────────────────
_IMPERATIVE = re.compile(
    r"(?i)\b(explain|define|compare|differentiate|distinguish|discuss|describe|"
    r"what is|what are|write(?:\s+short)?\s+notes|list|enumerate|"
    r"give (?:an )?example|draw and explain|derive|prove|illustrate)\b"
)


def _extract_questions(text: str) -> List[str]:
    normalised = re.sub(r"[ \t]+", " ", text.replace("\r", "\n"))
    normalised = re.sub(r"\n+", "\n", normalised)

    blocks = re.split(
        r"(?:^|\n)\s*(?:\d+\s*[.)]\s*|[a-z][.)]\s*)",
        normalised,
        flags=re.MULTILINE,
    )

    questions: List[str] = []
    seen: set = set()

    for block in blocks:
        cleaned = re.sub(r"\s+", " ", block).strip(" -")
        if len(cleaned) < 15:
            continue

        # '?' fragments
        for m in re.finditer(r"[^?]{10,220}\?", cleaned):
            q = m.group(0).strip()
            lq = q.lower()
            if lq not in seen and len(q) > 15:
                seen.add(lq)
                questions.append(q[0].upper() + q[1:])

        # imperative lines
        m2 = _IMPERATIVE.match(cleaned)
        if m2:
            q = cleaned[:220].strip()
            if not q.endswith("?"):
                q += "?"
            lq = q.lower()
            if lq not in seen:
                seen.add(lq)
                questions.append(q[0].upper() + q[1:])

    return questions


# ─────────────────────────────────────────────────────────────────────────────
# 3. Score a question against a topic
# ─────────────────────────────────────────────────────────────────────────────
def _score(question: str, topic: Dict) -> float:
    ql = question.lower()
    s  = 0.0
    for kw in topic["keywords"]:
        if kw in ql:
            s += len(kw.split()) ** 1.5   # longer phrases worth more
    return s


def _score_rescue(question: str, topic: Dict) -> float:
    """Broader scoring for the rescue pass — also matches individual words
    from the title and uses shorter tokens."""
    ql = question.lower()
    s  = 0.0

    # Title-word matching
    title_words = set(re.findall(r"[a-z]{3,}", topic["title"].lower()))
    q_words     = set(re.findall(r"[a-z]{3,}", ql))
    overlap     = title_words & q_words
    # Remove very generic words
    generic = {"the", "and", "for", "with", "from", "using", "are", "what",
               "how", "explain", "describe", "define", "discuss", "write",
               "give", "list", "draw", "compare", "example", "about", "between"}
    overlap -= generic
    s += len(overlap) * 2.0

    # Keyword matching (same as primary but also counts partial matches)
    for kw in topic["keywords"]:
        if kw in ql:
            s += len(kw.split()) ** 1.3
        elif len(kw) >= 5:
            # Check if all significant words from the keyword are in the question
            kw_parts = set(re.findall(r"[a-z]{3,}", kw))
            kw_parts -= generic
            if kw_parts and kw_parts.issubset(q_words):
                s += 0.5

    return s


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────
def build_index() -> Dict:
    base   = Path(__file__).resolve().parent.parent
    result: Dict[str, Dict] = {}

    for subject in SUBJECT_COURSE_KEYWORDS:
        print(f"\n{'='*60}")
        print(f"Subject: {subject.upper()}")

        # Find syllabus PDF
        syllabus_pdf: Optional[Path] = None
        for candidate in [
            base / "data" / "subjects" / subject / "syllabus" / "syllabus_cse.pdf",
            base / "data" / "curriculum" / "course_structure" / "syllabus_cse.pdf",
        ]:
            if candidate.exists():
                syllabus_pdf = candidate
                break

        if syllabus_pdf is None:
            print("  [skip] no syllabus found")
            continue

        pages       = _pdf_pages(syllabus_pdf)
        unit_topics = _extract_unit_topics(pages, subject)

        # ── Validation: ensure extracted topics look correct ──────────────
        use_fallback = False
        if subject == "ds" and not _validate_ds_topics(unit_topics):
            print("  [warn] DS topics don't look like Data Structures — using fallback")
            use_fallback = True
        elif subject == "oops" and not _validate_oops_topics(unit_topics):
            print("  [warn] OOPS topics don't look like OOP — using fallback")
            use_fallback = True
        elif not unit_topics:
            print("  [warn] no unit topics found from PDF")
            use_fallback = True

        if use_fallback and subject in FALLBACK_TOPIC_MAPS:
            unit_topics = FALLBACK_TOPIC_MAPS[subject]
            print("  [info] using fallback topic map")
        elif use_fallback:
            print("  [skip] no fallback available")
            continue

        # ── Enrich PDF-extracted keywords with fallback keywords ─────────
        # Even when PDF extraction succeeds, fallback maps may contain
        # broader domain-specific keywords (e.g., Java terms for OOPS)
        # that the syllabus text omits.  Merge them in.
        # Enrichment (except for OOPS as per user request to leave them unmatched)
        if not use_fallback and subject in FALLBACK_TOPIC_MAPS and subject != "oops":
            fb = FALLBACK_TOPIC_MAPS[subject]
            for t in unit_topics:
                fb_match = next((f for f in fb if f["unit"] == t["unit"]), None)
                if fb_match:
                    existing = set(t["keywords"])
                    for kw in fb_match["keywords"]:
                        if kw not in existing:
                            t["keywords"].append(kw)

        for t in unit_topics:
            print(f"  {t['unit']}: {t['title'][:70]}")

        # Extract questions from all question-paper PDFs
        qp_root       = base / "data" / "subjects" / subject / "question_papers"
        all_questions: List[Dict] = []
        if qp_root.exists():
            for pdf_path in sorted(qp_root.glob("*.pdf")):
                text = "\n".join(_pdf_pages(pdf_path))
                qs   = _extract_questions(text)
                for q in qs:
                    all_questions.append({"question": q, "source": pdf_path.name})
                print(f"  {pdf_path.name}: {len(qs)} Qs")

        print(f"  total: {len(all_questions)} questions")

        # ── Map questions → topics (primary pass) ────────────────────────
        topic_map: Dict[str, Dict] = {}
        for t in unit_topics:
            topic_map[t["unit"]] = {
                "title":     t["title"],
                "keywords":  t["keywords"][:50],   # keep more keywords
                "questions": [],
            }

        unmatched: List[Dict] = []
        for qd in all_questions:
            q, src        = qd["question"], qd["source"]
            best, best_sc = None, 0.0
            for t in unit_topics:
                sc = _score(q, t)
                if sc > best_sc:
                    best_sc = sc
                    best    = t["unit"]
            if best and best_sc > 0:
                topic_map[best]["questions"].append({
                    "question": q,
                    "source":   src,
                    "score":    round(best_sc, 2),
                })
            else:
                unmatched.append(qd)

        # ── Rescue pass: try broader matching for unmatched questions ────
        still_unmatched: List[Dict] = []
        for qd in unmatched:
            q, src        = qd["question"], qd["source"]
            best, best_sc = None, 0.0
            for t in unit_topics:
                sc = _score_rescue(q, t)
                if sc > best_sc:
                    best_sc = sc
                    best    = t["unit"]
            if best and best_sc >= 1.0:
                topic_map[best]["questions"].append({
                    "question": q,
                    "source":   src,
                    "score":    round(best_sc, 2),
                })
            else:
                still_unmatched.append(qd)

        for uk in topic_map:
            topic_map[uk]["questions"].sort(key=lambda x: x["score"], reverse=True)

        result[subject] = {"unit_topics": topic_map, "unmatched": still_unmatched}

        matched = sum(len(v["questions"]) for v in topic_map.values())
        print(f"  matched={matched}  unmatched={len(still_unmatched)}")

    out = base / "data" / "topic_question_index.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[done] saved to {out}")
    return result


if __name__ == "__main__":
    build_index()

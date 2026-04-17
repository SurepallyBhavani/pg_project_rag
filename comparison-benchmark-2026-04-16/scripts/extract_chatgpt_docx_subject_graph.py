from __future__ import annotations

from pathlib import Path
import csv
import json
import re
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DOCX_PATH = ROOT / "chatgpt ans.docx"
BENCHMARK_PATH = ROOT / "benchmark_queries_50.json"
CSV_OUTPUT = ROOT / "chatgpt_subject_graph_outputs.csv"
JSON_OUTPUT = ROOT / "chatgpt_subject_graph_outputs.json"


def main() -> int:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    target_items = [item for item in benchmark if item["category"] in {"subject", "graph"}]
    ordered_ids = [item["id"] for item in target_items]
    id_to_query = {item["id"]: item["query"] for item in target_items}
    text = _extract_docx_text(DOCX_PATH)

    parsed = _parse_sections(text, ordered_ids)
    rows = []
    for item_id in ordered_ids:
        answer = parsed.get(item_id, "").strip()
        rows.append(
            {
                "id": item_id,
                "category": "subject" if item_id.startswith("subject_") else "graph",
                "query": id_to_query[item_id],
                "chatgpt_output": answer,
            }
        )

    _write_csv(rows, CSV_OUTPUT)
    JSON_OUTPUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {CSV_OUTPUT}")
    return 0


def _extract_docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", xml)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_sections(text: str, ordered_ids: list[str]) -> dict[str, str]:
    sections: dict[str, str] = {}

    # subject_01 appears before the first explicit marker in the provided docx.
    first_marker = "subject_02,subject,"
    first_index = text.find(first_marker)
    if first_index != -1:
        sections["subject_01"] = text[:first_index].strip()

    for index, item_id in enumerate(ordered_ids[1:], start=1):
        category = "subject" if item_id.startswith("subject_") else "graph"
        marker = f"{item_id},{category},"
        start = text.find(marker)
        if start == -1:
            continue
        start = start + len(marker)

        next_start = len(text)
        for next_id in ordered_ids[index + 1:]:
            next_category = "subject" if next_id.startswith("subject_") else "graph"
            next_marker = f"{next_id},{next_category},"
            pos = text.find(next_marker, start)
            if pos != -1:
                next_start = pos
                break

        chunk = text[start:next_start].strip()
        first_comma = chunk.find(",,,")
        if first_comma != -1:
            chunk = chunk[first_comma + 3 :].strip()
        sections[item_id] = _cleanup_chunk(chunk)

    return sections


def _cleanup_chunk(chunk: str) -> str:
    chunk = re.sub(r"\s+", " ", chunk).strip()
    # Remove trailing unsupported/gibberish section if it bleeds into the last graph answer.
    cutoff = chunk.find("@@@ ### ???")
    if cutoff != -1:
        chunk = chunk[:cutoff].strip()
    return chunk


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "category", "query", "chatgpt_output"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

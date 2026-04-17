from __future__ import annotations

from pathlib import Path
import csv
import json
import re
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DOCX_PATH = ROOT / "notebooklm ans.docx"
BENCHMARK_PATH = ROOT / "benchmark_queries_50.json"
CSV_OUTPUT = ROOT / "notebooklm_outputs.csv"
JSON_OUTPUT = ROOT / "notebooklm_outputs.json"


def main() -> int:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    text = _extract_docx_text(DOCX_PATH)
    answers = _split_numbered_answers(text)

    rows = []
    for index, item in enumerate(benchmark, start=1):
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "query": item["query"],
                "notebooklm_output": answers.get(index, "").strip(),
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
    return re.sub(r"\s+", " ", text).strip()


def _split_numbered_answers(text: str) -> dict[int, str]:
    answers: dict[int, str] = {}
    start_phrase = "Here are the answers to your questions based on the uploaded academic sources:"
    start_index = text.find(start_phrase)
    if start_index != -1:
        text = text[start_index + len(start_phrase):].strip()

    matches = list(re.finditer(r"(?<!\d)(\d+)\.\s", text))
    for i, match in enumerate(matches):
        number = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        qmark_index = chunk.find("?")
        if qmark_index != -1:
            chunk = chunk[qmark_index + 1 :].strip()
        answers[number] = chunk
    return answers


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "category", "query", "notebooklm_output"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

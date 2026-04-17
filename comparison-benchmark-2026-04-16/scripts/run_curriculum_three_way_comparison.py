from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
BENCHMARK_PATH = ROOT / "benchmark_queries_50.json"
NOTEBOOKLM_PATH = ROOT / "notebooklm_outputs.json"
OUTPUT_ROOT = ROOT / "comparison_results"


def main() -> int:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    notebooklm_rows = {row["id"]: row for row in json.loads(NOTEBOOKLM_PATH.read_text(encoding="utf-8"))}

    eduassist_dir = PROJECT_ROOT / "rag-project-2026-04-09"
    baseline_dir = PROJECT_ROOT / "rag-baseline-2026-04-16"

    curriculum_queries = [
        item for item in benchmark
        if item["category"] == "curriculum"
        and item["id"] in notebooklm_rows
    ]

    judged_rows: List[Dict[str, Any]] = []
    for item in curriculum_queries:
        query = item["query"]
        eduassist_answer = _run_process_query(eduassist_dir, query)
        baseline_answer = _run_process_query(baseline_dir, query)
        notebooklm_answer = str(notebooklm_rows[item["id"]].get("notebooklm_output", ""))

        system_answers = {
            "notebooklm": notebooklm_answer,
            "baseline_rag": baseline_answer,
            "eduassist": eduassist_answer,
        }
        judgment = _heuristic_judgment(query, system_answers)
        judged_rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "query": query,
                "judgment": judgment,
            }
        )

    output_dir = _write_outputs(judged_rows)
    print(f"Curriculum three-way comparison written to: {output_dir}")
    return 0


def _run_process_query(project_dir: Path, query: str) -> str:
    script = (
        "import json\n"
        "from app import process_query\n"
        f"result = process_query({json.dumps(query)})\n"
        "print(json.dumps({'answer': str(result.get('answer', ''))}, ensure_ascii=False))\n"
    )
    completed = subprocess.run(
        ["python", "-c", script],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Failed running process_query in {project_dir}: {completed.stderr}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return ""
    payload = json.loads(lines[-1])
    return str(payload.get("answer", ""))


def _heuristic_judgment(query: str, system_answers: Dict[str, str]) -> Dict[str, Any]:
    systems = {}
    best_system = None
    best_total = -1.0
    for name, answer in system_answers.items():
        metrics = _score_answer(query, answer)
        systems[name] = metrics
        total = metrics["relevance"] + metrics["completeness"] + metrics["clarity"] + metrics["appropriateness"]
        if total > best_total:
            best_total = total
            best_system = name
    return {"systems": systems, "best_overall": best_system, "judge_mode": "heuristic"}


def _score_answer(query: str, answer: str) -> Dict[str, Any]:
    text = answer.strip()
    focus_terms = _focus_terms(query)
    answer_terms = _tokenize(text)
    overlap = len(set(focus_terms) & answer_terms)
    relevance_ratio = overlap / max(len(set(focus_terms)), 1)
    relevance = min(5, round(relevance_ratio * 5))

    length = len(text)
    completeness = 1
    if length > 60:
        completeness = 2
    if length > 140:
        completeness = 3
    if length > 260:
        completeness = 4
    if length > 420:
        completeness = 5

    structure_markers = ["unit", "semester", "l-t-p-c", "textbook", "prerequisite", "lab"]
    appropriateness = 2
    if any(marker in text.lower() for marker in structure_markers):
        appropriateness = 4
    if overlap >= 1 and any(marker in text.lower() for marker in structure_markers):
        appropriateness = 5

    sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
    clarity = 2
    if sentence_count >= 2:
        clarity = 3
    if sentence_count >= 4:
        clarity = 4
    if sentence_count >= 5 and length < 2000:
        clarity = 5

    note = f"Overlap={overlap}, length={length}, sentences={sentence_count}"
    return {
        "relevance": int(relevance),
        "completeness": int(completeness),
        "clarity": int(clarity),
        "appropriateness": int(appropriateness),
        "note": note,
    }


def _focus_terms(query: str) -> List[str]:
    stopwords = {
        "what", "is", "the", "of", "for", "in", "which", "according", "to", "course",
        "structure", "list", "all", "under", "are",
    }
    return [token for token in _tokenize(query) if token not in stopwords]


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}


def _write_outputs(judged_rows: List[Dict[str, Any]]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "curriculum_judgments.json").write_text(json.dumps(judged_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = _summarize(judged_rows)
    (output_dir / "curriculum_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (output_dir / "curriculum_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "avg_relevance", "avg_completeness", "avg_clarity", "avg_appropriateness", "avg_total", "wins"])
        writer.writeheader()
        writer.writerows(summary["systems"])

    lines = [
        "# Curriculum Three-Way Comparison Summary",
        "",
        f"- Compared queries: {summary['query_count']}",
        "",
        "## System Results",
        "",
    ]
    for item in summary["systems"]:
        lines.extend([
            f"### {item['system']}",
            f"- Avg relevance: {item['avg_relevance']:.2f}",
            f"- Avg completeness: {item['avg_completeness']:.2f}",
            f"- Avg clarity: {item['avg_clarity']:.2f}",
            f"- Avg appropriateness: {item['avg_appropriateness']:.2f}",
            f"- Avg total: {item['avg_total']:.2f}",
            f"- Query wins: {item['wins']}",
            "",
        ])
    (output_dir / "curriculum_report.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def _summarize(judged_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    score_store: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    wins = defaultdict(int)

    for row in judged_rows:
        judgment = row["judgment"]
        systems = judgment.get("systems", {})
        for system, metrics in systems.items():
            for metric in ["relevance", "completeness", "clarity", "appropriateness"]:
                score_store[system][metric].append(float(metrics.get(metric, 0)))
            total = sum(float(metrics.get(metric, 0)) for metric in ["relevance", "completeness", "clarity", "appropriateness"])
            score_store[system]["total"].append(total)
        best = judgment.get("best_overall")
        if best:
            wins[str(best)] += 1

    systems_summary = []
    for system, metrics in score_store.items():
        systems_summary.append(
            {
                "system": system,
                "avg_relevance": statistics.mean(metrics["relevance"]) if metrics["relevance"] else 0,
                "avg_completeness": statistics.mean(metrics["completeness"]) if metrics["completeness"] else 0,
                "avg_clarity": statistics.mean(metrics["clarity"]) if metrics["clarity"] else 0,
                "avg_appropriateness": statistics.mean(metrics["appropriateness"]) if metrics["appropriateness"] else 0,
                "avg_total": statistics.mean(metrics["total"]) if metrics["total"] else 0,
                "wins": wins.get(system, 0),
            }
        )
    systems_summary.sort(key=lambda item: item["avg_total"], reverse=True)

    return {
        "query_count": len(judged_rows),
        "systems": systems_summary,
    }


if __name__ == "__main__":
    raise SystemExit(main())

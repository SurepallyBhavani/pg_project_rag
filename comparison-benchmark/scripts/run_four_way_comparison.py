from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import subprocess
import statistics
import time
from typing import Any, Dict, List
import re

from dotenv import load_dotenv
load_dotenv()

import openai  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
BENCHMARK_PATH = ROOT / "benchmark_queries_50.json"
CHATGPT_PATH = ROOT / "chatgpt_subject_graph_outputs.json"
NOTEBOOKLM_PATH = ROOT / "notebooklm_outputs.json"
OUTPUT_ROOT = ROOT / "comparison_results"
def main() -> int:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    notebooklm_rows = {row["id"]: row for row in json.loads(NOTEBOOKLM_PATH.read_text(encoding="utf-8"))}
    chatgpt_rows = {row["id"]: row for row in json.loads(CHATGPT_PATH.read_text(encoding="utf-8"))}

    eduassist_dir = PROJECT_ROOT / "rag-project-2026-04-09"
    baseline_dir = PROJECT_ROOT / "rag-baseline-2026-04-16"

    common_queries = [
        item for item in benchmark
        if item["category"] in {"subject", "graph"}
        and item["id"] in chatgpt_rows
        and item["id"] in notebooklm_rows
    ]

    judged_rows: List[Dict[str, Any]] = []
    for item in common_queries:
        query = item["query"]
        eduassist_data = _run_process_query(eduassist_dir, query)
        baseline_data = _run_process_query(baseline_dir, query)
        chatgpt_answer = str(chatgpt_rows[item["id"]].get("chatgpt_output", ""))
        notebooklm_answer = str(notebooklm_rows[item["id"]].get("notebooklm_output", ""))

        system_answers = {
            "chatgpt": {"answer": chatgpt_answer, "context": []},
            "notebooklm": {"answer": notebooklm_answer, "context": []},
            "baseline_rag": baseline_data,
            "eduassist": eduassist_data,
        }
        judgment = _judge_query(query, item["category"], system_answers)
        judged_rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "query": query,
                "judgment": judgment,
            }
        )

    output_dir = _write_outputs(judged_rows)
    print(f"Four-way comparison written to: {output_dir}")
    return 0


def _run_process_query(project_dir: Path, query: str) -> Dict[str, Any]:
    script = (
        "import sys\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        "import json\n"
        "from app import process_query\n"
        f"result = process_query({json.dumps(query)})\n"
        "answer = str(result.get('answer', ''))\n"
        "evidence = result.get('evidence', [])\n"
        "sources = [e.get('snippet', '') for e in evidence if 'snippet' in e] if isinstance(evidence, list) else []\n"
        "print(json.dumps({'answer': answer, 'context': sources}, ensure_ascii=False))\n"
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
        return {"answer": "", "context": []}
    payload = json.loads(lines[-1])
    return {"answer": str(payload.get("answer", "")), "context": payload.get("context", [])}


def _judge_query(query: str, category: str, system_answers: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    prompt = {
        "query": query,
        "category": category,
        "systems": system_answers,
        "rubric": {
            "faithfulness": "0-5: Does the generated answer rely only on the context provided without hallucination?",
            "answer_relevance": "0-5: Does the answer directly address the user's string query?",
            "context_precision": "0-5: Are the retrieved context chunks (if present) highly relevant? If context is empty/missing, score as null.",
            "overall_quality": "0-5: General structure, readability, and academic appropriateness.",
        },
    }

    max_retries = 4
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                        "You are an impartial evaluator comparing four academic assistant answers (and their retrieved contexts, if any) to the same query. "
                        "Return strict JSON exactly matching this format: "
                        "{\"systems\": {\"system_name\": {\"faithfulness\": int, \"answer_relevance\": int, \"context_precision\": int|null, \"overall_quality\": int, \"note\": \"string\"}}, \"best_overall\": \"system_name\"}. "
                        "Assign integer scores from 0 to 5. If a system has no context chunks, return null for context_precision. "
                        "Also provide a short one-sentence note per system and identify the best_overall system for this query."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            result = json.loads(content)
            if "systems" not in result:
                return {"systems": {}, "best_overall": None, "error": "Invalid API schema"}
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 2)  # 2, 4, 8 seconds
            else:
                print(f"Failed to judge query '{query}' after {max_retries} attempts: {e}")
                return {"systems": {}, "best_overall": None, "error": str(e)}


def _write_outputs(judged_rows: List[Dict[str, Any]]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "judgments.json").write_text(json.dumps(judged_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = _summarize(judged_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "avg_faithfulness", "avg_answer_relevance", "avg_context_precision", "avg_overall_quality", "avg_total", "wins"])
        writer.writeheader()
        writer.writerows(summary["systems"])

    lines = [
        "# Four-Way Comparison Summary",
        "",
        f"- Compared queries: {summary['query_count']}",
        f"- Categories: {', '.join(summary['categories'])}",
        "",
        "## System Results",
        "",
    ]
    for item in summary["systems"]:
        lines.extend([
            f"### {item['system']}",
            f"- Avg Faithfulness: {item['avg_faithfulness']:.2f}",
            f"- Avg Answer Relevance: {item['avg_answer_relevance']:.2f}",
            f"- Avg Context Precision: {item['avg_context_precision']:.2f}" if item['avg_context_precision'] > 0 else f"- Avg Context Precision: N/A",
            f"- Avg Overall Quality: {item['avg_overall_quality']:.2f}",
            f"- Avg Total: {item['avg_total']:.2f}",
            f"- Query wins: {item['wins']}",
            "",
        ])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def _summarize(judged_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    score_store: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    wins = defaultdict(int)
    categories = sorted({row["category"] for row in judged_rows})

    for row in judged_rows:
        judgment = row["judgment"]
        systems = judgment.get("systems", {})
        for system, metrics in systems.items():
            if not isinstance(metrics, dict): continue
            
            for metric in ["faithfulness", "answer_relevance", "overall_quality"]:
                score_store[system][metric].append(float(metrics.get(metric) or 0))
                
            if metrics.get("context_precision") is not None:
                score_store[system]["context_precision"].append(float(metrics["context_precision"]))
            
            # Context precision is excluded from total for non-RAG models to keep comparisons fair,
            # or we calculate an average from available scores.
            valid_metrics = [
                float(metrics.get(m) or 0) for m in ["faithfulness", "answer_relevance", "overall_quality"]
            ]
            total = sum(valid_metrics)
            score_store[system]["total"].append(total)

        best = judgment.get("best_overall")
        if best:
            wins[str(best)] += 1

    systems_summary = []
    for system, metrics in score_store.items():
        systems_summary.append(
            {
                "system": system,
                "avg_faithfulness": statistics.mean(metrics["faithfulness"]) if metrics["faithfulness"] else 0,
                "avg_answer_relevance": statistics.mean(metrics["answer_relevance"]) if metrics["answer_relevance"] else 0,
                "avg_context_precision": statistics.mean(metrics["context_precision"]) if metrics.get("context_precision") else 0,
                "avg_overall_quality": statistics.mean(metrics["overall_quality"]) if metrics["overall_quality"] else 0,
                "avg_total": statistics.mean(metrics["total"]) if metrics["total"] else 0,
                "wins": wins.get(system, 0),
            }
        )
    systems_summary.sort(key=lambda item: item["avg_total"], reverse=True)

    return {
        "query_count": len(judged_rows),
        "categories": categories,
        "systems": systems_summary,
    }


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = PROJECT_ROOT / "artifacts" / "feedback.jsonl"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "feedback_analysis"


def main() -> int:
    if not FEEDBACK_PATH.exists():
        print("No feedback file found yet.")
        return 0

    rows = _load_feedback()
    if not rows:
        print("No valid feedback rows found.")
        return 0

    summary = _build_summary(rows)
    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(_build_report(summary), encoding="utf-8")
    print(f"Feedback analysis written to: {run_dir}")
    return 0


def _load_feedback() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with FEEDBACK_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = str(item.get("feedback", "")).strip().lower()
            if label not in {"helpful", "not_helpful"}:
                continue
            rows.append(item)
    return rows


def _build_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    helpful = [row for row in rows if row.get("feedback") == "helpful"]
    not_helpful = [row for row in rows if row.get("feedback") == "not_helpful"]

    by_query_type = _bucket_feedback(rows, "query_type")
    by_method = _bucket_feedback(rows, "retrieval_method")
    by_subject = _bucket_feedback(rows, "subject")

    recommendations = _build_recommendations(by_query_type, by_method, by_subject, len(rows))

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_feedback": len(rows),
        "helpful": len(helpful),
        "not_helpful": len(not_helpful),
        "helpful_ratio": round(len(helpful) / max(len(rows), 1), 3),
        "not_helpful_ratio": round(len(not_helpful) / max(len(rows), 1), 3),
        "query_type_summary": by_query_type,
        "retrieval_method_summary": by_method,
        "subject_summary": by_subject,
        "recommendations": recommendations,
    }


def _bucket_feedback(rows: List[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    helpful_counter: Counter[str] = Counter()
    unhelpful_counter: Counter[str] = Counter()

    for row in rows:
        bucket = str(row.get(key) or "unknown").strip() or "unknown"
        if row.get("feedback") == "helpful":
            helpful_counter[bucket] += 1
        else:
            unhelpful_counter[bucket] += 1

    all_keys = sorted(set(helpful_counter) | set(unhelpful_counter))
    summary: List[Dict[str, object]] = []
    for bucket in all_keys:
        helpful = helpful_counter[bucket]
        unhelpful = unhelpful_counter[bucket]
        total = helpful + unhelpful
        summary.append(
            {
                "bucket": bucket,
                "helpful": helpful,
                "not_helpful": unhelpful,
                "total": total,
                "helpful_ratio": round(helpful / max(total, 1), 3),
                "not_helpful_ratio": round(unhelpful / max(total, 1), 3),
            }
        )
    summary.sort(key=lambda item: (item["not_helpful_ratio"], item["total"]), reverse=True)
    return summary


def _build_recommendations(
    query_types: List[Dict[str, object]],
    methods: List[Dict[str, object]],
    subjects: List[Dict[str, object]],
    total_rows: int,
) -> List[str]:
    recommendations: List[str] = []

    for item in query_types:
        if item["total"] >= max(2, total_rows // 10) and item["not_helpful_ratio"] >= 0.5:
            recommendations.append(
                f"Review prompt and retrieval behavior for query type '{item['bucket']}', as it shows a high not-helpful ratio ({item['not_helpful_ratio']:.2f})."
            )

    for item in methods:
        if item["total"] >= max(2, total_rows // 10) and item["not_helpful_ratio"] >= 0.5:
            recommendations.append(
                f"Review retrieval method '{item['bucket']}' because it is associated with frequent not-helpful feedback ({item['not_helpful_ratio']:.2f})."
            )

    for item in subjects:
        if item["bucket"] != "None" and item["total"] >= max(2, total_rows // 10) and item["not_helpful_ratio"] >= 0.5:
            recommendations.append(
                f"Consider adding more or better quality material for subject '{item['bucket']}', which currently shows weaker feedback outcomes."
            )

    if not recommendations:
        recommendations.append("Current feedback does not show a concentrated weak area yet; continue collecting more user ratings before changing retrieval heuristics.")

    return recommendations


def _build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# EduAssist Feedback Analysis",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Total feedback records: {summary['total_feedback']}",
        f"- Helpful: {summary['helpful']}",
        f"- Not helpful: {summary['not_helpful']}",
        f"- Helpful ratio: {summary['helpful_ratio']:.3f}",
        f"- Not helpful ratio: {summary['not_helpful_ratio']:.3f}",
        "",
        "## Query Type Summary",
        "",
    ]

    for item in summary["query_type_summary"]:
        lines.append(
            f"- {item['bucket']}: helpful={item['helpful']}, not_helpful={item['not_helpful']}, not_helpful_ratio={item['not_helpful_ratio']:.3f}"
        )

    lines.extend(["", "## Retrieval Method Summary", ""])
    for item in summary["retrieval_method_summary"]:
        lines.append(
            f"- {item['bucket']}: helpful={item['helpful']}, not_helpful={item['not_helpful']}, not_helpful_ratio={item['not_helpful_ratio']:.3f}"
        )

    lines.extend(["", "## Subject Summary", ""])
    for item in summary["subject_summary"]:
        lines.append(
            f"- {item['bucket']}: helpful={item['helpful']}, not_helpful={item['not_helpful']}, not_helpful_ratio={item['not_helpful_ratio']:.3f}"
        )

    lines.extend(["", "## Recommendations", ""])
    for item in summary["recommendations"]:
        lines.append(f"- {item}")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

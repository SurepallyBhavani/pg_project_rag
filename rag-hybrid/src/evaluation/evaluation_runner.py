from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore


@dataclass
class EvaluationQuery:
    id: str
    category: str
    query: str
    expected_query_type: str
    expected_retrieval_method: str
    expected_subject: Optional[str]
    expected_contains: List[str]


class EvaluationRunner:
    def __init__(self, project_root: str, process_query_fn, retrieval_method_fn):
        self.project_root = Path(project_root)
        self.process_query = process_query_fn
        self.get_retrieval_method = retrieval_method_fn
        self.output_root = self.project_root / "artifacts" / "evaluation"
        self.dataset_path = self.project_root / "data" / "evaluation_queries.json"

    def run(self) -> Path:
        queries = self._load_queries()
        run_dir = self._create_run_dir()
        results = [self._evaluate_single(query) for query in queries]

        summary = self._build_summary(results)
        self._write_outputs(run_dir, results, summary)
        self._write_visualizations(run_dir, results, summary)
        return run_dir

    def _load_queries(self) -> List[EvaluationQuery]:
        payload = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        return [EvaluationQuery(**item) for item in payload]

    def _create_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _evaluate_single(self, query: EvaluationQuery) -> Dict[str, Any]:
        started = time.perf_counter()
        response = self.process_query(query.query)
        elapsed = round(time.perf_counter() - started, 3)

        route = response.get("query_route", {}) or {}
        answer = str(response.get("answer", ""))
        retrieval_method = self.get_retrieval_method(response)
        actual_subject = route.get("subject")
        sources = response.get("sources", []) or []
        confidence = response.get("confidence")

        contains_hits = {
            phrase: phrase.lower() in answer.lower()
            for phrase in query.expected_contains
        }
        keyword_coverage = (
            sum(1 for matched in contains_hits.values() if matched) / max(len(contains_hits), 1)
        )

        route_match = route.get("query_type") == query.expected_query_type
        retrieval_match = retrieval_method == query.expected_retrieval_method
        subject_match = (
            True if query.expected_subject is None
            else str(actual_subject or "").lower() == query.expected_subject.lower()
        )
        citations_present = len(sources) > 0
        confidence_applicable = confidence is not None

        overall_score = round(
            (
                (1.0 if route_match else 0.0) * 0.25
                + (1.0 if retrieval_match else 0.0) * 0.2
                + (1.0 if subject_match else 0.0) * 0.15
                + keyword_coverage * 0.25
                + (1.0 if citations_present else 0.0) * 0.15
            ),
            3,
        )

        return {
            "id": query.id,
            "category": query.category,
            "query": query.query,
            "expected_query_type": query.expected_query_type,
            "actual_query_type": route.get("query_type"),
            "route_match": route_match,
            "expected_retrieval_method": query.expected_retrieval_method,
            "actual_retrieval_method": retrieval_method,
            "retrieval_match": retrieval_match,
            "expected_subject": query.expected_subject,
            "actual_subject": actual_subject,
            "subject_match": subject_match,
            "expected_contains": query.expected_contains,
            "contains_hits": contains_hits,
            "keyword_coverage": round(keyword_coverage, 3),
            "citations_present": citations_present,
            "citation_count": len(sources),
            "confidence": confidence,
            "confidence_applicable": confidence_applicable,
            "latency_seconds": elapsed,
            "overall_score": overall_score,
            "answer_preview": answer[:500],
        }

    def _build_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = pd.DataFrame(results)
        total = len(df)
        category_summary = (
            df.groupby("category")
            .agg(
                queries=("id", "count"),
                avg_score=("overall_score", "mean"),
                avg_latency=("latency_seconds", "mean"),
                route_accuracy=("route_match", "mean"),
                retrieval_accuracy=("retrieval_match", "mean"),
                keyword_coverage=("keyword_coverage", "mean"),
                citation_coverage=("citations_present", "mean"),
            )
            .reset_index()
        )

        method_summary = (
            df.groupby("actual_retrieval_method")
            .agg(
                queries=("id", "count"),
                avg_score=("overall_score", "mean"),
                avg_latency=("latency_seconds", "mean"),
                avg_confidence=("confidence", "mean"),
            )
            .reset_index()
        )

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total_queries": total,
            "route_accuracy": round(float(df["route_match"].mean()), 3),
            "retrieval_accuracy": round(float(df["retrieval_match"].mean()), 3),
            "subject_accuracy": round(float(df["subject_match"].mean()), 3),
            "citation_coverage": round(float(df["citations_present"].mean()), 3),
            "average_keyword_coverage": round(float(df["keyword_coverage"].mean()), 3),
            "average_latency_seconds": round(float(df["latency_seconds"].mean()), 3),
            "average_overall_score": round(float(df["overall_score"].mean()), 3),
            "category_summary": category_summary.to_dict(orient="records"),
            "method_summary": method_summary.fillna(0).to_dict(orient="records"),
        }

    def _write_outputs(self, run_dir: Path, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        results_df = pd.DataFrame(results)
        results_df.to_csv(run_dir / "results.csv", index=False)

        summary_lines = [
            "# EduAssist Evaluation Report",
            "",
            f"- Generated at: {summary['generated_at']}",
            f"- Total queries: {summary['total_queries']}",
            f"- Route accuracy: {summary['route_accuracy']:.3f}",
            f"- Retrieval label accuracy: {summary['retrieval_accuracy']:.3f}",
            f"- Subject accuracy: {summary['subject_accuracy']:.3f}",
            f"- Citation coverage: {summary['citation_coverage']:.3f}",
            f"- Average keyword coverage: {summary['average_keyword_coverage']:.3f}",
            f"- Average latency (s): {summary['average_latency_seconds']:.3f}",
            f"- Average overall score: {summary['average_overall_score']:.3f}",
            "",
            "## Category Summary",
            "",
        ]

        for item in summary["category_summary"]:
            summary_lines.extend([
                f"### {item['category']}",
                f"- Queries: {item['queries']}",
                f"- Avg score: {item['avg_score']:.3f}",
                f"- Avg latency (s): {item['avg_latency']:.3f}",
                f"- Route accuracy: {item['route_accuracy']:.3f}",
                f"- Retrieval accuracy: {item['retrieval_accuracy']:.3f}",
                f"- Keyword coverage: {item['keyword_coverage']:.3f}",
                f"- Citation coverage: {item['citation_coverage']:.3f}",
                "",
            ])

        (run_dir / "report.md").write_text("\n".join(summary_lines), encoding="utf-8")

    def _write_visualizations(self, run_dir: Path, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        charts_dir = run_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        category_df = pd.DataFrame(summary["category_summary"])
        method_df = pd.DataFrame(summary["method_summary"])
        results_df = pd.DataFrame(results)

        self._plot_category_scores(category_df, charts_dir / "category_scores.png")
        self._plot_latency_by_category(category_df, charts_dir / "latency_by_category.png")
        self._plot_method_scores(method_df, charts_dir / "method_scores.png")
        self._plot_query_score_distribution(results_df, charts_dir / "query_score_distribution.png")

    def _plot_category_scores(self, df: pd.DataFrame, output_path: Path) -> None:
        plt.figure(figsize=(10, 5))
        plt.bar(df["category"], df["avg_score"], color="#2f536d")
        plt.ylim(0, 1.0)
        plt.title("Average Evaluation Score by Category")
        plt.ylabel("Average Score")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

    def _plot_latency_by_category(self, df: pd.DataFrame, output_path: Path) -> None:
        plt.figure(figsize=(10, 5))
        plt.bar(df["category"], df["avg_latency"], color="#b77b45")
        plt.title("Average Latency by Category")
        plt.ylabel("Latency (seconds)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

    def _plot_method_scores(self, df: pd.DataFrame, output_path: Path) -> None:
        plt.figure(figsize=(10, 5))
        plt.bar(df["actual_retrieval_method"], df["avg_score"], color="#5d7a3e")
        plt.ylim(0, 1.0)
        plt.title("Average Score by Retrieval Method")
        plt.ylabel("Average Score")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

    def _plot_query_score_distribution(self, df: pd.DataFrame, output_path: Path) -> None:
        plt.figure(figsize=(10, 5))
        plt.hist(df["overall_score"], bins=8, color="#8d5a97", edgecolor="white")
        plt.title("Distribution of Query Evaluation Scores")
        plt.xlabel("Overall Score")
        plt.ylabel("Number of Queries")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

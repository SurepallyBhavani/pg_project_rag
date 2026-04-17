from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import process_query, _display_retrieval_method
from src.evaluation.evaluation_runner import EvaluationRunner


def main() -> int:
    runner = EvaluationRunner(
        project_root=str(PROJECT_ROOT),
        process_query_fn=process_query,
        retrieval_method_fn=_display_retrieval_method,
    )
    output_dir = runner.run()
    print(f"Evaluation completed. Results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

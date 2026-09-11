"""Replay file-based runs into a real MLflow server (optional dep).

Usage:
    pip install mlflow
    mlflow server --host 127.0.0.1 --port 5000 &
    python bridge.py --runs ../../runs --experiment titanic-baseline

Each runs/<id>/run.json (see src/ai_roadmap/tracking.py) becomes one MLflow
run with params/metrics/tags preserved. Without `mlflow` installed the CLI
exits 2 with an install hint (offline CI stays green).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for candidate in [REPO, REPO / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
try:
    from ai_roadmap.tracking import RunLogger, summarize_runs
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))
    from ai_roadmap.tracking import RunLogger, summarize_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay runs/ into MLflow")
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--experiment", default="ai-roadmap")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args(argv)

    try:
        import mlflow
    except ImportError:
        print("`pip install mlflow` (and run `mlflow server`) to enable this bridge.")
        return 2
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    runs = summarize_runs(args.runs)
    if not runs:
        print(f"no runs found in {args.runs}")
        return 1
    mlflow.set_experiment(args.experiment)
    for summary in runs:
        logger = RunLogger(args.runs, run_id=str(summary["run_id"]))
        logger.params = dict(summary.get("params", {}))
        logger.metrics = {k: float(v) for k, v in summary.get("metrics", {}).items()}
        logger.tags = {k: str(v) for k, v in summary.get("tags", {}).items()}
        mlflow_id = logger.to_mlflow(args.experiment)
        print(f"{summary['run_id']} -> mlflow {mlflow_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

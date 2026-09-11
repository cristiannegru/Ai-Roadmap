"""Phase 7 — Offline experiment tracking (zero-dep MLflow bridge).

Maps to ``docs/mlops-roadmap.md`` (experiment tracking). :class:`RunLogger`
records params/metrics as one JSON file per run under ``runs/`` (git-ignored,
like ``mlruns/``), so learners get the track-compare-reproduce loop with no
server. :func:`to_mlflow` replays runs into a real MLflow tracking server
when ``mlflow`` is installed (lazy import — never a hard dependency).

Layout:
    runs/<run-id>/run.json   {"name", "params", "metrics", "tags", "status"}
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

__all__ = ["RunLogger", "summarize_runs"]


class RunLogger:
    """File-based run logger. Use as a context manager for auto-status."""

    def __init__(self, runs_dir: str | Path = "runs", run_id: str | None = None) -> None:
        self.runs_dir = Path(runs_dir)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.params: dict[str, object] = {}
        self.metrics: dict[str, float] = {}
        self.tags: dict[str, str] = {}
        self._status = "running"
        self._start = time.time()

    def log_param(self, key: str, value: object) -> None:
        """Record a hyperparameter / config value."""
        self.params[key] = value

    def log_metric(self, key: str, value: float) -> None:
        """Record a numeric metric (keeps the last value per key)."""
        self.metrics[key] = float(value)

    def set_tag(self, key: str, value: str) -> None:
        """Attach a string tag (e.g. owner, model family)."""
        self.tags[key] = value

    def finish(self, status: str = "finished") -> Path:
        """Persist the run JSON. Returns the file path."""
        self._status = status
        payload = {
            "run_id": self.run_id,
            "status": self._status,
            "duration_s": round(time.time() - self._start, 3),
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
        }
        out = self.runs_dir / self.run_id / "run.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return out

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.finish("failed" if exc is not None else "finished")

    def to_mlflow(self, experiment: str = "default") -> str:
        """Replay this run into MLflow. Requires `pip install mlflow`.

        Returns the MLflow run id. Raises ImportError with install hint when
        mlflow is missing (offline CI stays green).
        """
        try:
            import mlflow
        except ImportError as e:
            raise ImportError("`pip install mlflow` to enable MLflow export.") from e
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=self.run_id) as run:
            mlflow.log_params({k: str(v) for k, v in self.params.items()})
            mlflow.log_metrics(self.metrics)
            mlflow.set_tags(self.tags)
            return run.info.run_id


def summarize_runs(runs_dir: str | Path = "runs") -> list[dict[str, object]]:
    """Read all ``runs/*/run.json`` → sorted (by run_id) summary dicts."""
    root = Path(runs_dir)
    summaries: list[dict[str, object]] = []
    for path in sorted(root.glob("*/run.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue  # skip partially-written runs
    return summaries

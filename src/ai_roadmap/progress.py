"""Progress tracker — single source of truth for the 7 milestones.

Milestones mirror README.md > Progressive Milestones Tracker.
State is stored in `.progress.json` (git-ignored) so each learner
keeps their own checklist.

Usage:
    python -m ai_roadmap.progress --summary
    python -m ai_roadmap.progress --check 1 3
    python -m ai_roadmap.progress --uncheck 3
    python -m ai_roadmap.progress --reset
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STATE_FILE = Path(".progress.json")

MILESTONES: list[dict[str, Any]] = [
    {"id": 1, "title": "Python & Linear Algebra foundations", "phase": "Phase 1"},
    {
        "id": 2,
        "title": "Data wrangling script cleaning a custom CSV with Pandas",
        "phase": "Phase 2",
    },
    {"id": 3, "title": "Classification model using Scikit-Learn", "phase": "Phase 3"},
    {"id": 4, "title": "Custom CNN classifier using PyTorch", "phase": "Phase 4"},
    {"id": 5, "title": "PDF Q&A chatbot (RAG) using Streamlit", "phase": "Phase 5"},
    {"id": 6, "title": "Autonomous research agent group using CrewAI", "phase": "Phase 5"},
    {"id": 7, "title": "Dockerized API endpoint deployed to Render", "phase": "Phase 7"},
]


@dataclass
class Progress:
    done: list[int]

    @property
    def percent(self) -> float:
        return round(100 * len(set(self.done)) / len(MILESTONES), 1)

    def summary(self) -> str:
        lines = [
            "AI Roadmap — progress: " f"{len(set(self.done))}/{len(MILESTONES)} ({self.percent}%)"
        ]
        for m in MILESTONES:
            mark = "[x]" if m["id"] in self.done else "[ ]"
            lines.append(f"  {mark} Milestone {m['id']}: {m['title']} ({m['phase']})")
        return "\n".join(lines)


def load(state_file: Path = STATE_FILE) -> Progress:
    if not state_file.exists():
        return Progress(done=[])
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        done = [int(x) for x in data.get("done", [])]
        valid = {m["id"] for m in MILESTONES}
        return Progress(done=sorted(set(done) & valid))
    except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
        return Progress(done=[])


def save(progress: Progress, state_file: Path = STATE_FILE) -> None:
    state_file.write_text(json.dumps(asdict(progress), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Roadmap progress tracker")
    parser.add_argument("--summary", action="store_true", help="print checklist + percent")
    parser.add_argument("--check", nargs="+", type=int, default=[], help="mark milestone ids done")
    parser.add_argument(
        "--uncheck", nargs="+", type=int, default=[], help="mark milestone ids not done"
    )
    parser.add_argument("--reset", action="store_true", help="clear all progress")
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    args = parser.parse_args(argv)

    progress = load(args.state_file)
    valid = {m["id"] for m in MILESTONES}

    if args.reset:
        progress = Progress(done=[])
        save(progress, args.state_file)
        print(progress.summary())
        return 0

    for mid in args.check:
        if mid not in valid:
            print(f"Invalid milestone id: {mid}. Valid: {sorted(valid)}")
            return 2
        if mid not in progress.done:
            progress.done.append(mid)

    for mid in args.uncheck:
        progress.done = [d for d in progress.done if d != mid]

    if args.check or args.uncheck:
        progress.done = sorted(set(progress.done))
        save(progress, args.state_file)

    # Default action (no flags) is also summary — friendlier for beginners.
    print(progress.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

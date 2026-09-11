"""Project 06 — Autonomous research team (wraps the crew template, offline).

Category 4 (Expert) in docs/projects-roadmap.md.

Usage:
    python main.py --query "How does hybrid search work?"
    python main.py --query "..." --docs ./my_docs --out ./outputs/brief.md

Writes a cited Markdown brief. Live CrewAI swap documented in
templates/crewai_team/README.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

DEFAULT_DOCS = REPO / "data" / "samples" / "rag_docs"
DEFAULT_OUT = HERE / "outputs" / "brief.md"


def _load_crew():
    path = REPO / "templates" / "crewai_team" / "crew.py"
    spec = importlib.util.spec_from_file_location("crew_template", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["crew_template"] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autonomous research team (demo)")
    parser.add_argument("--query", required=True)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    crew = _load_crew()
    result = crew.run_crew(args.query, args.docs, k=args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(str(result["draft"]), encoding="utf-8")
    print(
        f"index: {result['index']} | revised: {result['revised']} " f"| issues: {result['issues']}"
    )
    print(f"saved brief: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

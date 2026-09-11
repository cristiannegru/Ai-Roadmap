"""Multi-agent research crew in demo mode (no API keys, offline).

Roles mirror CrewAI practice (researcher → writer → critic):
  1. Researcher retrieves top-k RAG passages for the query.
  2. Writer drafts an extractive brief with citations.
  3. Critic checks citations + coverage; on failure the writer revises once.

Usage:
    python crew.py --query "How does hybrid search work?" --docs ../../data/samples/rag_docs
    python crew.py --query "..." --docs ./my_docs --k 4 --show-hits

Live swap: install `crewai` + set `OPENAI_API_KEY`, then map each role to a
CrewAI Agent with the same inputs (query, hits). Retrieval stays identical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for candidate in [ROOT.parent.parent, ROOT.parent.parent / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
try:
    from ai_roadmap.rag_pipeline import build_index, load_docs_from_dir, retrieve
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(ROOT.parent.parent / "src"))
    from ai_roadmap.rag_pipeline import build_index, load_docs_from_dir, retrieve


def researcher(query: str, docs_dir: Path, k: int) -> tuple[list[dict[str, object]], str]:
    """Retrieve passages. Returns (hits, index_summary)."""
    docs = load_docs_from_dir(docs_dir)
    store, embedder, chunks = build_index(docs)
    hits = retrieve(store, embedder, query, k=k)
    return hits, f"{len(docs)} docs -> {len(chunks)} chunks"


def writer(query: str, hits: list[dict[str, object]]) -> str:
    """Draft a cited brief from retrieved hits (extractive, deterministic)."""
    if not hits:
        return f"# Brief: {query}\n\nNo relevant passages found."
    lines = [f"# Brief: {query}", ""]
    for i, hit in enumerate(hits, 1):
        lines.append(f"## {i}. `{hit['id']}` (score {hit['score']:.3f})")
        lines.append("")
        lines.append(str(hit["text"])[:500])
        lines.append("")
    lines.append(f"Sources: {', '.join(str(h['id']) for h in hits)}")
    return "\n".join(lines)


def critic(draft: str, hits: list[dict[str, object]]) -> list[str]:
    """Return a list of issues (empty = pass)."""
    issues: list[str] = []
    if not hits:
        return ["no retrieved passages"]
    cited = [str(h["id"]) for h in hits]
    if not any(cid in draft for cid in cited[:1]):
        issues.append("top passage not cited")
    if len(draft) < 200:
        issues.append("draft suspiciously short")
    return issues


def run_crew(query: str, docs_dir: Path, k: int = 3) -> dict[str, object]:
    """Run researcher → writer → critic (one revision). Returns the transcript."""
    hits, summary = researcher(query, docs_dir, k)
    draft = writer(query, hits)
    issues = critic(draft, hits)
    revised = False
    if issues and hits:
        # One revision: prepend the top passage verbatim so nothing is lost.
        draft = f"> {hits[0]['text']}\n\n" + draft
        revised = True
        issues = critic(draft, hits)
    return {
        "query": query,
        "index": summary,
        "draft": draft,
        "issues": issues,
        "revised": revised,
        "citations": [h["id"] for h in hits],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo research crew (offline)")
    parser.add_argument("--query", required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--show-hits", action="store_true")
    args = parser.parse_args(argv)

    result = run_crew(args.query, args.docs, k=args.k)
    print(f"index: {result['index']}")
    print(f"revised: {result['revised']} | residual issues: {result['issues']}")
    print()
    print(result["draft"])
    if args.show_hits:
        print()
        print(f"citations: {result['citations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

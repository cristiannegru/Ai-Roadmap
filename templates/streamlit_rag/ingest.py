"""Ingest docs into a SimpleVectorStore index (offline, no API keys).

Usage:
    python ingest.py --docs ../../data/samples/rag_docs --index ./index/rag
    python ingest.py --docs ./my_docs --index ./index/rag --chunk-size 150 --overlap 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for candidate in [ROOT.parent.parent, ROOT.parent.parent / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
# Support both `pip install -e .` (package importable) and raw checkout (src/ layout).
try:
    from ai_roadmap.rag_pipeline import build_index, load_docs_from_dir
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(ROOT.parent.parent / "src"))
    from ai_roadmap.rag_pipeline import build_index, load_docs_from_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest .md/.txt docs into a vector index")
    parser.add_argument("--docs", type=Path, required=True, help="folder with *.md/*.txt files")
    parser.add_argument("--index", type=Path, required=True, help="output base path (no ext)")
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    docs = load_docs_from_dir(args.docs)
    store, embedder, chunks = build_index(
        docs, chunk_size=args.chunk_size, overlap=args.overlap, dim=args.dim, seed=args.seed
    )
    base = store.save(args.index)
    embedder.save(Path(str(args.index) + ".embedder.npz"))
    print(f"docs: {len(docs)} -> chunks: {len(chunks)} -> dim: {store.dim}")
    print(f"saved: {base.with_suffix('.npz')} + {base.with_suffix('.json')}")
    print(f"saved: {args.index}.embedder.npz (IDF weights — app.py loads it automatically)")
    print("NOTE: the query side must use these exact IDF weights, not a fresh embedder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

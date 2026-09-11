"""Project 05 — PDF/RAG Q&A chatbot CLI (reuses Chunk-4 pipeline, offline).

Category 3 (Advanced) in docs/projects-roadmap.md.

Usage:
    python main.py ingest                              # index sample rag_docs
    python main.py ask --query "What is hybrid search?"
    python main.py ingest --docs ./my_docs --index ./outputs/custom
    python main.py ask --query "..." --index ./outputs/custom --k 4

PDFs: install `pypdf` to include *.pdf files in --docs (else skipped with
a warning). For a chat UI serving the same index, see templates/streamlit_rag/.
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
    from ai_roadmap.embeddings import HashingEmbedder
    from ai_roadmap.rag_pipeline import (
        answer_query,
        build_index,
        load_docs_from_dir,
    )
    from ai_roadmap.vector_store import SimpleVectorStore
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))
    from ai_roadmap.embeddings import HashingEmbedder
    from ai_roadmap.rag_pipeline import (
        answer_query,
        build_index,
        load_docs_from_dir,
    )
    from ai_roadmap.vector_store import SimpleVectorStore

DEFAULT_DOCS = REPO / "data" / "samples" / "rag_docs"
DEFAULT_INDEX = HERE / "outputs" / "rag_index"


def load_docs_with_optional_pdfs(docs_dir: Path) -> list[dict[str, str]]:
    """Load .md/.txt always; include .pdf only if `pypdf` is installed."""
    docs = load_docs_from_dir(docs_dir)
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if pdfs:
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"warning: {len(pdfs)} PDF(s) skipped — `pip install pypdf` to include.")
            return docs
        for pdf in pdfs:
            text = "\n".join((page.extract_text() or "") for page in PdfReader(str(pdf)).pages)
            if text.strip():
                docs.append({"id": pdf.stem, "text": text})
    return docs


def cmd_ingest(docs_dir: Path, index: Path) -> int:
    docs = load_docs_with_optional_pdfs(docs_dir)
    store, embedder, chunks = build_index(docs)
    base = store.save(index)
    embedder.save(Path(str(index) + ".embedder.npz"))
    print(f"docs: {len(docs)} -> chunks: {len(chunks)}")
    print(f"saved: {base.with_suffix('.npz').name} + {base.with_suffix('.json').name}")
    return 0


def cmd_ask(query: str, index: Path, k: int) -> int:
    try:
        store = SimpleVectorStore.load(index)
        embedder = HashingEmbedder.load(str(index) + ".embedder.npz")
    except FileNotFoundError:
        print(f"index missing at {index}. Run `python main.py ingest` first.")
        return 2
    result = answer_query(store, embedder, query, k=k)
    print(result["answer"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG Q&A chatbot CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    p_ing.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("--query", required=True)
    p_ask.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p_ask.add_argument("--k", type=int, default=3)
    args = parser.parse_args(argv)
    if args.cmd == "ingest":
        return cmd_ingest(args.docs, args.index)
    return cmd_ask(args.query, args.index, args.k)


if __name__ == "__main__":
    raise SystemExit(main())

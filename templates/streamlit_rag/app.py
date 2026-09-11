"""Streamlit RAG chat over a prebuilt SimpleVectorStore index (offline demo).

Usage:
    python ingest.py --docs ../../data/samples/rag_docs --index ./index/rag
    streamlit run app.py -- --index ./index/rag --seed 42

Swap to production: replace `answer_query` internals with an OpenAI/Anthropic
call over the same `hits` — retrieval code is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for candidate in [ROOT.parent.parent, ROOT.parent.parent / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
try:
    from ai_roadmap.embeddings import HashingEmbedder
    from ai_roadmap.rag_pipeline import answer_query
    from ai_roadmap.vector_store import SimpleVectorStore
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(ROOT.parent.parent / "src"))
    from ai_roadmap.embeddings import HashingEmbedder
    from ai_roadmap.rag_pipeline import answer_query
    from ai_roadmap.vector_store import SimpleVectorStore


def _parse_args() -> tuple[str, int]:
    # Streamlit forwards unknown flags after `--`; argparse would choke on its
    # own flags, so parse only our two options manually.
    index, seed = "./index/rag", 42
    args = sys.argv[1:]
    for i, token in enumerate(args):
        if token == "--index" and i + 1 < len(args):
            index = args[i + 1]
        if token == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
    return index, seed


def main() -> None:
    import streamlit as st

    index_base, _seed = _parse_args()  # seed kept for CLI compat; IDF comes from file

    st.set_page_config(page_title="AI Roadmap — RAG demo", layout="centered")
    st.title("📄 RAG Q&A (offline demo)")
    st.caption("Extractive answers with citations. No API key needed.")

    try:
        store = SimpleVectorStore.load(index_base)
    except FileNotFoundError:
        st.error(
            f"Index not found at `{index_base}`. Run first:\n\n"
            f"`python ingest.py --docs ../../data/samples/rag_docs --index {index_base}`"
        )
        st.stop()
    try:
        embedder = HashingEmbedder.load(f"{index_base}.embedder.npz")
    except FileNotFoundError:
        st.warning("Embedder file missing — re-run `ingest.py` to regenerate IDF weights.")
        st.stop()
    k = st.sidebar.slider("Top-k passages", 1, 5, 3)

    if "history" not in st.session_state:
        st.session_state.history = []
    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    query = st.chat_input("Ask about chunking, agents, CNNs, overfitting…")
    if query:
        st.session_state.history.append(("user", query))
        st.chat_message("user").write(query)
        result = answer_query(store, embedder, query, k=k)
        st.session_state.history.append(("assistant", str(result["answer"])))
        st.chat_message("assistant").write(str(result["answer"]))
        with st.expander("Retrieved passages"):
            for hit in result["hits"]:  # type: ignore[union-attr]
                st.markdown(f"**`{hit['id']}`** (score {hit['score']:.3f})")
                st.caption(str(hit["text"])[:400])


if __name__ == "__main__":
    main()

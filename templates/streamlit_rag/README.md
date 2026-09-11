# Streamlit RAG template (offline demo, Chunk 4)

Extractive Q&A over your own `.md/.txt` docs. Zero API keys.

```bash
cd templates/streamlit_rag
pip install -r requirements.txt
python ingest.py --docs ../../data/samples/rag_docs --index ./index/rag
streamlit run app.py -- --index ./index/rag --seed 42
```

Ask: *“What is hybrid search?”* → answer quotes `rag_systems` chunk + citations.

## Files

- `ingest.py` — chunk + embed + save index (`.npz` + `.json`). `--seed`/`--dim` must match `app.py`.
- `app.py` — chat UI, top-k slider, expandable retrieved passages.

## Go production

1. Replace `HashingEmbedder` with OpenAI/Cohere/`sentence-transformers` embeddings (same `(n, dim)` matrix shape).
2. Replace `SimpleVectorStore` with Chroma/Qdrant (same `add`/`search` shapes).
3. Replace `generate_answer` with an LLM call over the retrieved `hits` (keep citations).

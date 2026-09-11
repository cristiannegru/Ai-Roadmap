# Quiz — Phase 5: Generative AI, RAG & Agents

Self-test for `docs/rag-roadmap.md`, `docs/ai-agents-roadmap.md`, `docs/generative-ai-roadmap.md`
and notebook `05`. Code: `text_chunking`, `embeddings`, `vector_store`, `rag_pipeline`.

## Q1 — Why chunk *with overlap* instead of hard cuts?
<details><summary>Answer</summary>

Hard cuts can split a key sentence across chunks so neither matches a query. Overlap (e.g. 20
units) preserves local context at boundaries. Notebook 05 §1 shows the difference.
</details>

## Q2 — What does `HashingEmbedder.fit` compute, and why does retrieval fail without it?
<details><summary>Answer</summary>

Smoothed IDF weights per hash bin (`log((1+N)/(1+df)) + 1`). Without it, stopwords and hash
collisions drown out distinctive terms — our tests caught `ml_basics` outranking `rag_systems`
for a hybrid-search query (the Chunk-4 retrieval bug).
</details>

## Q3 — Why must the query side reuse the index-time IDF weights?
<details><summary>Answer</summary>

Index and query vectors must live in the same weighted space; a fresh embedder uses IDF = 1.0
and silently degrades ranking. `ingest.py` saves `.embedder.npz` and `app.py` loads it.
</details>

## Q4 — Define `hit@k` and report this repo's score.
<details><summary>Answer</summary>

1.0 if any top-k hit's doc matches the relevant doc, else 0.0 (compared on `doc:::` prefixes).
Notebook 05 asserts hit@2 ≥ 0.75 over 4 topic probes; actual is 1.0 with top-1 exact.
</details>

## Q5 — Extractive vs generative answers: what ships here, and what is the swap?
<details><summary>Answer</summary>

Extractive: quote the top chunk + `Sources:` citations — offline, deterministic, testable.
Production: format the same `hits` into an LLM prompt and return its completion with citations.
Retrieval code is unchanged either way.
</details>

## Q6 — Sparse vs dense vs hybrid search + reranking?
<details><summary>Answer</summary>

Sparse (BM25/TF-IDF): exact keyword match, great for codes/IDs. Dense (embeddings): semantic
match across wording. Hybrid runs both, then a cross-encoder reranker re-scores the top
candidates for precision.
</details>

## Q7 — Walk the ReAct loop.
<details><summary>Answer</summary>

Thought → Action (tool call) → Observation → repeat until done. The LLM interleaves reasoning
traces with tools (search, calculator, DB), using each observation to plan the next step.
</details>

## Q8 — Researcher → Writer → Critic: what does each role do in `crew.py`?
<details><summary>Answer</summary>

Researcher retrieves top-k RAG passages. Writer drafts a cited brief. Critic checks citation +
coverage and triggers one verbatim-prepend revision on failure. Mirrors CrewAI role delegation.
</details>

## Q9 — What three tools does the MCP-concept server expose, and what guards them?
<details><summary>Answer</summary>

`list_tables`, `describe_table`, `query_db` over SQLite stdio JSON. Guards: SELECT/WITH only,
single statement, 100-row cap, identifier-checked table names — destructive SQL is rejected.
</details>

## Q10 — Chunk size tradeoff in one paragraph?
<details><summary>Answer</summary>

Small chunks retrieve precisely but may lack the context needed to answer; large chunks keep
context but dilute similarity and waste prompt budget. 120 words + 20 overlap is this repo's
default; tune with hit@k on your own eval set.
</details>

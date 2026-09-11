# Research crew template (demo mode, Chunk 4)

Researcher → Writer → Critic over your docs. No API key.

```bash
cd templates/crewai_team
python crew.py --query "How does hybrid search work?" --docs ../../data/samples/rag_docs
python crew.py --query "What is ReAct?" --docs ../../data/samples/rag_docs --k 4 --show-hits
```

Output: cited brief + `revised:` flag + residual critic issues.

## Go live (CrewAI)

1. `pip install crewai openai` + `export OPENAI_API_KEY=...`
2. Map roles 1:1 — Researcher gets a retrieval tool wrapping `retrieve()`,
   Writer gets the hits, Critic keeps the same checks.
3. Retrieval code (`build_index`/`retrieve`) stays unchanged.

# 10% → 100% Upgrade Plan — ✅ COMPLETE (v2.0.0, 90 tests, all green)

Current state (v1.0.0, ~10%): 14 docs guides + README + assets. No code, no tests, no CI, no runnable projects.

Target (v2.0.0, 100%): docs + runnable code + projects + templates + MLOps + learning UX + community automation.

---

## Chunk map (each chunk = one reviewable upgrade)

| Chunk | % | Focus | Key outputs | Done when |
|-------|---|-------|-------------|-----------|
| **1. Foundation** | 10→25% | Repo hygiene + CI + community | `.gitignore`, `pyproject.toml`, `requirements*.txt`, `Makefile`, `pre-commit`, `.github/` (CI, links cron, 3 issue forms, PR template, CODEOWNERS), `scripts/validate_repo.py`, `scripts/check_links.py`, `src/ai_roadmap/progress.py`, `tests/`, `data/samples/`, `notebooks/`, `projects/`, `templates/` scaffolds | `make validate && make test && make links` all green |
| **2. Foundations code** | 25→40% | Phase 1–2 runnable | `src/ai_roadmap/{python_basics,math_utils,data_wrangling}.py`, `data/samples/*.csv`, `notebooks/01_*.ipynb`, `notebooks/02_*.ipynb`, `tests/test_*` | Every Phase 1–2 docs code block runs via `pytest` |
| **3. ML + DL code** | 40→55% | Phase 3–4 runnable | `src/ai_roadmap/{features,metrics,torch_utils}.py`, `notebooks/03_*.ipynb`, `notebooks/04_*.ipynb`, sklearn + PyTorch starters | `pytest` covers metrics/features; CNN trains on CPU <5 min |
| **4. GenAI/RAG/Agents** | 55→70% | Phase 5 templates | `templates/{streamlit_rag,crewai_team,mcp_server}/`, `notebooks/05_rag_minimal.ipynb`, eval harness | RAG chatbot runs locally with fake-embeddings test mode (no API key) |
| **5. Projects portfolio** | 70→85% | 7 runnable projects | `projects/01_*` … `projects/07_*` each with README + requirements + tests | Each project `pytest` + `python main.py --help` passes |
| **6. MLOps + Deploy** | 85→95% | Phase 7 production | `templates/fastapi_ml/` + `Dockerfile` + `docker-compose.yml`, `render.yaml`, MLflow config, load test | `docker build` + `pytest` + deployed URL in docs |
| **7. Learning UX + Release** | 95→100% | Quizzes, site, polish | `docs/quizzes/*.md`, flashcards, `mkdocs.yml` site, badges, `CHANGELOG v2.0.0`, full `make validate` | New learner can go 0→deploy using only this repo |

---

## Conventions (apply to every chunk)

- **Conventional Commits**: `feat:`, `fix:`, `docs:`, `chore:`
- **Every code file**: docstring + type hints + test (or explicit `--no-test` reason in PR)
- **Every project/template**: `README.md` (what/why/how to run) + pinned `requirements.txt`
- **No secrets**: only `.env.example`; CI fails if `.env` is tracked
- **Offline-first**: core Chunks 1–3 run with zero API keys; GenAI chunks include `--demo` fake mode
- **Verify before merge**: `make lint && make test && make validate`

## How to continue

Say **“next chunk”** and I implement Chunk 2 in full detail. Each chunk ends with a verification log.

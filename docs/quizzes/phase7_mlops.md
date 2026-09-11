# Quiz — Phase 7: Deployment & MLOps

Self-test for `docs/mlops-roadmap.md`, `templates/fastapi_ml/`, `docker-compose.yml`,
`render.yaml`, `scripts/load_test.py`. Code: `src/ai_roadmap/tracking.py`.

## Q1 — `RunLogger` vs MLflow: what is tracked where?
<details><summary>Answer</summary>

`RunLogger` writes `runs/<id>/run.json` (params, metrics, tags, status, duration) with zero
deps. `bridge.py` replays those files into a live MLflow server when comparison UI is needed.
</details>

## Q2 — Why does `app.py` load the artefact in `lifespan`, not on first request?
<details><summary>Answer</summary>

Fail-fast: a missing/corrupt artefact crashes the container at startup (visible in deploy logs),
not on the first user request at 3am. Health checks then reflect true readiness.
</details>

## Q3 — Train-time vs serve-time: why does the template save feature lists in the bundle?
<details><summary>Answer</summary>

`/info` exposes them for debugging, and future validators can reject schema drift. The bundle
(pipeline + numeric + categorical + target) is self-describing — no tribal knowledge.
</details>

## Q4 — Name three Dockerfile hardening choices in this repo.
<details><summary>Answer</summary>

`python:3.11-slim` base, non-root `appuser`, `HEALTHCHECK`, `PIP_NO_CACHE_DIR=1`,
`PYTHONDONTWRITEBYTECODE=1`, baked artefact (no download at boot), root `.dockerignore`.
</details>

## Q5 — What do `docker-compose.yml`'s two services expose, and what is the volume for?
<details><summary>Answer</summary>

`api` :8000 (Titanic) and `rag` :8501 (Streamlit). The `rag-index` volume overrides the baked
sample index with your own prebuilt one (`./my-index:/app/index:ro`).
</details>

## Q6 — Render Blueprint in 3 steps?
<details><summary>Answer</summary>

Push to GitHub → Dashboard → New → Blueprint → select repo (reads `render.yaml`: docker runtime,
`dockerfilePath`, `healthCheckPath: /health`, `PORT`). Starter plan suffices for the demo.
</details>

## Q7 — Load test numbers from Chunk 6 verification: what do p50/p95 tell you?
<details><summary>Answer</summary>

50/50 × 200, failures 0, mean 29.8ms, p50 13.8ms, p95 156ms (first requests include training +
cold start). p95 — not the mean — is the latency SLO to watch; gate deploys on failures > 0.
</details>

## Q8 — When does the Docker CI workflow run, and what does the smoke test do?
<details><summary>Answer</summary>

On PRs touching the API/template/RAG-docker paths (or manual dispatch). Builds all 3 images,
boots the API container, waits on `/health`, POSTs a real prediction, then removes the container.
</details>

## Q9 — `.dockerignore` vs `.gitignore`: why both?
<details><summary>Answer</summary>

`.gitignore` controls version control (CSVs force-added, secrets never). `.dockerignore`
controls build context (keeps images small and secret-free: `.git`, outputs, runs, caches excluded).
</details>

## Q10 — Milestone 7 acceptance: what must a learner demonstrate?
<details><summary>Answer</summary>

`docker compose up --build` serving both UIs locally, a green `load_test.py` run, and (stretch)
a live Render URL from the Blueprint. See the roadmap's Milestone 7 checkbox.
</details>

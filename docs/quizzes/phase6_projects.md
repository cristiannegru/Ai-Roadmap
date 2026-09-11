# Quiz — Phase 6: Projects Portfolio

Self-test for `docs/projects-roadmap.md` and `projects/`. Run each command before answering.

## Q1 — Which project maps to which tier (beginner → MLOps bridge)?
<details><summary>Answer</summary>

01 spam + 02 housing (beginner) → 03 resume + 04 vision (intermediate) → 05 RAG chatbot
(advanced) → 06 research crew (expert) → 07 FastAPI (MLOps bridge). See `projects/README.md`.
</details>

## Q2 — Spam dataset: how is it built, and why is F1 1.0 unsurprising?
<details><summary>Answer</summary>

Seeded templates (6 spam / 6 ham patterns with slot variation). Templates are linearly separable
by design (prize/urgent vs lunch/report vocab) — the project teaches plumbing, not SOTA accuracy.
</details>

## Q3 — Resume verdict logic: when SHORTLIST vs REVIEW?
<details><summary>Answer</summary>

`SHORTLIST` iff skill score ≥ 60% AND years ≥ `--min-years`; else `REVIEW`. The sample resume
shortlists for python/pytorch/aws at 3 years and drops to REVIEW at 10.
</details>

## Q4 — Vision: what are the 3 objects, and what would break detection?
<details><summary>Answer</summary>

Two bright rectangles + one disc on dark background (±3px tolerance asserted). Raising the
threshold above object brightness, shrinking below `min_area`, or heavy noise merges/splits blobs.
</details>

## Q5 — 05 chatbot: how do you index your own docs, and where does the chat UI come from?
<details><summary>Answer</summary>

`python main.py ingest --docs ./my_docs --index ./outputs/custom` (+ `pip install pypdf` for PDFs).
`templates/streamlit_rag/` serves any such index as a chat app.
</details>

## Q6 — 06 crew: what artefact is written, and what proves the critic ran?
<details><summary>Answer</summary>

`outputs/brief.md` (cited Markdown). CLI prints `revised:` + residual `issues:` — the transcript
of the critic pass, empty on clean runs.
</details>

## Q7 — 07 API: which endpoints exist, and what status does bad input return?
<details><summary>Answer</summary>

`GET /health`, `POST /predict` (plus `/docs` Swagger). Pydantic rejects bad payloads with 422
(verified: `pclass: 9` → 422); feature mismatches map to 400, never 500.
</details>

## Q8 — Why does 07 train at startup instead of shipping a joblib?
<details><summary>Answer</summary>

Deterministic ~1s training removes artefact staleness (no out-of-sync weights) and keeps the
project dependency-free of binaries. The template (`fastapi_ml`) shows the artefact variant.
</details>

## Q9 — Which two Chunk-5 bugs came from tests, and what locks each fix?
<details><summary>Answer</summary>

(1) `passenger_id` as a feature → `exclude=` + predict-path test. (2) Year-range phone false
positive → ≥10-digit + separator rule + exact-phone assertion.
</details>

## Q10 — Port a project to production in three steps (any project)?
<details><summary>Answer</summary>

1. Swap the data source (synthetic → real CSV/PDF читайте READMEs' "Go production").
2. Swap the model/component (TF-IDF → transformer, threshold → YOLO, extractive → LLM).
3. Serve via the 07/template pattern (FastAPI + Docker + compose). Shapes stay stable by design.
</details>

# Changelog

All notable changes to the **AI Roadmap** project will be documented in this file. This project adheres to Semantic Versioning (`vMAJOR.MINOR.PATCH`).

---

## [2.0.1] - 2026-09-11 — CI hotfix (found by first GitHub run)
### Fixed
- **CI install failure**: `requirements-dev.txt` listed `lychee` and `markdownlint-cli`, which are not pip packages (consumed via GH Action / npx) — `pip install` failed in ~8s and fail-fast cancelled the matrix. Removed with pointers to the real sources. Reproduced + verified in a fresh venv.
- **Validator false positives**: the large-file walk flagged local caches (`.mypy_cache/`, …). Now skips git-ignored artefact dirs; covered by `tests/test_validate.py`.
### Added
- `tests/test_validate.py`: validator passes on the repo; cache dirs skipped.

## [2.0.0] - 2026-09-11 — Chunk 7: Learning UX + release (95% → 100%)
### Added
- **Quizzes** (`docs/quizzes/phase{1..7}_*.md`): 70 Q&A with `<details>` answers, each tied to repo modules/notebooks/verified numbers (F1 1.0, R² 0.996, hit@2 1.0, p50 13.8ms).
- **Flashcards** (`docs/flashcards.md`): 40 rapid-fire cards across phases + career, linking back to quizzes and the interview guide.
- **Docs site** (`mkdocs.yml` + `docs/index.md`, Material theme, strict-clean build): tabbed nav over all guides, quizzes, flashcards, and career playbooks. Preview: `pip install mkdocs-material && mkdocs serve`.
- **README routes**: 5-minute offline quickstart, role-based route table (student/dev/freelancer/founder/job-seeker), milestones wired to `progress --check`, self-test section, repository map.
- **Content tests** (`tests/test_docs.py`, 3 tests): 10 answered Qs per quiz, ≥40 flashcards, mkdocs nav ↔ files match. Suite: 90 passed.
- **Validator v2**: now enforces 18 files, 10 dirs, 14 guides, 9 quizzes/site docs (incl. per-quiz question/answer counts).
### Changed
- Version 2.0.0 — the 10% → 100% upgrade is complete. See `ROADMAP_TO_100.md` for the chunk log.

## [1.6.0] - 2026-09-11 — Chunk 6: MLOps & deployment (85% → 95%)
### Added
- **Experiment tracking** (`src/ai_roadmap/tracking.py`, zero deps): `RunLogger` (context manager, failed/finished status, `runs/<id>/run.json`), `summarize_runs`, `to_mlflow` replay (lazy import, clean hint when missing).
- **API template** (`templates/fastapi_ml/`): env-driven `settings.py`, `train.py` (pipeline + run log + joblib bundle with feature metadata), `app.py` (lifespan fail-fast, `/health`, `/info`, `/predict` with 400 mapping), `Dockerfile` (bakes training into the image), template-local `test_app.py` (passes).
- **RAG image** (`templates/streamlit_rag/Dockerfile`): bakes the sample index, `/app/index` volume override.
- **Local + cloud deploy**: root `docker-compose.yml` (api :8000 + rag :8501, healthchecks), `render.yaml` Blueprint, root `.dockerignore`, `.github/workflows/docker.yml` (builds all 3 images + container smoke test on relevant PRs).
- **Load testing** (`scripts/load_test.py`, stdlib): threaded POST sweep with mean/p50/p95 + status counts, exit-1 gate. Live-verified: 50/50 × 200, p50 13.8ms.
- **MLflow bridge** (`templates/mlflow_tracking/`): replays `runs/` into a live server; exits 2 with hint when mlflow absent.
- **Tests**: `test_tracking.py` (4), `test_mlops.py` (4: compose/render YAML contracts, load-test gate, bridge guard). Suite: 87 passed.
### Changed
- Version 1.6.0. `docs/mlops-roadmap.md` gains a runnable section; `templates/README.md` marks MLOps items ✅. `make lint`/CI cover `templates/` + `projects/`.

## [1.5.0] - 2026-09-11 — Chunk 5: Projects portfolio (70% → 85%)
### Added
- **7 runnable projects** (each: `README.md` + `requirements.txt` + CLI + tests in `tests/test_projects.py`, 9 tests):
  - `01_spam_classifier` — TF-IDF + Naive Bayes on seeded synthetic SMS (F1 1.0), train/`--predict` CLI, joblib artefact.
  - `02_housing_regression` — Ridge via shared pipeline on `housing_sample.csv` (R² 0.996), feature-flag predict CLI.
  - `03_resume_parser` — stdlib regex + keyword skills matcher with SHORTLIST/REVIEW verdict (sample resume included).
  - `04_vision_detector` — NumPy/PIL threshold → flood-fill components → boxes (3/3 synthetic objects ±3px), annotated PNG.
  - `05_pdf_rag_chatbot` — `ingest`/`ask` CLI over Chunk-4 pipeline (+ lazy `pypdf` support, Streamlit serves its index).
  - `06_crewai_research_team` — wraps the crew template, writes cited `brief.md`.
  - `07_fastapi_titanic_api` — FastAPI + Pydantic (400s on bad input), `/health` + `/predict`, cached training, slim non-root `Dockerfile` + `HEALTHCHECK` (context = repo root).
- **Tooling scope**: `make lint`, `black`, and CI now cover `templates/` + `projects/`; `httpx` added to dev deps for API tests (E402 exempted for intentional `sys.path` bootstraps via per-file-ignores).
### Fixed
- **ID-as-feature leak**: `passenger_id` flowed into the titanic pipeline (07 crashed on predict with missing-column `ValueError`; notebook 03 silently learned from row ids). Both now `exclude=["passenger_id"]`; notebook 03 re-executed (15 cells OK).
- **Phone false positive**: resume parser caught `2020-2022` as a phone number — now requires ≥10 digits + real separators. Locked by assertion.
### Changed
- Version 1.5.0. `projects/README.md` is now the runnable index; `docs/projects-roadmap.md` links each tier to its folder.

## [1.4.0] - 2026-09-11 — Chunk 4: GenAI/RAG/Agents (55% → 70%)
### Added
- **RAG core** (NumPy + stdlib, zero API keys): `text_chunking.py` (`Chunk` with provenance, words/chars units, overlap), `embeddings.py` (`HashingEmbedder`: hashed BOW + TF-IDF `fit`, L2 rows, `save/load` for query-side IDF), `vector_store.py` (`SimpleVectorStore`: cosine search, duplicate/dim guards, `.npz`+`.json` persistence), `rag_pipeline.py` (`load_docs_from_dir`, `build_index`, `retrieve`, extractive `generate_answer` with citations, `answer_query`, `hit_at_k` eval).
- **Knowledge base**: `data/samples/rag_docs/` — 4 topic-separated docs (ml_basics, deep_learning, rag_systems, ai_agents) giving top-1 retrieval on all 4 probe queries.
- **Templates**: `streamlit_rag/` (ingest CLI + chat app with top-k slider, IDF embedder file shipped alongside index), `crewai_team/` (researcher→writer→critic demo CLI with one revision + live CrewAI swap guide), `mcp_server/` (stdlib stdio JSON server: list_tables/describe_table/query_db with SELECT-only + single-statement guards, sample `notes` table).
- **Notebook**: `05_rag_minimal.ipynb` (17 cells: overlap demo → embedding ranking → index → 4 topic queries → cited answers → hit@2 eval ≥0.75 → saved index). Executed in verification.
- **Tests**: `test_text_chunking.py` (5), `test_rag.py` (6 incl. IDF/persistence/empty-store), `test_templates.py` (4 incl. crew citation + MCP guards + stdio smoke). Suite: 70 passed (was 54). Coverage 91% overall, 93–98% on new modules.
### Fixed
- **Retrieval quality bug** (found by tests): plain hashed counts at dim=256 ranked `ml_basics` above `rag_systems` for a hybrid-search query (stopword + collision noise). Fixed with TF-IDF bin weighting (`fit` on index chunks) + dim 1024 + `reranking` vocabulary alignment in sample docs. All 4 probes now top-1 with margins; locked by exact top-1 assertions.
- **Embedder/index skew trap**: query side must reuse index-time IDF — `ingest.py` now saves `.embedder.npz` and `app.py` loads it (warns + stops if missing) instead of rebuilding a fresh embedder.
### Changed
- Version 1.4.0. `docs/rag-roadmap.md`, `docs/ai-agents-roadmap.md`, `notebooks/README.md`, `templates/README.md` gain runnable-code sections.

## [1.3.0] - 2026-09-11 — Chunk 3: ML + DL code (40% → 55%)
### Added
- **Phase 3 pipelines** (`src/ai_roadmap/features.py`): `infer_feature_types`, `build_preprocessor` (median-impute + scale, mode-impute + one-hot with `handle_unknown="ignore"`), `make_classification_pipeline` (LogReg), `make_regression_pipeline` (Ridge), `train_test_split_df` (stratified with tiny-data fallback).
- **Phase 3 metrics** (`src/ai_roadmap/metrics.py`, NumPy from-scratch): `mae/mse/rmse/r2_score`, `accuracy/confusion_matrix/precision/recall/f1` (zero-division safe), `roc_auc` (sklearn ranking integral), `classification_report`. All assert sklearn agreement in tests.
- **Phase 4 PyTorch** (`src/ai_roadmap/torch_utils.py`): `set_seed`, `get_device` (cuda>mps>cpu), `SimpleMLP`, `SimpleCNN` (AdaptivePool, size-agnostic 16×16/28×28, ~25k params), `make_synthetic_digits` (quadrant patterns, no download, deterministic), `make_loaders`, `train_classifier` (Adam+CE, history dict), `evaluate_accuracy`, `count_parameters`, `save/load_model` (device-aware).
- **Notebooks**: `03_sklearn_end_to_end.ipynb` (15 cells: clean→split→LogReg vs RandomForest→confusion matrix→`titanic_logreg.joblib`) + `04_pytorch_cnn.ipynb` (13 cells: synthetic digits→MLP vs CNN→curves→`cnn_digits.pt`, asserts >80% val acc). Both executed in verification.
- **Tests**: `test_features.py` (5), `test_metrics.py` (7), `test_torch_utils.py` (6). Suite: 54 passed (was 36). Coverage 89% overall, 90–97% on new modules.
### Fixed
- **MPS device bug** (found by notebook execution): `load_model` left weights on CPU while `evaluate_accuracy` moved inputs to MPS → `RuntimeError: Input type (MPSFloatType) and weight type (torch.FloatTensor)`. Now `load_model` moves the model to the target device and `evaluate_accuracy` defensively calls `model.to(device)`. Locked by an extended save/load roundtrip test.
### Changed
- `torch>=2.3` promoted from optional to core dependency (`requirements.txt` + `pyproject.toml`). `docs/machine-learning-roadmap.md` + `docs/deep-learning-roadmap.md` gain “Runnable code” sections. Version 1.3.0.

## [1.2.0] - 2026-09-11 — Chunk 2: Foundations code (25% → 40%)
### Added
- **Phase 1 Python** (`src/ai_roadmap/python_basics.py`, stdlib-only): `safe_divide`, `chunk_list` (RAG preview), `flatten_nested`, `count_words` (NLP preview), `read/write_text_file`, `Experiment` OOP class (MLflow preview). Doctests included.
- **Phase 1 Math** (`src/ai_roadmap/math_utils.py`, NumPy): `dot_product`, `cosine_similarity`, `euclidean_distance`, `matrix_multiply`, `eigen_decomposition`, `solve_linear_system`, `describe`, `bayes_theorem`, `sigmoid`, `relu`, `softmax`, `linear_regression_closed_form`.
- **Phase 2 Data** (`src/ai_roadmap/data_wrangling.py`, Pandas/Seaborn): `load_csv`, `basic_info`, `impute_missing`, `encode_categoricals`, `scale_numeric`, `remove_duplicates`, `detect_outliers_iqr`, `plot_numeric_hist`, `plot_correlation_heatmap`, `clean_dataframe` one-call Milestone-2 pipeline with report dict.
- **Datasets**: `data/samples/titanic_sample.csv` (61 rows, 6 missing ages, 1 duplicate) + `housing_sample.csv` (81 rows, 5+7 missing, 1 duplicate). Synthetic, `seed(42)`, <3KB each.
- **Notebooks**: `notebooks/01_python_foundations.ipynb` (13 cells) + `notebooks/02_numpy_pandas.ipynb` (15 cells, end-to-end Milestone-2 solution with saved PNGs to `outputs/`). Both Colab-badged, both executed top-to-bottom in verification.
- **Tests**: `test_python_basics.py` (9 tests), `test_math_utils.py` (12 tests), `test_data_wrangling.py` (9 tests). Total suite: 36 passed (was 6). Coverage 86% overall, 91–100% on new modules.
### Changed
- `data/samples/README.md` documents exact row/missing/duplicate contracts. `docs/machine-learning-roadmap.md` gains a “Runnable code in this repo” section. Version bumped to 1.2.0.

## [1.1.0] - 2026-09-10 — Chunk 1: Foundation (10% → 25%)
### Added
- **Packaging & env**: `pyproject.toml` (src layout, ruff/black/mypy/pytest config), `requirements.txt`, `requirements-dev.txt`, `.env.example`, `Makefile` (`install, lint, test, validate, links, progress, clean`).
- **Hygiene**: `.gitignore` (Python, venv, data/models, learner `.progress.json`), `.pre-commit-config.yaml`, `.markdownlint.json`.
- **CI & community**: `.github/workflows/ci.yml` (Python 3.10–3.12 × lint+test+validate+coverage, markdown lint), `.github/workflows/links.yml` (weekly lychee), 3 issue forms (`bug_report, feature_request, resource_suggestion`), `PULL_REQUEST_TEMPLATE.md`, `CODEOWNERS`.
- **Tooling**: `scripts/validate_repo.py` (files/dirs/14 guides/YouTube-link/secrets/size checks), `scripts/check_links.py` (offline counter + `--online` verifier).
- **Tracker**: `src/ai_roadmap/progress.py` CLI (`--summary/--check/--uncheck/--reset`, `.progress.json` state) mirroring the 7 README milestones; `tests/test_progress.py` (6 tests).
- **Scaffolds**: `data/samples/`, `notebooks/`, `projects/`, `templates/` each with a README contract; `ROADMAP_TO_100.md` chunk plan.
### Changed
- `.gitignore` now also ignores learner-local `.progress.json`.

## [1.0.0] - 2026-06-03
### Added
- **Interactive Visual Flowcharts**: Programmatic SVGs and generated high-resolution PNG graphics mapping learning journeys (`assets/`).
- **Comprehensive Documentation Path**: 14 detailed roadmap documents in the `docs/` folder, mapping out Math, ML, DL, CV, NLP, Generative AI, RAG, AI Agents, MLOps, resume formats, internship tactics, freelancing pricing, and open-source contributions.
- **50+ Projects Guide**: Tiered project guide mapping out 50 projects with technology suggestions and video references.
- **Complete Interview Prep**: Q&A guides covering theoretical ML/DL, engineering LLMs, RAG system design, and Python coding.
- **Curated Playlists & Videos**: Top-tier, hand-selected YouTube links attached to every single learning topic across all 7 phases.
- **Repository Guidelines**: Standard open-source metadata files (`LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`).

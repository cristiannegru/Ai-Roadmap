# Flashcards — 40 rapid-fire cards (all phases)

Cover the term, say the answer, then check. Each card links to the quiz with the full explanation.

## Phase 1 — Python & Math

| # | Front | Back |
|---|-------|------|
| 1 | `safe_divide(8, 0)` → ? | `0.0` (default, no crash) — [Quiz 1](quizzes/phase1_python_math.md) |
| 2 | `chunk_list` reappears where? | RAG document chunking — [Quiz 1](quizzes/phase1_python_math.md) |
| 3 | Cosine ≈ 1 means? Zero vector? | Same direction; zero → `0.0` — [Quiz 1](quizzes/phase1_python_math.md) |
| 4 | Bayes posterior: prior .01, like .9, evid .108 | ≈ 0.083 (base rates matter) — [Quiz 1](quizzes/phase1_python_math.md) |
| 5 | sigmoid / relu / softmax outputs | (0,1) prob / max(0,x) / sums-to-1 — [Quiz 1](quizzes/phase1_python_math.md) |
| 6 | `Experiment`'s two dicts | `params` (inputs) vs `metrics` (outputs) — [Quiz 1](quizzes/phase1_python_math.md) |

## Phase 2 — Data

| # | Front | Back |
|---|-------|------|
| 7 | Titanic sample: rows vs passengers? | 61 rows, 60 passengers (1 dupe) — [Quiz 2](quizzes/phase2_data.md) |
| 8 | Mean vs median imputation? | Median resists outliers (fare) — [Quiz 2](quizzes/phase2_data.md) |
| 9 | `drop_first=True` why? | Avoids dummy-variable trap — [Quiz 2](quizzes/phase2_data.md) |
| 10 | 1.5×IQR rule? | Outside Q1−1.5·IQR / Q3+1.5·IQR — [Quiz 2](quizzes/phase2_data.md) |
| 11 | `clean_dataframe` report keys? | rows, dupes removed, missing before/after, options — [Quiz 2](quizzes/phase2_data.md) |

## Phase 3 — ML

| # | Front | Back |
|---|-------|------|
| 12 | Preprocessing inside pipeline why? | No test→train leakage — [Quiz 3](quizzes/phase3_ml.md) |
| 13 | `handle_unknown="ignore"`? | Survives new categories at predict — [Quiz 3](quizzes/phase3_ml.md) |
| 14 | 95% accuracy, all-negative predictions? | Useless under imbalance; use F1 — [Quiz 3](quizzes/phase3_ml.md) |
| 15 | ROC-AUC needs what inputs? | Continuous scores, not labels — [Quiz 3](quizzes/phase3_ml.md) |
| 16 | Save pipeline, not classifier, why? | Preprocessing stats ship with weights — [Quiz 3](quizzes/phase3_ml.md) |

## Phase 4 — DL

| # | Front | Back |
|---|-------|------|
| 17 | CNN > MLP on images why? | Weight sharing + locality — [Quiz 4](quizzes/phase4_dl.md) |
| 18 | `AdaptiveAvgPool2d` buys? | Fixed head for any input size — [Quiz 4](quizzes/phase4_dl.md) |
| 19 | `get_device("auto")` order? | cuda → mps → cpu — [Quiz 4](quizzes/phase4_dl.md) |
| 20 | Forgot `model.eval()`? | Dropout/BN wrong + wasted grads — [Quiz 4](quizzes/phase4_dl.md) |
| 21 | ReLU > sigmoid hidden? | No saturation, gradients alive — [Quiz 4](quizzes/phase4_dl.md) |

## Phase 5 — RAG & Agents

| # | Front | Back |
|---|-------|------|
| 22 | Overlap in chunking why? | Saves split sentences — [Quiz 5](quizzes/phase5_genai.md) |
| 23 | `fit` computes what? | IDF weights per bin — [Quiz 5](quizzes/phase5_genai.md) |
| 24 | Query side must reuse…? | Index-time IDF (`.embedder.npz`) — [Quiz 5](quizzes/phase5_genai.md) |
| 25 | hit@k definition? Repo score? | Top-k doc match rate; 1.0 — [Quiz 5](quizzes/phase5_genai.md) |
| 26 | ReAct loop steps? | Thought → Action → Observation — [Quiz 5](quizzes/phase5_genai.md) |
| 27 | MCP server's 3 tools? | list/describe/query (SELECT-only) — [Quiz 5](quizzes/phase5_genai.md) |

## Phase 6 — Projects

| # | Front | Back |
|---|-------|------|
| 28 | SHORTLIST needs? | ≥60% skills + min years — [Quiz 6](quizzes/phase6_projects.md) |
| 29 | Vision: how many objects? | 3 (2 rects + disc) — [Quiz 6](quizzes/phase6_projects.md) |
| 30 | Bad API input status? | 422 (Pydantic), 400 (mismatch) — [Quiz 6](quizzes/phase6_projects.md) |
| 31 | Why train at startup in 07? | No stale artefacts — [Quiz 6](quizzes/phase6_projects.md) |

## Phase 7 — MLOps

| # | Front | Back |
|---|-------|------|
| 32 | `runs/<id>/run.json` holds? | params, metrics, tags, status — [Quiz 7](quizzes/phase7_mlops.md) |
| 33 | Lifespan load why? | Fail-fast at boot, not 3am — [Quiz 7](quizzes/phase7_mlops.md) |
| 34 | Compose services + ports? | api :8000, rag :8501 — [Quiz 7](quizzes/phase7_mlops.md) |
| 35 | p50/p95 from verification? | 13.8ms / 156ms, 0 failures — [Quiz 7](quizzes/phase7_mlops.md) |
| 36 | Docker CI triggers when? | PRs touching images/compose — [Quiz 7](quizzes/phase7_mlops.md) |

## Career

| # | Front | Back |
|---|-------|------|
| 37 | Precision vs recall priority? | Misses costly → recall — [Quiz 3](quizzes/phase3_ml.md) |
| 38 | L1 vs L2 (interview)? | L1 zeroes (selection); L2 shrinks — see `interview-preparation.md` |
| 39 | Vanishing gradient fixes? | ReLU, residuals, batchnorm — see `interview-preparation.md` |
| 40 | RAG at 10M users sketch? | Queue ingest, vector cluster, Redis cache, k8s — see `interview-preparation.md` |

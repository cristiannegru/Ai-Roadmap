# Projects (runnable portfolio — Chunk 5 ✅)

Each subfolder is self-contained: `README.md` + `requirements.txt` + CLI + tests in `tests/test_projects.py`.

| # | Folder | Tier | Stack | Run |
|---|--------|------|-------|-----|
| 01 | `01_spam_classifier/` | Beginner | TF-IDF, Naive Bayes | `python main.py` |
| 02 | `02_housing_regression/` | Beginner | Ridge, sklearn pipeline | `python main.py` |
| 03 | `03_resume_parser/` | Intermediate | Regex, keywords (spaCy swap) | `python main.py --resume sample_resume.txt` |
| 04 | `04_vision_detector/` | Intermediate | NumPy, PIL (YOLO swap) | `python main.py --show` |
| 05 | `05_pdf_rag_chatbot/` | Advanced | RAG pipeline, pypdf opt | `python main.py ingest && python main.py ask --query "..."` |
| 06 | `06_crewai_research_team/` | Expert | Crew template wrapper | `python main.py --query "..."` |
| 07 | `07_fastapi_titanic_api/` | MLOps bridge | FastAPI, Docker | `uvicorn app:app --port 8000` |

Verify all: `pytest tests/test_projects.py -q` (9 tests, all offline).

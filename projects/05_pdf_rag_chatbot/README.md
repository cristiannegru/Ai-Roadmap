# 05 — PDF/RAG Q&A Chatbot (Advanced)

CLI chatbot over your docs: ingest once, ask anything. Offline, no keys.

```bash
cd projects/05_pdf_rag_chatbot
pip install -r requirements.txt
python main.py ingest
python main.py ask --query "What is hybrid search?"
```

Drop `.md`/`.txt` (or `.pdf` with `pip install pypdf`) into `--docs` to
index your own knowledge base. The saved index is served as a chat UI by
`templates/streamlit_rag/`.

"""Tests for Chunk 4 templates (crew + MCP server, stdlib-first)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAG_DOCS = ROOT / "data" / "samples" / "rag_docs"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_crew_demo_cites_rag_doc():
    crew = _load("crew_template", ROOT / "templates" / "crewai_team" / "crew.py")
    result = crew.run_crew("How does hybrid search with reranking work?", RAG_DOCS, k=2)
    assert "rag_systems" in str(result["citations"][0])
    assert "Sources:" in str(result["draft"])
    assert result["issues"] == []  # critic passes on real retrieval


def test_crew_handles_empty_docs_dir(tmp_path):
    crew = _load("crew_template2", ROOT / "templates" / "crewai_team" / "crew.py")
    (tmp_path / "blank.md").write_text("   ", encoding="utf-8")
    try:
        crew.run_crew("anything", tmp_path, k=2)
        raise AssertionError("expected ValueError for empty docs")
    except ValueError:
        pass


def test_mcp_server_tools_and_guards(tmp_path):
    server = _load("mcp_template", ROOT / "templates" / "mcp_server" / "server.py")
    db = tmp_path / "demo.db"
    server.init_db(db)
    tables = server.handle({"id": 1, "tool": "list_tables", "args": {}}, db)
    assert tables["ok"] and "notes" in tables["result"]["tables"]
    desc = server.handle({"id": 2, "tool": "describe_table", "args": {"table": "notes"}}, db)
    assert desc["ok"] and any(c["name"] == "title" for c in desc["result"]["columns"])
    rows = server.handle(
        {"id": 3, "tool": "query_db", "args": {"sql": "SELECT title FROM notes"}}, db
    )
    assert rows["ok"] and len(rows["result"]["rows"]) == 3
    # Guards: destructive + multi-statement + unknown tool rejected.
    drop = server.handle({"id": 4, "tool": "query_db", "args": {"sql": "DROP TABLE notes"}}, db)
    assert not drop["ok"]
    multi = server.handle({"id": 5, "tool": "query_db", "args": {"sql": "SELECT 1; SELECT 2"}}, db)
    assert not multi["ok"]
    unknown = server.handle({"id": 6, "tool": "nope", "args": {}}, db)
    assert not unknown["ok"]


def test_mcp_server_stdio_smoke(tmp_path):
    db = tmp_path / "demo.db"
    req = json.dumps({"id": 1, "tool": "list_tables", "args": {}})
    proc = subprocess.run(
        [sys.executable, str(ROOT / "templates" / "mcp_server" / "server.py"), "--db", str(db)],
        input=req + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    resp = json.loads(proc.stdout.strip().splitlines()[-1])
    assert resp["ok"] and "notes" in resp["result"]["tables"]

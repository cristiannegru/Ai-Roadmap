"""Minimal MCP-concept SQLite server over line-delimited JSON stdio (stdlib only).

Speaks a tiny tool protocol with the same tool names you'd expose via the
real Model Context Protocol SDK, so agents built against this template
migrate by swapping the transport:

  request  {"id": 1, "tool": "<name>", "args": {...}}
  success  {"id": 1, "ok": true, "result": ...}
  failure  {"id": 1, "ok": false, "error": "..."}

Tools:
  list_tables   {}                                   -> {"tables": [...]}
  describe_table {"table": "notes"}                  -> {"columns": [...]}
  query_db      {"sql": "SELECT ..."}                -> {"columns": [...], "rows": [...]}

Safety: only single-statement SELECT/WITH queries; everything else rejected.
If `--db` is missing it is created with a sample `notes` table.

Usage:
    python server.py --db ./demo.db
    echo '{"id":1,"tool":"list_tables","args":{}}' | python server.py --db ./demo.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

TOOLS = ("list_tables", "describe_table", "query_db")


def init_db(path: Path) -> None:
    """Create ``path`` with a sample `notes` table if it does not exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT, body TEXT)")
        conn.executemany(
            "INSERT INTO notes (title, body) VALUES (?, ?)",
            [
                ("rag", "chunk embed retrieve cite"),
                ("agents", "react tools critic revision"),
                ("cnn", "conv relu pool checkpoint"),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def handle(request: dict[str, object], db_path: Path) -> dict[str, object]:
    """Execute one tool request dict, always returning a response dict."""
    rid = request.get("id")
    tool = request.get("tool")
    args = request.get("args", {})
    if not isinstance(args, dict):
        return {"id": rid, "ok": False, "error": "args must be an object"}
    if tool not in TOOLS:
        return {"id": rid, "ok": False, "error": f"unknown tool: {tool!r}"}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            if tool == "list_tables":
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                return {"id": rid, "ok": True, "result": {"tables": [r[0] for r in rows]}}
            if tool == "describe_table":
                table = str(args.get("table", ""))
                if not table.isidentifier():
                    return {"id": rid, "ok": False, "error": "invalid table name"}
                cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
                if not cols:
                    return {"id": rid, "ok": False, "error": f"no such table: {table}"}
                return {
                    "id": rid,
                    "ok": True,
                    "result": {"columns": [{"name": c["name"], "type": c["type"]} for c in cols]},
                }
            # query_db — read-only guard
            sql = str(args.get("sql", "")).strip().rstrip(";").strip()
            if not sql.upper().startswith(("SELECT", "WITH")):
                return {"id": rid, "ok": False, "error": "only SELECT/WITH allowed"}
            if ";" in sql:
                return {"id": rid, "ok": False, "error": "single statement only"}
            cur = conn.execute(sql)
            rows = cur.fetchmany(100)
            return {
                "id": rid,
                "ok": True,
                "result": {
                    "columns": [d[0] for d in cur.description or []],
                    "rows": [list(r) for r in rows],
                },
            }
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 — errors are the protocol payload
        return {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal MCP-concept SQLite server")
    parser.add_argument("--db", type=Path, default=Path("./demo.db"))
    args = parser.parse_args(argv)
    init_db(args.db)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(
                json.dumps({"id": None, "ok": False, "error": f"bad JSON: {e}"}) + "\n"
            )
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps(handle(request, args.db)) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# MCP-concept SQLite server (stdlib only, Chunk 4)

Exposes `list_tables`, `describe_table`, `query_db` over line-delimited JSON
stdio — the same tool names you'd register with the real MCP SDK.

```bash
cd templates/mcp_server
python server.py --db ./demo.db   # creates sample notes table on first run
echo '{"id":1,"tool":"list_tables","args":{}}' | python server.py --db ./demo.db
echo '{"id":2,"tool":"query_db","args":{"sql":"SELECT * FROM notes"}}' | python server.py --db ./demo.db
```

Safety: single-statement `SELECT`/`WITH` only, 100-row cap, identifier-checked
table names. Destructive statements are rejected.

## Go live (real MCP SDK)

Keep the three tool names + arg shapes; swap the stdio loop for
`mcp.server` transports. Agent code calling these tools stays identical.

# SysDiag MCP refactor


## Install

```bash
conda create -n sysdiag python=3.11 -y
conda activate sysdiag

pip install -r requirements.txt

pip install --force-reinstall --no-deps typing_inspection
conda run -n sysdiag pip install --force-reinstall -r requirements.txt

```

Set your OpenAI key:

```bash
export OPENAI_API_KEY=your_key_here
```

## Run

Start the API server:

```bash
python -m server.main
```

## Start a diagnosis
```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"issue": "High load average and slow response times"}'
```

Then poll status:

```bash
curl http://localhost:8000/diagnose/<session_id>
```
`session_id` can be found in the server log


## Follow up questions
```sh
curl -X POST http://localhost:8000/diagnose/<session_id>/followup \
  -H "Content-Type: application/json" \
  -d '{"question": "Could this be related to the database?"}'
```


## To see all the sessions and reports
```sh
curl http://localhost:8001/sessions | jq
```

## you can use it as a chatpot too
```sh
# defaults to localhost:8000
python chat.yp

# custom host/port
SYSDIAG_HOST=192.168.1.10 SYSDIAG_PORT=9000 python chat.py
```


## Notes

- The orchestrator launches `python -m client.mcp_server` as a subprocess (inside the `_run_loop` function)

## Architecture: Component Interaction Diagram

```
  MacBook (client machine)                Remote Server (1.1.1.1)
  ─────────────────────────               ────────────────────────────────────────────

┌──────────────────────────┐             ┌──────────────────────────────────────────┐
│   USER  (chat.py)        │             │   SERVER  (FastAPI — server/main.py)     │
│                          │──────────── │                                          │
│  python chat.py          │  :8000/HTTP │  • Creates session (SessionStore)        │
│  SYSDIAG_HOST=1.1.1.1    │ ──────────► │  • Spawns background task                │
│                          │ ◄────────── │  • Returns session_id immediately        │
└──────────────────────────┘             └──────────────────┬───────────────────────┘
                                                            │  run() / followup()
                                                            ▼
                                         ┌──────────────────────────────────────────┐
                                         │   AGENT ORCHESTRATOR  (server/agent.py)  │
                                         │                                          │
                                         │  1. Reads MCP_CLIENT_URL env var         │
                                         │  2. Connects to MacBook MCP via SSE      │
                                         │  3. Opens ClientSession, list_tools()    │
                                         │  4. Enters agentic loop:                 │
                                         │  ┌────────────────────────────────────┐  │
                                         │  │ messages → OpenAI GPT-4o →         │  │
                                         │  │   tool_calls or final report        │  │
                                         │  │     ↑            │                  │  │
                                         │  │     └─tool result┘                  │  │
                                         │  └────────────────────────────────────┘  │
                                         └───────────┬──────────────────┬───────────┘
                                                     │ Chat API (HTTPS) │ call_tool()
                                                     ▼                  │ SSE :8001
                                         ┌─────────────────┐            │
                                         │  OpenAI (GPT-4o) │            │
                                         │                  │            │
                                         │  Decides which   │            ▼
                                         │  tool to call,   │  ┌─────────────────────────────┐
                                         │  or writes final │  │  MCP SERVER  (MacBook)      │
                                         │  RCA report      │  │  client/mcp_server.py       │
                                         └─────────────────┘  │  transport: SSE  port: 8001  │
                                                              │                             │
                                                              │  FastMCP exposes tools:     │
                                                              │  • run_ping                 │
                                                              │  • list_processes           │
                                                              │  • check_disk               │
                                                              │  • check_memory / …         │
                                                              └─────────────────────────────┘
```

**How to run:**

On remote server (1.1.1.1) — start the FastAPI server:
```bash
MCP_CLIENT_URL=http://<client-ip>:8001/sse python -m server.main
# if use ssh tunnel
MCP_CLIENT_URL=http://localhost:8001/sse python -m server.main
# if use ngrok
MCP_CLIENT_URL=https://abc123.ngrok.io/sse python -m server.main
```

Terminal 1 (MacBook) — MCP tool server:
```bash
MCP_TRANSPORT=sse MCP_HOST=0.0.0.0 MCP_PORT=8001 python -m client.mcp_server
```

Terminal 2 (MacBook) — SSH reverse tunnel:
```sh
ssh -R 8001:localhost:8001 user@1.1.1.1
```

Terminal 3 (MacBook) — chat client
```bash
SYSDIAG_HOST=1.1.1.1 SYSDIAG_PORT=8000 python chat.py
```

**Flow summary:**

1. **User** runs `chat.py` on MacBook, which sends the issue to the remote FastAPI server on port `8000`.
2. **Server** creates a session and fires off `AgentOrchestrator` in the background.
3. **Orchestrator** connects to the MacBook's MCP tool server over SSE on port `8001`.
4. Each iteration: sends the conversation to **OpenAI GPT-4o**. If GPT wants a tool, the orchestrator calls it over **SSE** to the MacBook.
5. **MCP Server** (on MacBook) runs the diagnostic command locally and returns the result.
6. Results are fed back into the conversation until GPT produces a final RCA report.
7. **User** receives the report interactively via `chat.py` and can ask follow-up questions.


## MacBook Behind NAT (port 8001 not reachable from remote server)

If your MacBook is behind a router/firewall, the remote server cannot reach port `8001` directly. Two solutions:

### Option 1: SSH Reverse Tunnel (recommended — free, no extra tools)

On your MacBook, open a reverse tunnel so the remote server can reach your local MCP server via its own `localhost`:
```bash
ssh -R 8001:localhost:8001 user@1.1.1.1
```
Keep this session open, then start the remote server with:
```bash
MCP_CLIENT_URL=http://localhost:8001/sse python -m server.main
```

### Option 2: ngrok

On your MacBook:
```bash
ngrok http 8001
# outputs something like: https://abc123.ngrok.io
```
Then on the remote server:
```bash
MCP_CLIENT_URL=https://abc123.ngrok.io/sse python -m server.main
```

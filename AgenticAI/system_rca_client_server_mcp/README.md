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
curl http://localhost:8000/sessions | jq
```

## Notes

- The orchestrator launches `python -m client.mcp_server` as a subprocess (inside the `_run_loop` function)

## Architecture: Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER (HTTP Client)                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  POST /diagnose  {issue: "..."}
                               │  GET  /diagnose/{session_id}
                               │  POST /diagnose/{session_id}/followup
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SERVER  (FastAPI — server/main.py)               │
│                                                                     │
│  • Creates session (SessionStore)                                   │
│  • Spawns background task → AgentOrchestrator                       │
│  • Returns session_id immediately (async polling model)             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  run() / followup()
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               AGENT ORCHESTRATOR  (server/agent.py)                 │
│                                                                     │
│  1. Builds StdioServerParameters                                    │
│  2. Launches MCP subprocess (stdio_client)                          │
│  3. Opens ClientSession, calls list_tools()                         │
│  4. Enters agentic loop:                                            │
│     ┌─────────────────────────────────────────────────────┐        │
│     │  messages → OpenAI GPT-4o  →  tool_calls or report  │        │
│     │       ↑                              │               │        │
│     │       └──── tool result ────────────┘               │        │
│     └─────────────────────────────────────────────────────┘        │
└──────────┬───────────────────────────────────┬─────────────────────┘
           │  Chat API  (HTTPS)                │  mcp_session.call_tool()
           ▼                                   ▼  (over stdin/stdout pipe)
┌──────────────────────┐         ┌─────────────────────────────────────┐
│   OpenAI  (GPT-4o)   │         │   MCP SERVER  (client/mcp_server.py)│
│                      │         │                                     │
│  Decides which tool  │         │  FastMCP exposes TOOL_REGISTRY:     │
│  to call next, or    │         │  • run_ping                         │
│  writes final report │         │  • list_processes                   │
└──────────────────────┘         │  • check_disk / check_memory / …    │
                                 │                                     │
                                 │  Executes real shell commands on    │
                                 │  the target machine, returns output │
                                 └─────────────────────────────────────┘
```

**Flow summary:**

1. **User** sends an issue description via HTTP to the FastAPI server.
2. **Server** creates a session and fires off `AgentOrchestrator` in the background.
3. **Orchestrator** launches `client/mcp_server.py` as a subprocess (stdio pipe), discovers its tools, then enters a loop.
4. Each iteration: sends the conversation to **OpenAI GPT-4o**. If GPT wants a tool, the orchestrator calls it over the **MCP stdio pipe**.
5. **MCP Server** runs the actual diagnostic command on the machine and returns the result.
6. Results are fed back into the conversation until GPT produces a final RCA report.
7. **User** polls `GET /diagnose/{session_id}` until status is `DONE`, then reads the report.

"""
server/main.py — SysDiag MCP Server

Responsibilities
────────────────
  • Expose an MCP-compatible HTTP API via FastAPI
  • Tunnel it to the internet with ngrok so remote clients can reach it
  • Run the LangGraph diagnostic agent
  • When the agent needs a tool executed, park the call as a "pending"
    request; the client polls for it, runs it locally, and posts the result
  • Feed results back into the agent and continue the loop
  • Serve the final RCA report

Endpoints
─────────
  GET  /mcp                          — capability advertisement
  POST /mcp/sessions/register        — open a session
  POST /mcp/sessions/{id}/issue      — submit the problem description
  GET  /mcp/sessions/{id}/state      — poll: pending tool call or final report
  POST /mcp/sessions/{id}/tool-result — client posts command output
  GET  /mcp/sessions              — list active sessions (admin)

Run
───
  uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
  # or: python -m server.main
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Add project root so shared/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.models import (
    IssueRequest, IssueResponse,
    MCPInfo,
    RegisterRequest, RegisterResponse,
    SessionState, SessionStatus,
    ToolCallRequest, ToolResultRequest, ToolStatus,
)
from server.agent import AgentOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sysdiag.server")


###### In-memory session store

class SessionStore:
    """
    Thread-safe(ish) in-memory store for diagnostic sessions.
    For production replace with Redis or a DB.
    """

    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create(self, session_id: str, client_info: dict) -> None:
        async with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "client_info": client_info,
                "status": SessionStatus.PENDING,
                "issue": None,
                "pending_tool_call": None,
                "tool_result_event": asyncio.Event(),
                "tool_result": None,
                "final_report": "",
                "tool_call_count": 0,
                "error": "",
            }

    async def get(self, session_id: str) -> dict:
        async with self._lock:
            s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(session_id)
        return s

    async def update(self, session_id: str, **kwargs) -> None:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            self._sessions[session_id].update(kwargs)

    async def all(self) -> list[dict]:
        async with self._lock:
            return list(self._sessions.values())


store = SessionStore()


###### ngrok tunnel

def start_ngrok(port: int) -> str:
    """Start an ngrok tunnel and return the public URL."""
    try:
        from pyngrok import ngrok, conf

        ngrok_token = os.getenv("NGROK_AUTHTOKEN")
        if ngrok_token:
            conf.get_default().auth_token = ngrok_token

        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
        log.info("ngrok tunnel active: %s -> localhost:%d", public_url, port)
        return public_url
    except Exception as exc:
        log.warning("ngrok not started: %s", exc)
        return f"http://localhost:{port}"


###### App lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    port = int(os.getenv("PORT", 8000))
    public_url = start_ngrok(port)
    app.state.public_url = public_url
    log.info("SysDiag MCP Server ready at %s", public_url)
    log.info("Clients should connect to: %s/mcp", public_url)
    yield
    # Cleanup: close ngrok tunnels
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception:
        pass


app = FastAPI(
    title="SysDiag MCP Server",
    description="Model Context Protocol server for AI-driven system diagnostics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


###### Background diagnosis task

async def run_diagnosis(session_id: str, issue: str) -> None:
    """
    Run the LangGraph agent in a background task.

    Whenever the agent wants to call a tool, we:
      1. Park the call in the session as `pending_tool_call`
      2. Wait for the client to POST /tool-result (signalled via asyncio.Event)
      3. Feed the result back to the agent and continue
    """
    try:
        await store.update(session_id, status=SessionStatus.RUNNING)
        orchestrator = AgentOrchestrator(session_id, store)
        report = await orchestrator.run(issue)
        await store.update(
            session_id,
            status=SessionStatus.DONE,
            final_report=report,
            pending_tool_call=None,
        )
        log.info("[%s] Diagnosis complete.", session_id)
    except Exception as exc:
        log.exception("[%s] Diagnosis failed: %s", session_id, exc)
        await store.update(
            session_id,
            status=SessionStatus.ERROR,
            error=str(exc),
            pending_tool_call=None,
        )


###### Endpoints

@app.get("/mcp", response_model=MCPInfo)
async def mcp_info():
    """MCP capability advertisement."""
    base = getattr(app.state, "public_url", "http://localhost:8000")
    return MCPInfo(
        endpoints={
            "register":    f"{base}/mcp/sessions/register",
            "issue":       f"{base}/mcp/sessions/{{id}}/issue",
            "state":       f"{base}/mcp/sessions/{{id}}/state",
            "tool_result": f"{base}/mcp/sessions/{{id}}/tool-result",
            "sessions":    f"{base}/mcp/sessions",
        }
    )


@app.post("/mcp/sessions/register", response_model=RegisterResponse)
async def register_session(req: RegisterRequest):
    """Client opens a diagnostic session."""
    session_id = str(uuid.uuid4())
    await store.create(session_id, {
        "client_id": req.client_id,
        "hostname":  req.hostname,
        "platform":  req.platform,
    })
    log.info("New session %s from client=%s host=%s", session_id, req.client_id, req.hostname)
    return RegisterResponse(session_id=session_id)


@app.post("/mcp/sessions/{session_id}/issue", response_model=IssueResponse)
async def submit_issue(session_id: str, req: IssueRequest, background: BackgroundTasks):
    """Client submits the problem description; kicks off the agent."""
    try:
        session = await store.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found.")

    if session["status"] != SessionStatus.PENDING:
        raise HTTPException(409, "Session already has an issue submitted.")

    if req.session_id != session_id:
        raise HTTPException(400, "session_id mismatch.")

    await store.update(session_id, issue=req.issue)
    background.add_task(run_diagnosis, session_id, req.issue)

    return IssueResponse(
        session_id=session_id,
        status=SessionStatus.RUNNING,
        message="Diagnosis started. Poll /state for updates.",
    )


@app.get("/mcp/sessions/{session_id}/state", response_model=SessionState)
async def get_state(session_id: str):
    """
    Client polls this endpoint to:
      a) receive a pending tool call to execute, OR
      b) learn the session is DONE and retrieve the final report.
    """
    try:
        s = await store.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found.")

    return SessionState(
        session_id=session_id,
        status=s["status"],
        pending_tool_call=s["pending_tool_call"],
        final_report=s["final_report"],
        tool_call_count=s["tool_call_count"],
        error=s["error"],
    )


@app.post("/mcp/sessions/{session_id}/tool-result")
async def post_tool_result(session_id: str, req: ToolResultRequest):
    """
    Client posts the output of a tool it ran locally.
    Wakes the agent's waiting coroutine.
    """
    try:
        s = await store.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found.")

    pending = s["pending_tool_call"]
    if pending is None:
        raise HTTPException(409, "No tool call is pending for this session.")

    if req.call_id != pending.call_id:
        raise HTTPException(400, f"call_id mismatch: expected {pending.call_id}.")

    # Deliver result and signal the agent coroutine
    async with asyncio.Lock():
        s["tool_result"] = req
        s["tool_result_event"].set()

    log.info("[%s] Tool result received: %s → %s", session_id, req.tool_name, req.status)
    return {"ok": True}


@app.get("/mcp/sessions")
async def list_sessions():
    """Admin: list all active sessions."""
    sessions = await store.all()
    return [
        {
            "session_id":      s["session_id"],
            "status":          s["status"],
            "hostname":        s["client_info"].get("hostname", "?"),
            "tool_call_count": s["tool_call_count"],
            "issue":           (s["issue"] or "")[:80],
        }
        for s in sessions
    ]


###### Entry point

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server.main:app", host="0.0.0.0", port=port, reload=False)

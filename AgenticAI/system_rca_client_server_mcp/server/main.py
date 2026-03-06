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
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.models import DiagnoseRequest, DiagnoseResponse, DiagnoseState, SessionStatus
from server.agent import AgentOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sysdiag.server")


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create(self, session_id: str) -> None:
        async with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "status": SessionStatus.PENDING,
                "issue": None,
                "tool_call_count": 0,
                "final_report": "",
                "error": "",
            }

    async def get(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    async def update(self, session_id: str, **kwargs) -> None:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            self._sessions[session_id].update(kwargs)

    async def all(self) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._sessions.values())


store = SessionStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    log.info("SysDiag server ready")
    yield


app = FastAPI(
    title="SysDiag Server",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_diagnosis(session_id: str, issue: str) -> None:
    try:
        orchestrator = AgentOrchestrator(session_id=session_id, store=store)
        report = await orchestrator.run(issue)
        await store.update(
            session_id,
            status=SessionStatus.DONE,
            final_report=report,
        )
    except Exception as exc:
        log.exception("[%s] Diagnosis failed: %s", session_id, exc)
        await store.update(
            session_id,
            status=SessionStatus.ERROR,
            error=str(exc),
        )


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(req: DiagnoseRequest, background: BackgroundTasks) -> DiagnoseResponse:
    session_id = str(uuid.uuid4())
    await store.create(session_id)
    await store.update(session_id, issue=req.issue, status=SessionStatus.RUNNING)
    background.add_task(run_diagnosis, session_id, req.issue)
    return DiagnoseResponse(session_id=session_id, status=SessionStatus.RUNNING)


@app.get("/diagnose/{session_id}", response_model=DiagnoseState)
async def diagnose_state(session_id: str) -> DiagnoseState:
    try:
        session = await store.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")

    return DiagnoseState(
        session_id=session_id,
        status=session["status"],
        tool_call_count=session["tool_call_count"],
        final_report=session["final_report"],
        error=session["error"],
    )


@app.get("/sessions")
async def sessions() -> list[dict[str, Any]]:
    return await store.all()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server.main:app", host="0.0.0.0", port=port, reload=False)

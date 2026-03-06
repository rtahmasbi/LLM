from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class DiagnoseRequest(BaseModel):
    issue: str = Field(..., min_length=1)


class DiagnoseResponse(BaseModel):
    session_id: str
    status: SessionStatus


class DiagnoseState(BaseModel):
    session_id: str
    status: SessionStatus
    tool_call_count: int = 0
    final_report: str = ""
    error: str = ""

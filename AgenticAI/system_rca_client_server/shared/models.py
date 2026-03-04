"""
Both the server and client import from this file so the schema
stays in one place.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


###### Enums

class SessionStatus(str, Enum):
    PENDING   = "pending"    # client registered, no issue submitted yet
    RUNNING   = "running"    # agent is actively diagnosing
    DONE      = "done"       # final report produced
    ERROR     = "error"      # unrecoverable error


class ToolStatus(str, Enum):
    OK      = "ok"
    BLOCKED = "blocked"      # guardrail fired on the client
    ERROR   = "error"        # subprocess failed


###### MCP Session lifecycle

class RegisterRequest(BaseModel):
    """Client → Server: open a new diagnostic session."""
    client_id: str = Field(..., description="Stable identifier for this client machine.")
    hostname: str  = Field(..., description="Hostname reported by the client.")
    platform: str  = Field(..., description="e.g. 'Linux 6.8.0-x86_64'")


class RegisterResponse(BaseModel):
    session_id: str
    message: str = "Session opened. Submit an issue to begin diagnosis."


class IssueRequest(BaseModel):
    """Client → Server: describe the problem to investigate."""
    session_id: str
    issue: str = Field(..., min_length=4)


class IssueResponse(BaseModel):
    session_id: str
    status: SessionStatus
    message: str


###### Tool call round-trip

class ToolCallRequest(BaseModel):
    """
    Server → Client (polled by client): the agent wants to run a tool.
    The client executes it locally and posts a ToolResultRequest back.
    """
    session_id: str
    call_id: str          # uuid, so results can be matched
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultRequest(BaseModel):
    """Client → Server: result of a tool execution."""
    session_id: str
    call_id: str
    tool_name: str
    status: ToolStatus
    output: str           # raw text output (or error/blocked message)


###### Status & report polling

class SessionState(BaseModel):
    """Server → Client: full current state of a session."""
    session_id: str
    status: SessionStatus
    pending_tool_call: ToolCallRequest | None = None
    final_report: str = ""
    tool_call_count: int = 0
    error: str = ""


###### MCP meta-endpoints

class MCPInfo(BaseModel):
    """GET /mcp — server capability advertisement."""
    name: str = "SysDiag MCP Server"
    version: str = "1.0.0"
    protocol: str = "sysdiag-mcp/1"
    endpoints: dict[str, str] = Field(default_factory=dict)

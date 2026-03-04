#!/usr/bin/env python3
"""
client/main.py — SysDiag MCP Client

Responsibilities
────────────────
  • Register with the MCP server (reached via the ngrok public URL)
  • Submit the user's issue description
  • Poll /state for pending tool calls
  • Execute each tool call LOCALLY (with guardrails)
  • POST the result back to the server
  • Print the final RCA report when done

Usage
─────
  python -m client.main --server https://xxxx.ngrok-free.app --issue "System is slow"
  python -m client.main --server https://xxxx.ngrok-free.app   # interactive mode
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client.src.tools import TOOL_REGISTRY
from shared.models import (
    IssueRequest, RegisterRequest,
    SessionStatus, ToolResultRequest, ToolStatus,
)

POLL_INTERVAL   = 2.0   # seconds between /state polls
REQUEST_TIMEOUT = 30    # httpx timeout


###### HTTP helpers

def _post(base: str, path: str, payload: dict) -> dict:
    url = base.rstrip("/") + path
    with httpx.Client(timeout=REQUEST_TIMEOUT) as c:
        r = c.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def _get(base: str, path: str) -> dict:
    url = base.rstrip("/") + path
    with httpx.Client(timeout=REQUEST_TIMEOUT) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.json()


###### Client orchestration

def run_session(server_url: str, issue: str) -> None:
    """Full lifecycle: register → submit issue → tool loop → report."""

    client_id = os.getenv("CLIENT_ID", str(uuid.uuid4()))
    hostname   = socket.gethostname()
    plat       = f"{platform.system()} {platform.release()} {platform.machine()}"

    # 1. Register
    print(f"[*] Connecting to MCP server: {server_url}")
    reg_resp = _post(
        server_url,
        "/mcp/sessions/register",
        RegisterRequest(
            client_id=client_id,
            hostname=hostname,
            platform=plat,
        ).model_dump(),
    )
    session_id = reg_resp["session_id"]
    print(f"[*] Session opened: {session_id}")

    # 2. Submit issue
    _post(
        server_url,
        f"/mcp/sessions/{session_id}/issue",
        IssueRequest(session_id=session_id, issue=issue).model_dump(),
    )
    print(f"[*] Issue submitted: {issue}")
    print("[*] Agent is investigating... (polling for tool requests)\n")

    # 3. Poll / execute loop
    last_call_id = None
    while True:
        time.sleep(POLL_INTERVAL)
        state = _get(server_url, f"/mcp/sessions/{session_id}/state")
        status = state["status"]

        ###### Done or error
        if status == SessionStatus.DONE:
            print("\n" + "=" * 60)
            print("FINAL ROOT-CAUSE ANALYSIS REPORT")
            print("=" * 60)
            print(state.get("final_report", "(no report)"))
            break

        if status == SessionStatus.ERROR:
            print(f"\n[ERROR] Server reported an error: {state.get('error', '?')}")
            sys.exit(1)

        ###### Pending tool call
        pending = state.get("pending_tool_call")
        if not pending:
            continue  # still thinking, poll again

        call_id   = pending["call_id"]
        tool_name = pending["tool_name"]
        arguments = pending.get("arguments", {})

        # Avoid re-executing the same call if we already posted a result
        # (server may not have cleared it yet)
        if call_id == last_call_id:
            continue

        print(f"[>>] Executing: {tool_name}({_fmt_args(arguments)})")

        # Execute locally
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            output = f"[ERROR] Unknown tool: {tool_name}"
            tool_status = ToolStatus.ERROR
        else:
            try:
                output = fn(**arguments)
                # Determine status from output prefix
                if output.startswith("[BLOCKED]"):
                    tool_status = ToolStatus.BLOCKED
                elif output.startswith("[ERROR]") or output.startswith("[UNAVAILABLE]"):
                    tool_status = ToolStatus.ERROR
                else:
                    tool_status = ToolStatus.OK
            except Exception as exc:
                output = f"[ERROR] Exception in {tool_name}: {exc}"
                tool_status = ToolStatus.ERROR

        print(f"[<<] {tool_name}: {tool_status.value} ({len(output)} chars)")

        # Post result
        _post(
            server_url,
            f"/mcp/sessions/{session_id}/tool-result",
            ToolResultRequest(
                session_id=session_id,
                call_id=call_id,
                tool_name=tool_name,
                status=tool_status,
                output=output,
            ).model_dump(),
        )
        last_call_id = call_id


def _fmt_args(args: dict) -> str:
    if not args:
        return ""
    parts = [f"{k}={repr(v)}" for k, v in args.items()]
    s = ", ".join(parts)
    return s[:80] + ("…" if len(s) > 80 else "")


###### Entry point

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SysDiag MCP Client — runs diagnostic commands locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m client.main --server https://xxxx.ngrok-free.app \\
         --issue "High load average and slow response times"

  python -m client.main --server https://xxxx.ngrok-free.app
        """,
    )
    parser.add_argument(
        "--server",
        default=os.getenv("MCP_SERVER_URL", ""),
        help="MCP server URL (ngrok public URL). Can also be set via MCP_SERVER_URL env var.",
    )
    parser.add_argument(
        "--issue",
        default="",
        help="Issue description. If omitted, you will be prompted interactively.",
    )
    args = parser.parse_args()

    server_url = args.server.strip()
    if not server_url:
        sys.exit(
            "Error: --server is required.\n"
            "Example: python -m client.main --server https://xxxx.ngrok-free.app"
        )

    # Verify server reachable
    try:
        info = _get(server_url, "/mcp")
        print(f"[*] Connected to: {info.get('name', '?')} v{info.get('version', '?')}")
    except Exception as exc:
        sys.exit(f"[ERROR] Cannot reach MCP server at {server_url}: {exc}")

    issue = args.issue.strip()
    if not issue:
        try:
            issue = input("Describe the issue: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)

    if not issue:
        sys.exit("Error: issue description cannot be empty.")

    run_session(server_url, issue)


if __name__ == "__main__":
    main()

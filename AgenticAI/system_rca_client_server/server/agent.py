"""
server/agent.py — LangGraph orchestrator (server side)

Key difference from the standalone version
───────────────────────────────────────────
When the agent issues a tool call, instead of running it here we:
  1. Write a ToolCallRequest into the session store (the client polls for it)
  2. Await an asyncio.Event that the /tool-result endpoint will set
  3. Read the ToolResultRequest the client posted
  4. Inject it as a "tool" message and continue the LangGraph loop

The graph topology:
  analyst → tool_executor → analyst  (loop)
          → reporter → END
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, TypedDict

from openai import AsyncOpenAI

log = logging.getLogger("sysdiag.agent")

MODEL = "gpt-4o"
MAX_TOOL_CALLS = 20
MAX_TOKENS = 4096

SYSTEM_PROMPT = """\
You are SysDiag, an expert Linux system-reliability engineer.
A user has reported a system issue. A remote client agent will execute
diagnostic commands on the target machine and send you the outputs.

Guidelines
──────────
1. Think step-by-step before each tool call — state what you expect to learn.
2. After each result, reason about what it reveals and what to check next.
3. Never repeat the same command.
4. Stop calling tools once you have enough evidence (or after 15 tool calls).
5. You cannot run destructive commands — the client enforces guardrails.

Available tools are standard Linux diagnostics:
  list_processes, check_memory, check_disk, check_cpu_info,
  check_system_load, check_network, read_journal_logs, read_dmesg,
  read_log_file, check_service_status, find_open_files, run_perf_stat,
  run_command (generic, guardrail-validated).

Final report format (use EXACTLY when done)
────────────────────────────────────────────
## Root-Cause Analysis

### Summary
<one-sentence root cause>

### Symptoms Observed
- …

### Evidence
| Tool | Key finding |
|------|-------------|
| …    | …           |

### Root Cause
<detailed explanation>

### Recommended Remediation
1. …

### Confidence
<Low / Medium / High> — <rationale>
"""

# Tool schemas — identical to the standalone version
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a single read-only diagnostic shell command. "
                "No shell operators (|, ;, &, $) are allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Full command, e.g. 'ps aux'"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_processes",
            "description": "List running processes sorted by CPU or memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sort_by": {"type": "string", "enum": ["cpu", "mem", "pid"]},
                    "top_n":   {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_memory",
            "description": "Return system memory and swap usage (free -h).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_disk",
            "description": "Return disk space usage for all mounted filesystems (df -h).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_cpu_info",
            "description": "Return CPU model, core count, and frequency (lscpu).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_system_load",
            "description": "Return load averages and vmstat sample.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_network",
            "description": "Return open sockets and listening ports (ss -tulnp).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_journal_logs",
            "description": "Read systemd journal logs filtered by unit and priority.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit":     {"type": "string"},
                    "lines":    {"type": "integer"},
                    "priority": {
                        "type": "string",
                        "enum": ["emerg", "alert", "crit", "err",
                                 "warning", "notice", "info", "debug"],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_dmesg",
            "description": "Return recent kernel ring-buffer messages (with permission-aware fallback).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lines": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_log_file",
            "description": "Read the tail of a log file under /var/log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":       {"type": "string"},
                    "tail_lines": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_service_status",
            "description": "Return systemctl status for a named service (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"}
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_open_files",
            "description": "List open file descriptors for a PID (or all processes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {"type": "string"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_perf_stat",
            "description": "Run perf stat to collect CPU performance counters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid":      {"type": "string"},
                    "duration": {"type": "integer"},
                },
            },
        },
    },
]


class AgentOrchestrator:
    """Drives the LangGraph-style agent loop for one diagnostic session."""

    TOOL_RESULT_TIMEOUT = 120  # seconds to wait for client to post a result

    def __init__(self, session_id: str, store):
        self.session_id = session_id
        self.store = store
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def run(self, issue: str) -> str:
        """Run the full diagnosis loop and return the final report."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Issue reported: {issue}"},
        ]
        tool_call_count = 0

        while True:
            ###### Analyst step
            log.info("[%s] Calling analyst (tool_calls so far: %d)", self.session_id, tool_call_count)
            response = await self.client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                tools=TOOL_SCHEMAS,
                tool_choice="auto" if tool_call_count < MAX_TOOL_CALLS else "none",
                messages=messages,
            )

            msg = response.choices[0].message
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }

            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            ###### No more tool calls → done
            if not msg.tool_calls or tool_call_count >= MAX_TOOL_CALLS:
                if not (msg.content or "").strip():
                    # Hit cap without a report — ask for one
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have reached the maximum tool-call limit. "
                            "Based on the evidence collected, write the final "
                            "Root-Cause Analysis using the required format."
                        ),
                    })
                    final = await self.client.chat.completions.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        messages=messages,
                    )
                    return final.choices[0].message.content or "(no report)"
                return msg.content

            ###### Tool execution round-trip
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    kwargs = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    kwargs = {}

                call_id = str(uuid.uuid4())
                pending = {
                    "session_id": self.session_id,
                    "call_id":    call_id,
                    "tool_name":  fn_name,
                    "arguments":  kwargs,
                }

                # Publish the pending call and reset the event
                s = await self.store.get(self.session_id)
                s["tool_result_event"].clear()
                s["tool_result"] = None

                from shared.models import ToolCallRequest
                await self.store.update(
                    self.session_id,
                    pending_tool_call=ToolCallRequest(**pending),
                    tool_call_count=tool_call_count + 1,
                )

                log.info("[%s] Waiting for client to execute: %s(%s)",
                         self.session_id, fn_name, str(kwargs)[:80])

                # Wait for client to post result
                try:
                    await asyncio.wait_for(
                        s["tool_result_event"].wait(),
                        timeout=self.TOOL_RESULT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    output = (
                        f"[TIMEOUT] Client did not return a result for "
                        f"'{fn_name}' within {self.TOOL_RESULT_TIMEOUT}s."
                    )
                else:
                    result: Any = s["tool_result"]
                    output = result.output if result else "[ERROR] Empty result."

                # Clear the pending slot
                await self.store.update(self.session_id, pending_tool_call=None)

                messages.append({
                    "role":        "tool",
                    "tool_call_id": tc.id,
                    "name":        fn_name,
                    "content":     output,
                })
                tool_call_count += 1
                log.info("[%s] Tool %s complete (total: %d)", self.session_id, fn_name, tool_call_count)

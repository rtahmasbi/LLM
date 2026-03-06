from __future__ import annotations

import json
import logging
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

log = logging.getLogger("sysdiag.agent")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "20"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

SYSTEM_PROMPT = """\
You are SysDiag, an expert Linux system-reliability engineer.
A user has reported a system issue. You have access to diagnostic tools
running on the target machine through MCP.

Guidelines
──────────
1. Think step-by-step before each tool call — state what you expect to learn.
2. After each result, reason about what it reveals and what to check next.
3. Never repeat the same command.
4. Stop calling tools once you have enough evidence (or after 15 tool calls).
5. You cannot run destructive commands.

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


class AgentOrchestrator:
    """Drives the diagnosis loop and calls remote tools over MCP."""

    def __init__(
        self,
        session_id: str,
        store,
        client_command: list[str] | None = None,
    ) -> None:
        self.session_id = session_id
        self.store = store
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.client_command = client_command or ["python", "-m", "client.mcp_server"]

    async def _call_mcp_tool(
        self,
        mcp_session: ClientSession,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> str:
        result = await mcp_session.call_tool(tool_name, kwargs)

        if hasattr(result, "content"):
            parts: list[str] = []
            for item in result.content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts) if parts else str(result)

        return str(result)

    async def run(self, issue: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Issue reported: {issue}"},
        ]
        tool_call_count = 0

        server_params = StdioServerParameters(
            command=self.client_command[0],
            args=self.client_command[1:],
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as mcp_session:
                await mcp_session.initialize()
                tools_info = await mcp_session.list_tools()

                openai_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    }
                    for tool in tools_info.tools
                ]

                while True:
                    log.info("[%s] Calling analyst (tool_calls=%d)", self.session_id, tool_call_count)

                    response = await self.client.chat.completions.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        tools=openai_tools,
                        tool_choice="auto" if tool_call_count < MAX_TOOL_CALLS else "none",
                        messages=messages,
                    )

                    msg = response.choices[0].message
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": msg.content,
                    }

                    if msg.tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ]

                    messages.append(assistant_msg)

                    if not msg.tool_calls or tool_call_count >= MAX_TOOL_CALLS:
                        if not (msg.content or "").strip():
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "You have reached the maximum tool-call limit. "
                                        "Based on the evidence collected, write the final "
                                        "Root-Cause Analysis using the required format."
                                    ),
                                }
                            )
                            final = await self.client.chat.completions.create(
                                model=MODEL,
                                max_tokens=MAX_TOKENS,
                                messages=messages,
                            )
                            return final.choices[0].message.content or "(no report)"
                        return msg.content or "(no report)"

                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        try:
                            kwargs = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            kwargs = {}

                        log.info("[%s] MCP tool call: %s(%s)", self.session_id, fn_name, kwargs)

                        try:
                            output = await self._call_mcp_tool(mcp_session, fn_name, kwargs)
                        except Exception as exc:
                            output = f"[ERROR] MCP tool failed: {exc}"

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": fn_name,
                                "content": output,
                            }
                        )
                        tool_call_count += 1
                        await self.store.update(
                            session_id=self.session_id,
                            tool_call_count=tool_call_count,
                        )

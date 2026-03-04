"""
LangGraph-based agentic orchestration for system diagnostics.

Graph topology
──────────────
  [START]
     │
     ▼
  analyst          ← calls OpenAI with tool schemas; decides what to run
     │
     ├─ (tool_calls present) ──► tool_executor  ← runs the tools safely
     │                                │
     │                                └──────────► analyst  (loop)
     │
     └─ (no tool_calls / done) ──► reporter     ← writes the final report
                                        │
                                     [END]

State fields
────────────
  issue           : str                  - original user report
  messages        : list[dict]           - full conversation history
  tool_call_count : int                  - number of tool invocations so far
  final_report    : str                  - populated by reporter node
"""

from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from openai import OpenAI

from .tools import TOOL_SCHEMAS, TOOL_REGISTRY

##### Constants

MODEL = "gpt-4o"
MAX_TOOL_CALLS = 20          # hard cap to prevent runaway loops
MAX_TOKENS_PER_CALL = 4096

SYSTEM_PROMPT = """\
You are SysDiag, an expert Linux system-reliability engineer and kernel developer.
A user has reported an issue with their system. Your job is to investigate it
systematically by running diagnostic tools, interpret the results, and produce
a structured Root-Cause Analysis (RCA) report.

Guidelines
──────────
1. Think step-by-step. Before calling a tool, briefly state what you expect to
   learn from it.
2. Use the most specific tool available (e.g. check_service_status for a
   named daemon rather than a raw journalctl call).
3. After each tool result, reason about what it tells you and what to check next.
4. Do NOT run the same command twice.
5. When you have gathered enough evidence — or after 15 tool calls — stop calling
   tools and produce the final report.
6. NEVER attempt destructive operations. All write/modify commands are blocked.

Final report format (use this EXACTLY when you are done investigating)
───────────────────────────────────────────────────────────────────────
## Root-Cause Analysis

### Summary
<One-sentence description of the root cause>

### Symptoms Observed
- …

### Evidence
| Tool used | Key finding |
|-----------|-------------|
| …         | …           |

### Root Cause
<Detailed explanation>

### Recommended Remediation
1. …
2. …

### Confidence
<Low / Medium / High> — <brief rationale>
"""



class AgentState(TypedDict):
    issue: str
    messages: list[dict[str, Any]]
    tool_call_count: int
    final_report: str



def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Export it or add it to a .env file."
        )
    return OpenAI(api_key=api_key)


def analyst_node(state: AgentState) -> AgentState:
    """
    Call the LLM. If it returns tool_calls, append them to messages.
    If it returns plain text, set final_report and clear pending tool calls.
    """
    client = build_client()

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_PER_CALL,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        messages=state["messages"],
    )

    msg = response.choices[0].message
    # Convert to plain dict for serialisation
    assistant_message: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}

    if msg.tool_calls:
        assistant_message["tool_calls"] = [
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

    new_messages = state["messages"] + [assistant_message]

    return {
        **state,
        "messages": new_messages,
        "final_report": msg.content if not msg.tool_calls else state.get("final_report", ""),
    }


def tool_executor_node(state: AgentState) -> AgentState:
    """
    Execute every pending tool call from the last assistant message,
    append individual tool result messages, and increment counter.
    """
    last_msg = state["messages"][-1]
    tool_calls = last_msg.get("tool_calls", [])

    new_messages = list(state["messages"])
    executed = 0

    for tc in tool_calls:
        fn_name = tc["function"]["name"]
        try:
            kwargs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            kwargs = {}

        fn = TOOL_REGISTRY.get(fn_name)
        if fn is None:
            result = f"[ERROR] Unknown tool: {fn_name}"
        else:
            try:
                result = fn(**kwargs)
            except Exception as exc:  # noqa: BLE001
                result = f"[ERROR] Tool '{fn_name}' raised an exception: {exc}"

        new_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": fn_name,
                "content": str(result),
            }
        )
        executed += 1

    return {
        **state,
        "messages": new_messages,
        "tool_call_count": state["tool_call_count"] + executed,
    }


def reporter_node(state: AgentState) -> AgentState:
    """
    If the analyst didn't produce a final report (e.g. hit the tool cap),
    ask it to summarise what it found.
    """
    if state.get("final_report", "").strip():
        return state  # already has a report

    client = build_client()
    summary_prompt = (
        "You have reached the maximum number of diagnostic tool calls. "
        "Based on all the evidence collected so far, please write the final "
        "Root-Cause Analysis report using the required format."
    )
    messages = state["messages"] + [{"role": "user", "content": summary_prompt}]

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_PER_CALL,
        messages=messages,
    )
    report = response.choices[0].message.content or "(No report generated)"

    return {**state, "final_report": report}


def should_continue(state: AgentState) -> str:
    """
    After analyst_node:
      - If there are pending tool calls AND we haven't hit the cap → 'tools'
      - Otherwise → 'done'
    """
    last_msg = state["messages"][-1]
    has_tool_calls = bool(last_msg.get("tool_calls"))
    at_cap = state["tool_call_count"] >= MAX_TOOL_CALLS

    if has_tool_calls and not at_cap:
        return "tools"
    return "done"


def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Returns a compiled graph ready for .invoke().
    """
    from langgraph.graph import StateGraph, END

    builder = StateGraph(AgentState)

    builder.add_node("analyst", analyst_node)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("reporter", reporter_node)

    builder.set_entry_point("analyst")

    builder.add_conditional_edges(
        "analyst",
        should_continue,
        {
            "tools": "tool_executor",
            "done": "reporter",
        },
    )

    builder.add_edge("tool_executor", "analyst")
    builder.add_edge("reporter", END)

    return builder.compile()


def run_diagnosis(issue: str) -> tuple[str, list[dict]]:
    """
    Entry point: run the full diagnostic graph for the reported issue.

    Returns
    -------
    report   : str             - the final RCA report
    messages : list[dict]      - full conversation history
    """
    graph = build_graph()

    initial_state: AgentState = {
        "issue": issue,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Issue reported: {issue}"},
        ],
        "tool_call_count": 0,
        "final_report": "",
    }

    final_state = graph.invoke(initial_state)
    return final_state["final_report"], final_state["messages"]

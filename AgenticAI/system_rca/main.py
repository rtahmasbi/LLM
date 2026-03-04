#!/usr/bin/env python3
"""
SysDiag CLI — interactive system diagnostic agent
Usage: python main.py [--issue "describe your problem"]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.agent import run_diagnosis
from src.guardrails import ALLOWED_COMMANDS, BLOCKED_COMMANDS, BLOCKED_PATH_PREFIXES


############# helpers

def _divider(char: str = "-", width: int = 60) -> None:
    print(char * width)


def print_banner() -> None:
    _divider("=")
    print("  SysDiag -- AI System Diagnostic Agent")
    print("  Powered by OpenAI + LangGraph")
    _divider("=")


def print_policy_summary() -> None:
    print("\n[Guardrails active]")
    print(f"  Allowed commands : {len(ALLOWED_COMMANDS)}")
    print(f"  Blocked commands : {len(BLOCKED_COMMANDS)}")
    print(f"  Blocked paths    : {', '.join(BLOCKED_PATH_PREFIXES)}")


def stream_messages(messages: list[dict]) -> None:
    """Print the agent conversation history to stdout."""
    _divider()
    print("DIAGNOSTIC TRACE")
    _divider()

    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "") or ""

        if role == "system":
            continue

        elif role == "user":
            print(f"\n[USER]\n{content}")

        elif role == "assistant":
            tool_calls = m.get("tool_calls", [])
            if tool_calls:
                calls_text = "\n".join(
                    f"  >> {tc['function']['name']}({tc['function']['arguments'][:120]})"
                    for tc in tool_calls
                )
                body = (content + "\n" + calls_text).strip()
            else:
                body = content[:600]
            print(f"\n[ANALYST]\n{body}")

        elif role == "tool":
            name = m.get("name", "tool")
            result = content[:800]
            print(f"\n[TOOL: {name}]\n{result}")


def _run_single(issue: str) -> tuple[str, list[dict]]:
    """Run one diagnostic cycle and return (report, messages)."""
    print("Investigating... (this may take a minute)\n")
    start = time.time()
    report, messages = run_diagnosis(issue)
    elapsed = time.time() - start
    tool_count = sum(1 for m in messages if m.get("role") == "tool")
    print(f"Done in {elapsed:.1f}s | Tool calls: {tool_count}\n")
    return report, messages


def run_interactive() -> None:
    """Run in interactive REPL mode."""
    print_banner()
    print_policy_summary()
    print("\nDescribe your system issue. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            issue = input("Issue: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not issue:
            continue
        if issue.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        report, messages = _run_single(issue)
        stream_messages(messages)
        _divider("=")
        print("FINAL REPORT")
        _divider("=")
        print(report)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SysDiag -- AI-powered system diagnostic agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --issue "System is very slow, load average is 20"
  python main.py --issue "nginx won't start after last update" --trace
  python main.py --issue "Disk full alert on /dev/sda1" --json-out report.json
        """,
    )
    parser.add_argument(
        "--issue",
        type=str,
        default="",
        help="Issue description (if omitted, run interactive mode)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print full tool-call trace even in single-issue mode",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        metavar="FILE",
        help="Save full conversation + report as JSON to FILE",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "Export it or create a .env file with OPENAI_API_KEY=sk-..."
        )

    if args.issue:
        print_banner()
        print(f"\nIssue: {args.issue}\n")

        report, messages = _run_single(args.issue)

        if args.trace:
            stream_messages(messages)

        _divider("=")
        print("FINAL REPORT")
        _divider("=")
        print(report)

        if args.json_out:
            out = {"issue": args.issue, "report": report, "messages": messages}
            Path(args.json_out).write_text(json.dumps(out, indent=2))
            print(f"\nSaved to {args.json_out}")
    else:
        run_interactive()


if __name__ == "__main__":
    main()



""""
conda activate system_rca
cd /home/ras/GITHUB/LLM/AgenticAI/system_rca/

python main.py


"""
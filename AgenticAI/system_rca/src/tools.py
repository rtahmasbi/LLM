"""
Diagnostic tool implementations used by the LangGraph agent nodes.

Each tool is a plain Python function that returns a string result
(success output or an error message). The OpenAI tool schemas are
defined at the bottom of this file and imported by the agent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .guardrails import validate_command, validate_path, is_command_available

########## Safe subprocess runner

def _run(args: list[str], timeout: int = 15) -> str:
    """Run a subprocess and return combined stdout+stderr, truncated to 8 KB."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        # Truncate to avoid flooding the LLM context
        if len(output) > 8192:
            output = output[:8192] + "\n... [output truncated]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {timeout}s: {' '.join(args)}"
    except FileNotFoundError:
        return f"[ERROR] Binary not found: {args[0]}"
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] {exc}"


########## Individual tool functions

def run_command(command: str) -> str:
    """
    Run an arbitrary diagnostic shell command after guardrail validation.
    The command is split on whitespace — no shell expansion is performed.
    """
    ok, reason = validate_command(command)
    if not ok:
        return f"[BLOCKED] {reason}"

    args = command.strip().split()
    return _run(args)


def list_processes(sort_by: str = "cpu", top_n: int = 20) -> str:
    """
    List running processes sorted by cpu or mem usage.
    Wraps `ps aux --sort=...`
    """
    if not is_command_available("ps"):
        return "[ERROR] 'ps' is not installed."

    sort_flag = {
        "cpu": "-pcpu",
        "mem": "-pmem",
        "pid": "pid",
    }.get(sort_by.lower(), "-pcpu")

    args = ["ps", "aux", f"--sort={sort_flag}"]
    output = _run(args)
    lines = output.splitlines()
    # Keep header + top_n data rows
    return "\n".join(lines[: top_n + 1])


def check_memory() -> str:
    """Return memory usage summary via `free -h`."""
    if not is_command_available("free"):
        return "[ERROR] 'free' is not installed."
    return _run(["free", "-h"])


def check_disk() -> str:
    """Return disk usage via `df -h`."""
    if not is_command_available("df"):
        return "[ERROR] 'df' is not installed."
    return _run(["df", "-h", "--exclude-type=tmpfs", "--exclude-type=devtmpfs"])


def check_cpu_info() -> str:
    """Return CPU info via lscpu."""
    if is_command_available("lscpu"):
        return _run(["lscpu"])
    # Fallback: read /proc/cpuinfo header
    ok, reason = validate_path("/proc/cpuinfo")
    if not ok:
        return f"[BLOCKED] {reason}"
    return _run(["head", "-40", "/proc/cpuinfo"])


def check_system_load() -> str:
    """Return load averages and uptime."""
    if not is_command_available("uptime"):
        return "[ERROR] 'uptime' is not installed."
    uptime_out = _run(["uptime"])

    vmstat_out = ""
    if is_command_available("vmstat"):
        vmstat_out = "\n\nvmstat (1 sample):\n" + _run(["vmstat", "1", "2"])

    return uptime_out + vmstat_out


def check_network() -> str:
    """Return network socket summary via `ss -tulnp`."""
    if is_command_available("ss"):
        return _run(["ss", "-tulnp"])
    if is_command_available("netstat"):
        return _run(["netstat", "-tulnp"])
    return "[ERROR] Neither 'ss' nor 'netstat' is installed."


def read_journal_logs(
    unit: str = "",
    lines: int = 100,
    priority: str = "warning",
) -> str:
    """
    Read systemd journal logs.
    priority: emerg|alert|crit|err|warning|notice|info|debug
    """
    if not is_command_available("journalctl"):
        return "[ERROR] 'journalctl' is not installed (systemd not present?)."

    valid_priorities = {
        "emerg", "alert", "crit", "err",
        "warning", "notice", "info", "debug",
    }
    if priority not in valid_priorities:
        priority = "warning"

    args = ["journalctl", "-n", str(lines), f"--priority={priority}", "--no-pager"]
    if unit:
        # Sanitise unit name
        unit_clean = re.sub(r"[^a-zA-Z0-9@._\-:]", "", unit)
        args += ["-u", unit_clean]

    return _run(args)


def read_dmesg(lines: int = 80) -> str:
    """Return recent kernel ring-buffer messages.

    Falls back to journalctl -k or /var/log/kern.log when the process
    lacks CAP_SYSLOG / dmesg_restrict blocks access.
    """
    if not is_command_available("dmesg"):
        return "[UNAVAILABLE] 'dmesg' is not installed."

    output = _run(["dmesg", "--time-format=reltime", "-T"])

    permission_phrases = (
        "operation not permitted",
        "permission denied",
        "read kernel buffer failed",
        "klogctl",
    )
    if any(p in output.lower() for p in permission_phrases):
        note = (
            "[PERMISSION DENIED] dmesg requires elevated privileges on this system.\n"
            "  kernel.dmesg_restrict=1 is likely set.\n"
            "  To allow unprivileged access: sudo sysctl -w kernel.dmesg_restrict=0\n\n"
        )

        # Fallback 1: journalctl -k (kernel messages via systemd journal)
        if is_command_available("journalctl"):
            fallback = _run(
                ["journalctl", "-k", "-n", str(lines), "--no-pager",
                 "--output=short-precise"]
            )
            if fallback and not any(p in fallback.lower() for p in permission_phrases):
                return note + "Fallback via journalctl -k:\n" + fallback

        # Fallback 2: /var/log/kern.log
        kern_log = "/var/log/kern.log"
        if Path(kern_log).exists() and is_command_available("tail"):
            fallback = _run(["tail", "-n", str(lines), kern_log])
            if fallback:
                return note + f"Fallback via {kern_log}:\n" + fallback

        return (
            note +
            "[NO FALLBACK AVAILABLE] Neither journalctl -k nor /var/log/kern.log "
            "could be read. The agent should proceed without kernel log data."
        )

    tail = "\n".join(output.splitlines()[-lines:])
    return tail or "(no dmesg output)"


def read_log_file(path: str, tail_lines: int = 100) -> str:
    """
    Read the tail of a log file.
    Only safe paths under /var/log are permitted.
    """
    ok, reason = validate_path(path)
    if not ok:
        return f"[BLOCKED] {reason}"

    resolved = str(Path(path).resolve())
    # Extra restriction: must be under /var/log
    if not resolved.startswith("/var/log"):
        return f"[BLOCKED] read_log_file only allows paths under /var/log. Got: {resolved}"

    if not is_command_available("tail"):
        return "[ERROR] 'tail' is not installed."

    return _run(["tail", "-n", str(tail_lines), resolved])


def check_service_status(service_name: str) -> str:
    """Return the status of a systemd service (read-only)."""
    if not is_command_available("systemctl"):
        return "[ERROR] 'systemctl' not available."

    service_clean = re.sub(r"[^a-zA-Z0-9@._\-]", "", service_name)
    if not service_clean:
        return "[ERROR] Invalid service name."

    return _run(["systemctl", "status", "--no-pager", service_clean])


def find_open_files(pid: int | str = "") -> str:
    """
    List open files for a PID (or all processes if pid="").
    Uses lsof.
    """
    if not is_command_available("lsof"):
        return "[ERROR] 'lsof' is not installed."

    args = ["lsof"]
    if pid:
        try:
            pid_int = int(pid)
        except ValueError:
            return f"[ERROR] Invalid PID: {pid}"
        args += ["-p", str(pid_int)]
    else:
        args += ["-n", "-P"]

    output = _run(args, timeout=20)
    lines = output.splitlines()
    return "\n".join(lines[:200])  # cap lines


def run_perf_stat(pid: int | str = "", duration: int = 3) -> str:
    """
    Run `perf stat` on a PID or system-wide for a short duration.
    Requires perf to be installed and CAP_PERFMON (or perf_event_paranoid <= 1).
    """
    if not is_command_available("perf"):
        return "[ERROR] 'perf' is not installed. Install linux-tools-$(uname -r)."

    duration = min(max(int(duration), 1), 10)  # clamp 1-10 s

    if pid:
        try:
            pid_int = int(pid)
        except ValueError:
            return f"[ERROR] Invalid PID: {pid}"
        args = ["perf", "stat", "-p", str(pid_int), "sleep", str(duration)]
    else:
        args = ["perf", "stat", "-a", "sleep", str(duration)]

    return _run(args, timeout=duration + 10)


########## OpenAI Tool Schemas

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a single read-only diagnostic shell command (e.g. ps, grep, "
                "cat /var/log/...). The command is validated against a security "
                "policy before execution. Do NOT use shell operators (|, ;, &, $)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The full command to run, e.g. 'ps aux' or 'cat /var/log/syslog'.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_processes",
            "description": "List running processes sorted by CPU or memory usage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "enum": ["cpu", "mem", "pid"],
                        "description": "Sort field.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top processes to return (default 20).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_memory",
            "description": "Return system memory and swap usage.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_disk",
            "description": "Return disk space usage for all mounted filesystems.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_cpu_info",
            "description": "Return detailed CPU model, cores, and frequency information.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_system_load",
            "description": "Return system load averages and a vmstat sample.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_network",
            "description": "Return open network sockets and listening ports.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_journal_logs",
            "description": (
                "Read systemd journal logs, optionally filtered by service unit "
                "and minimum priority level."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "unit": {
                        "type": "string",
                        "description": "Systemd unit name, e.g. 'nginx.service'. Leave empty for all units.",
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Maximum log lines to return (default 100).",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["emerg", "alert", "crit", "err", "warning", "notice", "info", "debug"],
                        "description": "Minimum log priority to include (default 'warning').",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_dmesg",
            "description": "Return recent kernel ring-buffer messages (hardware errors, OOM, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lines": {
                        "type": "integer",
                        "description": "Number of recent lines to return (default 80).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_log_file",
            "description": (
                "Read the tail of a specific log file. "
                "Only files under /var/log are permitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the log file, e.g. '/var/log/syslog'.",
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "Number of tail lines to return (default 100).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_service_status",
            "description": "Return the systemctl status of a named service (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Systemd service name, e.g. 'nginx' or 'postgresql.service'.",
                    }
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_open_files",
            "description": "List open file descriptors for a process (or all processes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "string",
                        "description": "PID to inspect. Leave empty to list all processes.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_perf_stat",
            "description": (
                "Run 'perf stat' to collect CPU performance counters. "
                "Requires perf to be installed and appropriate kernel permissions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "string",
                        "description": "PID to profile. Leave empty for system-wide profiling.",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Sampling duration in seconds (1–10, default 3).",
                    },
                },
            },
        },
    },
]

# Map function name → callable
TOOL_REGISTRY: dict[str, Any] = {
    "run_command": run_command,
    "list_processes": list_processes,
    "check_memory": check_memory,
    "check_disk": check_disk,
    "check_cpu_info": check_cpu_info,
    "check_system_load": check_system_load,
    "check_network": check_network,
    "read_journal_logs": read_journal_logs,
    "read_dmesg": read_dmesg,
    "read_log_file": read_log_file,
    "check_service_status": check_service_status,
    "find_open_files": find_open_files,
    "run_perf_stat": run_perf_stat,
}

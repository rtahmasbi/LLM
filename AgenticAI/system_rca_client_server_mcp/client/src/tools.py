from __future__ import annotations

import os
import shlex
import subprocess
from typing import Callable


def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (result.stdout or "")
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return output[:20000] or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {timeout}s"
    except Exception as exc:
        return f"[ERROR] Failed to run {' '.join(cmd)}: {exc}"


def _safe_shell(command: str) -> str:
    blocked = {"|", ";", "&", "$", "`", ">", "<"}
    if any(token in command for token in blocked):
        return "[BLOCKED] Shell operators are not allowed."
    return command


def run_command(command: str) -> str:
    """Run a single read-only diagnostic shell command. No shell operators are allowed."""
    checked = _safe_shell(command)
    if checked.startswith("[BLOCKED]"):
        return checked
    return _run(shlex.split(checked))


def list_processes(sort_by: str = "cpu", top_n: int = 20) -> str:
    """List running processes sorted by CPU, memory, or PID."""
    sort_map = {
        "cpu": "-%cpu",
        "mem": "-%mem",
        "pid": "pid",
    }
    key = sort_map.get(sort_by, "-%cpu")
    cmd = ["sh", "-lc", f"ps aux --sort={key} | head -n {int(top_n) + 1}"]
    return _run(cmd)


def check_memory() -> str:
    """Return system memory and swap usage."""
    return _run(["free", "-h"])


def check_disk() -> str:
    """Return disk space usage for all mounted filesystems."""
    return _run(["df", "-h"])


def check_cpu_info() -> str:
    """Return CPU model, core count, and frequency."""
    return _run(["lscpu"])


def check_system_load() -> str:
    """Return load averages and a short vmstat sample."""
    return _run(["sh", "-lc", "uptime && echo && vmstat 1 3"])


def check_network() -> str:
    """Return open sockets and listening ports."""
    return _run(["ss", "-tulnp"])


def read_journal_logs(unit: str = "", lines: int = 100, priority: str = "warning") -> str:
    """Read systemd journal logs filtered by unit and priority."""
    cmd = ["journalctl", "-n", str(int(lines)), "-p", priority, "--no-pager"]
    if unit:
        cmd.extend(["-u", unit])
    return _run(cmd)


def read_dmesg(lines: int = 100) -> str:
    """Return recent kernel ring-buffer messages."""
    return _run(["sh", "-lc", f"dmesg | tail -n {int(lines)}"])


def read_log_file(path: str, tail_lines: int = 100) -> str:
    """Read the tail of a log file under /var/log."""
    real_path = os.path.realpath(path)
    if not real_path.startswith("/var/log/"):
        return "[BLOCKED] Only files under /var/log are allowed."
    return _run(["tail", "-n", str(int(tail_lines)), real_path])


def check_service_status(service_name: str) -> str:
    """Return systemctl status for a named service."""
    return _run(["systemctl", "status", service_name, "--no-pager"])


def find_open_files(pid: str = "") -> str:
    """List open file descriptors for a PID or, if omitted, all processes."""
    cmd = ["lsof"]
    if pid:
        cmd.extend(["-p", str(pid)])
    return _run(cmd)


def run_perf_stat(pid: str = "", duration: int = 3) -> str:
    """Run perf stat to collect CPU performance counters."""
    if pid:
        cmd = ["perf", "stat", "-p", str(pid), "sleep", str(int(duration))]
    else:
        cmd = ["perf", "stat", "sleep", str(int(duration))]
    return _run(cmd, timeout=max(10, int(duration) + 5))


def run_ping() -> str:
    """Check if the system can reach the internet."""
    return _run(["ping", "-c", "1", "8.8.8.8"], timeout=10)


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
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
    "run_ping": run_ping,
}

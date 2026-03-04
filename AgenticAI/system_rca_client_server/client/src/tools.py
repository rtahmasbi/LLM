"""
client/src/tools.py — local tool executor (client side)

Receives a ToolCallRequest from the server, runs the corresponding
diagnostic function on *this* machine, and returns a ToolResultRequest.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .guardrails import (
    validate_command, validate_path,
    is_command_available, ALLOWED_COMMANDS,
)

TRUNCATE = 8192  # max output bytes sent back to server


def _run(args: list[str], timeout: int = 15) -> str:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).strip()
        if len(out) > TRUNCATE:
            out = out[:TRUNCATE] + "\n... [truncated]"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Timed out after {timeout}s: {' '.join(args)}"
    except FileNotFoundError:
        return f"[ERROR] Binary not found: {args[0]}"
    except Exception as exc:
        return f"[ERROR] {exc}"


###### Tool implementations

def run_command(command: str) -> str:
    ok, reason = validate_command(command)
    if not ok:
        return f"[BLOCKED] {reason}"
    return _run(command.strip().split())


def list_processes(sort_by: str = "cpu", top_n: int = 20) -> str:
    if not is_command_available("ps"):
        return "[ERROR] 'ps' not installed."
    sort_flag = {"cpu": "-pcpu", "mem": "-pmem", "pid": "pid"}.get(sort_by.lower(), "-pcpu")
    lines = _run(["ps", "aux", f"--sort={sort_flag}"]).splitlines()
    return "\n".join(lines[: top_n + 1])


def check_memory() -> str:
    if not is_command_available("free"):
        return "[ERROR] 'free' not installed."
    return _run(["free", "-h"])


def check_disk() -> str:
    if not is_command_available("df"):
        return "[ERROR] 'df' not installed."
    return _run(["df", "-h", "--exclude-type=tmpfs", "--exclude-type=devtmpfs"])


def check_cpu_info() -> str:
    if is_command_available("lscpu"):
        return _run(["lscpu"])
    ok, reason = validate_path("/proc/cpuinfo")
    if not ok:
        return f"[BLOCKED] {reason}"
    return _run(["head", "-40", "/proc/cpuinfo"])


def check_system_load() -> str:
    out = _run(["uptime"]) if is_command_available("uptime") else "[ERROR] uptime missing"
    if is_command_available("vmstat"):
        out += "\n\nvmstat:\n" + _run(["vmstat", "1", "2"])
    return out


def check_network() -> str:
    if is_command_available("ss"):
        return _run(["ss", "-tulnp"])
    if is_command_available("netstat"):
        return _run(["netstat", "-tulnp"])
    return "[ERROR] Neither ss nor netstat installed."


def read_journal_logs(unit: str = "", lines: int = 100, priority: str = "warning") -> str:
    if not is_command_available("journalctl"):
        return "[ERROR] journalctl not available."
    valid = {"emerg","alert","crit","err","warning","notice","info","debug"}
    if priority not in valid:
        priority = "warning"
    args = ["journalctl", "-n", str(lines), f"--priority={priority}", "--no-pager"]
    if unit:
        import re
        unit = re.sub(r"[^a-zA-Z0-9@._\-:]", "", unit)
        args += ["-u", unit]
    return _run(args)


def read_dmesg(lines: int = 80) -> str:
    if not is_command_available("dmesg"):
        return "[UNAVAILABLE] dmesg not installed."

    output = _run(["dmesg", "--time-format=reltime", "-T"])

    permission_phrases = (
        "operation not permitted", "permission denied",
        "read kernel buffer failed", "klogctl",
    )
    if any(p in output.lower() for p in permission_phrases):
        note = (
            "[PERMISSION DENIED] dmesg requires elevated privileges.\n"
            "  kernel.dmesg_restrict=1 is likely set.\n"
            "  Fix: sudo sysctl -w kernel.dmesg_restrict=0\n\n"
        )
        if is_command_available("journalctl"):
            fb = _run(["journalctl", "-k", "-n", str(lines),
                       "--no-pager", "--output=short-precise"])
            if fb and not any(p in fb.lower() for p in permission_phrases):
                return note + "Fallback via journalctl -k:\n" + fb
        kern = "/var/log/kern.log"
        if Path(kern).exists() and is_command_available("tail"):
            return note + f"Fallback via {kern}:\n" + _run(["tail", "-n", str(lines), kern])
        return note + "[NO FALLBACK] Proceed without kernel log data."

    return "\n".join(output.splitlines()[-lines:]) or "(no dmesg output)"


def read_log_file(path: str, tail_lines: int = 100) -> str:
    ok, reason = validate_path(path)
    if not ok:
        return f"[BLOCKED] {reason}"
    resolved = str(Path(path).resolve())
    if not resolved.startswith("/var/log"):
        return f"[BLOCKED] read_log_file only allows /var/log. Got: {resolved}"
    if not is_command_available("tail"):
        return "[ERROR] tail not installed."
    return _run(["tail", "-n", str(tail_lines), resolved])


def check_service_status(service_name: str) -> str:
    if not is_command_available("systemctl"):
        return "[ERROR] systemctl not available."
    import re
    svc = re.sub(r"[^a-zA-Z0-9@._\-]", "", service_name)
    return _run(["systemctl", "status", "--no-pager", svc])


def find_open_files(pid: str = "") -> str:
    if not is_command_available("lsof"):
        return "[ERROR] lsof not installed."
    args = ["lsof"]
    if pid:
        try:
            args += ["-p", str(int(pid))]
        except ValueError:
            return f"[ERROR] Invalid PID: {pid}"
    else:
        args += ["-n", "-P"]
    return "\n".join(_run(args, timeout=20).splitlines()[:200])


def run_perf_stat(pid: str = "", duration: int = 3) -> str:
    if not is_command_available("perf"):
        return "[ERROR] perf not installed. Try: apt install linux-tools-$(uname -r)"
    duration = min(max(int(duration), 1), 10)
    if pid:
        try:
            args = ["perf", "stat", "-p", str(int(pid)), "sleep", str(duration)]
        except ValueError:
            return f"[ERROR] Invalid PID: {pid}"
    else:
        args = ["perf", "stat", "-a", "sleep", str(duration)]
    return _run(args, timeout=duration + 10)


###### Dispatch table

TOOL_REGISTRY: dict = {
    "run_command":        run_command,
    "list_processes":     list_processes,
    "check_memory":       check_memory,
    "check_disk":         check_disk,
    "check_cpu_info":     check_cpu_info,
    "check_system_load":  check_system_load,
    "check_network":      check_network,
    "read_journal_logs":  read_journal_logs,
    "read_dmesg":         read_dmesg,
    "read_log_file":      read_log_file,
    "check_service_status": check_service_status,
    "find_open_files":    find_open_files,
    "run_perf_stat":      run_perf_stat,
}

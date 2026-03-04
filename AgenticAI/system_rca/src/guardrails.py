"""
Guardrails module — defines what commands and paths are allowed or blocked.
"""

import re
import shutil
from pathlib import Path

################ Blocked directories
BLOCKED_DIRECTORIES: set[str] = {
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/ssh",
    "/root",
    "/home",          # individual user home dirs
    "/proc/tty",
    "/sys/firmware",
    "/boot",
    "/dev",
    "/run/secrets",
    "/var/lib/docker/volumes",
    "/snap",
}

# Path prefixes that are globally off-limits
BLOCKED_PATH_PREFIXES: tuple[str, ...] = (
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/ssl/private",
    "/root",
    "/home/",
    "/boot",
    "/dev/",
    "/sys/firmware",
    "/run/secrets",
)

################ Blocked / sensitive commands
# Any command whose base name (argv[0]) matches one of these is refused.
BLOCKED_COMMANDS: set[str] = {
    # Privilege escalation
    "sudo", "su", "doas", "pkexec",
    # Destructive / data-altering
    "rm", "rmdir", "shred", "dd", "mkfs", "fdisk", "parted",
    "wipefs", "mkswap", "format",
    # Network modification
    "iptables", "ip6tables", "nftables", "ufw", "firewall-cmd",
    "tc", "modprobe", "insmod", "rmmod",
    # User / credential management
    "passwd", "chpasswd", "useradd", "userdel", "usermod",
    "groupadd", "groupdel", "chown", "chmod",
    # Package managers (could install/remove software)
    "apt", "apt-get", "dpkg", "yum", "dnf", "rpm",
    "pacman", "snap", "flatpak", "brew",
    # Shell invocations
    "bash", "sh", "zsh", "fish", "ksh", "csh", "tcsh",
    # Crypto / key operations
    "gpg", "openssl", "ssh-keygen", "ssh-add",
    # Curl/wget (data exfil risk)
    "curl", "wget", "nc", "netcat", "ncat",
    # Editors (could overwrite files)
    "vi", "vim", "nano", "emacs", "tee",
    # Reboot / shutdown
    "reboot", "shutdown", "halt", "poweroff", "init",
    # Cron / scheduler mutation
    "crontab",
    # Kill (blanket; we allow a restricted version via a dedicated tool)
    "killall", "pkill",
    # Strace on arbitrary processes (privacy)
    # "strace",  — allowed for diagnostic PIDs only; see safe_commands
}

# ── Shell meta-characters that should never appear in a command ──────────────
DANGEROUS_SHELL_PATTERNS: list[re.Pattern] = [
    re.compile(r"[;&|`$]"),      # shell chaining / substitution
    re.compile(r"\beval\b"),
    re.compile(r"\bexec\b"),
    re.compile(r">\s*/"),        # redirect to root paths
    re.compile(r"<\("),          # process substitution
]

# ── Approved diagnostic commands ─────────────────────────────────────────────
# Entries are the base binary name; arguments are validated separately.
ALLOWED_COMMANDS: set[str] = {
    # Process inspection
    "ps", "pstree", "top", "htop", "atop", "pidof", "pgrep",
    # CPU / memory perf
    "perf", "vmstat", "mpstat", "iostat", "sar", "sysstat",
    "free", "numastat",
    # Disk
    "df", "du", "lsblk", "blkid", "findmnt", "iostat",
    "smartctl",
    # Network (read-only)
    "ss", "netstat", "ip", "ifconfig", "ping", "traceroute",
    "nslookup", "dig", "host", "lsof",
    # Logs
    "journalctl", "dmesg", "last", "lastlog", "who", "w",
    "uptime",
    # System info
    "uname", "hostname", "lscpu", "lsmem", "lshw", "lspci",
    "lsusb", "dmidecode", "inxi",
    # File inspection (read-only)
    "cat", "less", "head", "tail", "grep", "awk", "sed",
    "find", "stat", "file", "wc", "sort", "uniq", "cut",
    # Misc safe tools
    "date", "timedatectl", "systemctl",   # status/show only
    "service",                             # status only
    "env", "printenv",
    "id", "whoami", "groups",
    "mount",                               # read-only listing
    "ulimit",
}


class GuardrailViolation(Exception):
    """Raised when a command or path violates security policy."""


def _base_command(command_str: str) -> str:
    """Extract the leading binary name from a command string."""
    return command_str.strip().split()[0] if command_str.strip() else ""


def validate_command(command: str) -> tuple[bool, str]:
    """
    Return (True, "") if the command passes all guardrails.
    Return (False, reason) if it is blocked.
    """
    stripped = command.strip()

    # 1. Empty command
    if not stripped:
        return False, "Empty command."

    # 2. Shell meta-character injection
    for pattern in DANGEROUS_SHELL_PATTERNS:
        if pattern.search(stripped):
            return False, (
                f"Command contains forbidden shell metacharacter "
                f"matching pattern '{pattern.pattern}'."
            )

    base = _base_command(stripped)

    # 3. Explicitly blocked commands
    if base in BLOCKED_COMMANDS:
        return False, f"Command '{base}' is in the blocked-commands list."

    # 4. Must be in the allowed list
    if base not in ALLOWED_COMMANDS:
        return False, (
            f"Command '{base}' is not in the approved diagnostic commands list."
        )

    # 5. Check that the binary is actually installed
    if not shutil.which(base):
        return False, f"Command '{base}' is not installed on this system."

    # 6. Argument-level path checks
    tokens = stripped.split()
    for token in tokens[1:]:
        ok, reason = validate_path(token)
        if not ok:
            return False, reason

    return True, ""


def validate_path(path_str: str) -> tuple[bool, str]:
    """
    Return (True, "") if the path is allowed.
    Return (False, reason) if it touches a blocked directory.
    """
    # Only validate if it looks like an absolute path
    if not path_str.startswith("/"):
        return True, ""

    try:
        resolved = str(Path(path_str).resolve())
    except Exception:
        resolved = path_str

    for prefix in BLOCKED_PATH_PREFIXES:
        if resolved.startswith(prefix):
            return False, (
                f"Access to path '{resolved}' is blocked "
                f"(restricted prefix: '{prefix}')."
            )

    return True, ""


def is_command_available(command: str) -> bool:
    """Return True if the binary exists on PATH."""
    return shutil.which(_base_command(command)) is not None

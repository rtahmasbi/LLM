"""
Guardrails module — defines what commands and paths are allowed or blocked.
Runs on the CLIENT to enforce security before any subprocess is spawned.
"""

import re
import shutil
from pathlib import Path

###### Blocked directories
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

###### Blocked / sensitive commands
BLOCKED_COMMANDS: set[str] = {
    "sudo", "su", "doas", "pkexec",
    "rm", "rmdir", "shred", "dd", "mkfs", "fdisk", "parted",
    "wipefs", "mkswap", "format",
    "iptables", "ip6tables", "nftables", "ufw", "firewall-cmd",
    "tc", "modprobe", "insmod", "rmmod",
    "passwd", "chpasswd", "useradd", "userdel", "usermod",
    "groupadd", "groupdel", "chown", "chmod",
    "apt", "apt-get", "dpkg", "yum", "dnf", "rpm",
    "pacman", "snap", "flatpak", "brew",
    "bash", "sh", "zsh", "fish", "ksh", "csh", "tcsh",
    "gpg", "openssl", "ssh-keygen", "ssh-add",
    "curl", "wget", "nc", "netcat", "ncat",
    "vi", "vim", "nano", "emacs", "tee",
    "reboot", "shutdown", "halt", "poweroff", "init",
    "crontab",
    "killall", "pkill",
}

###### Shell meta-characters
DANGEROUS_SHELL_PATTERNS: list[re.Pattern] = [
    re.compile(r"[;&|`$]"),
    re.compile(r"\beval\b"),
    re.compile(r"\bexec\b"),
    re.compile(r">\s*/"),
    re.compile(r"<\("),
]

###### Approved diagnostic commands
ALLOWED_COMMANDS: set[str] = {
    "ps", "pstree", "top", "htop", "atop", "pidof", "pgrep",
    "perf", "vmstat", "mpstat", "iostat", "sar", "sysstat",
    "free", "numastat",
    "df", "du", "lsblk", "blkid", "findmnt", "smartctl",
    "ss", "netstat", "ip", "ifconfig", "ping", "traceroute",
    "nslookup", "dig", "host", "lsof",
    "journalctl", "dmesg", "last", "lastlog", "who", "w", "uptime",
    "uname", "hostname", "lscpu", "lsmem", "lshw", "lspci",
    "lsusb", "dmidecode", "inxi",
    "cat", "less", "head", "tail", "grep", "awk", "sed",
    "find", "stat", "file", "wc", "sort", "uniq", "cut",
    "date", "timedatectl", "systemctl", "service",
    "env", "printenv", "id", "whoami", "groups",
    "mount", "ulimit",
}


class GuardrailViolation(Exception):
    pass


def _base(cmd: str) -> str:
    return cmd.strip().split()[0] if cmd.strip() else ""


def validate_command(command: str) -> tuple[bool, str]:
    stripped = command.strip()
    if not stripped:
        return False, "Empty command."
    for pat in DANGEROUS_SHELL_PATTERNS:
        if pat.search(stripped):
            return False, f"Forbidden shell metacharacter: '{pat.pattern}'."
    base = _base(stripped)
    if base in BLOCKED_COMMANDS:
        return False, f"'{base}' is in the blocked-commands list."
    if base not in ALLOWED_COMMANDS:
        return False, f"'{base}' is not in the approved diagnostic commands list."
    if not shutil.which(base):
        return False, f"'{base}' is not installed on this system."
    for token in stripped.split()[1:]:
        ok, reason = validate_path(token)
        if not ok:
            return False, reason
    return True, ""


def validate_path(path_str: str) -> tuple[bool, str]:
    if not path_str.startswith("/"):
        return True, ""
    try:
        resolved = str(Path(path_str).resolve())
    except Exception:
        resolved = path_str
    for prefix in BLOCKED_PATH_PREFIXES:
        if resolved.startswith(prefix):
            return False, f"Access to '{resolved}' is blocked (prefix: '{prefix}')."
    return True, ""


def is_command_available(command: str) -> bool:
    return shutil.which(_base(command)) is not None

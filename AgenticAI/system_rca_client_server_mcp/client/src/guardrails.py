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


def validate_command(cmd: list[str]) -> str | None:
    """
    Validate a command list before execution.
    Returns an error string if blocked, or None if allowed.
    """
    if not cmd:
        return "[BLOCKED] Empty command."

    binary = Path(cmd[0]).name  # strip path prefix, e.g. /usr/bin/ps → ps

    if binary in BLOCKED_COMMANDS:
        return f"[BLOCKED] Command '{binary}' is not permitted."

    if binary not in ALLOWED_COMMANDS:
        return f"[BLOCKED] Command '{binary}' is not in the allowed list."

    return None


def validate_shell_string(command: str) -> str | None:
    """
    Validate a raw shell string for dangerous patterns.
    Returns an error string if blocked, or None if allowed.
    """
    for pattern in DANGEROUS_SHELL_PATTERNS:
        if pattern.search(command):
            return f"[BLOCKED] Dangerous pattern detected: '{pattern.pattern}'"

    tokens = command.split()
    if tokens:
        binary = Path(tokens[0]).name
        if binary in BLOCKED_COMMANDS:
            return f"[BLOCKED] Command '{binary}' is not permitted."
        if binary not in ALLOWED_COMMANDS:
            return f"[BLOCKED] Command '{binary}' is not in the allowed list."

    return None


def validate_path(path: str) -> str | None:
    """
    Validate a file path against blocked prefixes.
    Returns an error string if blocked, or None if allowed.
    """
    real = str(Path(path).resolve())
    for prefix in BLOCKED_PATH_PREFIXES:
        if real.startswith(prefix):
            return f"[BLOCKED] Access to '{prefix}' is not permitted."
    return None

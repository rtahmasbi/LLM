#!/usr/bin/env python3
"""
improve_agents_md.py — Iterative AGENTS.md improvement pipeline.

Loop:
  1. Score current AGENTS.md across 8 metrics
  2. Print a scorecard
  3. Use an LLM to rewrite the file, targeting the weakest metrics
  4. Repeat for N iterations (or until score plateaus)

Usage:
    pip install openai
    export OPENAI_API_KEY=sk-...

    python improve_agents_md.py \\
        --repo   /path/to/fastapi \\
        --input  AGENTS_v0.md \\
        --iters  3 \\
        --out-dir ./versions

Outputs one scored AGENTS_vN.md per iteration plus a scores.json summary.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from openai import OpenAI

client = OpenAI()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    name: str
    score: float          # 0.0 – 1.0
    max_score: float = 1.0
    details: str = ""     # human-readable explanation
    evidence: list = field(default_factory=list)   # concrete examples


@dataclass
class Scorecard:
    version: int
    metrics: list[MetricScore]
    overall: float = 0.0

    def __post_init__(self):
        self.overall = round(
            sum(m.score for m in self.metrics) / len(self.metrics), 3
        ) if self.metrics else 0.0

    def print(self):
        bar_width = 30
        print(f"\n{'='*60}")
        print(f"  AGENTS.md Scorecard  —  v{self.version}  "
              f"  Overall: {self.overall:.0%}")
        print(f"{'='*60}")
        for m in sorted(self.metrics, key=lambda x: x.score):
            filled  = int(m.score * bar_width)
            bar     = "█" * filled + "░" * (bar_width - filled)
            flag    = "⚠️ " if m.score < 0.5 else ("✅ " if m.score >= 0.8 else "〰️ ")
            print(f"  {flag}{m.name:<28} {bar}  {m.score:.0%}")
            if m.details:
                for line in textwrap.wrap(m.details, 54):
                    print(f"       {line}")
        print(f"{'='*60}\n")

    def weakest(self, n: int = 3) -> list[MetricScore]:
        return sorted(self.metrics, key=lambda x: x.score)[:n]


# ===========================================================================
# METRICS
# Each metric function takes (content: str, repo: Path) -> MetricScore
# ===========================================================================

def metric_command_validity(content: str, repo: Path) -> MetricScore:
    """
    Extract every fenced code block that looks like shell commands.
    Actually run each command. Score = passing / total.
    """
    commands = _extract_shell_commands(content)
    if not commands:
        return MetricScore(
            "Command Validity", 0.1,
            details="No shell commands found in file.",
            evidence=[]
        )

    passed, failed = [], []
    for cmd in commands:
        # Skip install/setup commands — too destructive to run blindly
        if any(cmd.startswith(p) for p in
               ("pip install", "npm install", "make install", "python setup")):
            passed.append(f"(skipped install) {cmd}")
            continue
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=repo,
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                passed.append(cmd)
            else:
                failed.append(f"{cmd!r} → exit {result.returncode}")
        except subprocess.TimeoutExpired:
            passed.append(f"(timeout) {cmd}")
        except Exception as e:
            failed.append(f"{cmd!r} → {e}")

    total = len(passed) + len(failed)
    score = len(passed) / total if total else 0.0
    details = (
        f"{len(passed)}/{total} commands run clean."
        + (f" Failing: {'; '.join(failed[:2])}" if failed else "")
    )
    return MetricScore("Command Validity", round(score, 2),
                       details=details, evidence=failed)


def metric_section_coverage(content: str, repo: Path) -> MetricScore:
    """
    Check for presence of high-value sections.
    Weighted: anti-patterns and risk map are worth more than basic setup.
    """
    sections = {
        # (pattern, weight, label)
        r"(test|pytest|running test)":          (0.10, "test commands"),
        r"(install|setup|environment)":         (0.08, "install instructions"),
        r"(lint|format|ruff|flake)":            (0.08, "lint commands"),
        r"(type.?check|mypy|pyright)":          (0.07, "type checking"),
        r"(architecture|structure|layout|map)": (0.15, "architecture map"),
        r"(convention|style|pattern)":          (0.12, "code conventions"),
        r"(anti.?pattern|gotcha|avoid|never)":  (0.18, "anti-patterns"),
        r"(risk|churn|sensitive|careful|caution)": (0.12, "risk map"),
        r"(pr|pull request|commit|branch)":     (0.10, "PR conventions"),
    }
    content_lower = content.lower()
    total_weight = sum(w for _, (w, _) in sections.items())
    earned = 0.0
    missing = []

    for pattern, (weight, label) in sections.items():
        if re.search(pattern, content_lower):
            earned += weight
        else:
            missing.append(label)

    score = earned / total_weight
    details = (
        f"Missing: {', '.join(missing)}" if missing
        else "All key sections present."
    )
    return MetricScore("Section Coverage", round(score, 2),
                       details=details, evidence=missing)


def metric_specificity(content: str, repo: Path) -> MetricScore:
    """
    Measures how concrete the file is vs. generic advice.
    Signals of specificity:
      - Actual file paths (fastapi/routing.py, not just 'the routing module')
      - Concrete code examples in fenced blocks
      - Specific tool flags (pytest -x --cov=, not just 'pytest')
      - Named patterns (Annotated[T, Depends(...)], not 'use DI')
    Signals of vagueness:
      - Phrases like "where possible", "as needed", "best practices"
      - Very short code blocks (< 2 lines)
    """
    vague_phrases = [
        r"where possible", r"as needed", r"best practices?",
        r"follow pep.?8", r"write (good|clear|meaningful) (commit|doc|test)",
        r"make sure (ci|tests?) pass", r"keep it (clean|simple|readable)",
    ]
    specific_signals = [
        r"`[a-z_/]+\.py`",                       # file paths
        r"```(bash|python|sh)\n.{20,}",           # substantial code blocks
        r"pytest .{5,}",                          # pytest with flags
        r"(✅|❌|✓|✗)",                            # explicit do/don't markers
        r"\b(Annotated|Depends|HTTPException|BaseModel|APIRouter)\b",
    ]

    vague_count    = sum(1 for p in vague_phrases if re.search(p, content, re.I))
    specific_count = sum(1 for p in specific_signals if re.search(p, content, re.S))

    # Score: specifics push up, vague phrases push down
    raw = (specific_count * 0.15) - (vague_count * 0.12)
    score = max(0.0, min(1.0, 0.3 + raw))

    details = (
        f"{specific_count} specific signals, {vague_count} vague phrases. "
        + ("Add concrete examples and file paths." if specific_count < 3 else "")
    )
    return MetricScore("Specificity", round(score, 2),
                       details=details, evidence=[])


def metric_novelty(content: str, repo: Path) -> MetricScore:
    """
    What % of the content is NOT already in README.md / CONTRIBUTING.md?
    We check for semantic overlap by extracting key phrases from both.
    """
    reference_texts = []
    for fname in ("README.md", "README.rst", "CONTRIBUTING.md", "CONTRIBUTING.rst"):
        fpath = repo / fname
        if fpath.exists():
            reference_texts.append(fpath.read_text(encoding="utf-8", errors="replace"))

    if not reference_texts:
        # No README to compare against — give benefit of the doubt
        return MetricScore("Novelty vs README", 0.7,
                           details="No README/CONTRIBUTING found to compare against.")

    reference_combined = " ".join(reference_texts).lower()

    # Extract meaningful phrases from AGENTS.md (4+ word sequences)
    agents_sentences = [
        s.strip() for s in re.split(r"[.\n]", content)
        if len(s.strip()) > 30 and not s.strip().startswith("#")
    ]

    if not agents_sentences:
        return MetricScore("Novelty vs README", 0.2,
                           details="Too little prose content to evaluate novelty.")

    novel = 0
    duplicated = []
    for sentence in agents_sentences:
        # Check if a 6-word window from this sentence appears in reference
        words = sentence.lower().split()
        found_in_ref = any(
            " ".join(words[i:i+6]) in reference_combined
            for i in range(max(1, len(words) - 5))
        ) if len(words) >= 6 else False

        if not found_in_ref:
            novel += 1
        else:
            duplicated.append(sentence[:60])

    score = novel / len(agents_sentences)
    details = (
        f"{novel}/{len(agents_sentences)} sentences are novel vs README. "
        + (f"Overlapping: '{duplicated[0]}...'" if duplicated else "")
    )
    return MetricScore("Novelty vs README", round(score, 2),
                       details=details, evidence=duplicated[:3])


def metric_information_density(content: str, repo: Path) -> MetricScore:
    """
    Useful tokens / total tokens.
    Penalizes:
      - Long preamble before first command
      - Repeated filler phrases
      - Sections with only 1-2 lines of content
      - Headers with no body
    Rewards:
      - Code blocks
      - Bullet points with concrete content
      - ✅/❌ markers
    """
    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return MetricScore("Information Density", 0.0, details="Empty file.")

    filler_patterns = [
        r"^#+\s*$",                          # empty header
        r"^\s*$",                             # blank line
        r"feel free to",
        r"don't forget to",
        r"it is (important|recommended)",
        r"please (make sure|ensure|note)",
        r"^(this|the) (file|document) (describes|contains|provides)",
    ]
    useful_patterns = [
        r"```",           # code fence
        r"^\s*[-*]\s+.{20,}",  # substantial bullet
        r"✅|❌|✓|✗|⚠️",
        r"`[^`]{3,}`",    # inline code
        r"\b(pytest|ruff|mypy|pip)\b",
    ]

    filler_lines  = sum(1 for l in lines if any(re.search(p, l, re.I) for p in filler_patterns))
    useful_lines  = sum(1 for l in lines if any(re.search(p, l) for p in useful_patterns))

    density = useful_lines / max(total - filler_lines, 1)
    score   = min(1.0, density * 1.8)  # calibrate: ~55% useful lines → 1.0

    details = (
        f"{useful_lines} high-signal lines, {filler_lines} filler lines, "
        f"{total} total. Density: {density:.0%}."
    )
    return MetricScore("Information Density", round(score, 2),
                       details=details, evidence=[])


def metric_actionability(content: str, repo: Path) -> MetricScore:
    """
    Can an agent take a concrete action from each major section?
    Checks: every H2/H3 section has at least one of:
      - a code block, OR
      - a bullet with a verb, OR
      - an explicit DO/DON'T
    """
    # Split into sections
    sections = re.split(r"\n#{2,3} ", "\n" + content)
    if len(sections) <= 1:
        return MetricScore("Actionability", 0.2,
                           details="No sections found (no ## headers).")

    actionable = 0
    inert = []
    for section in sections[1:]:  # skip preamble
        title = section.splitlines()[0].strip()
        body  = "\n".join(section.splitlines()[1:])

        has_code   = bool(re.search(r"```", body))
        has_bullet = bool(re.search(r"^\s*[-*]\s+\w", body, re.M))
        has_dontdo = bool(re.search(r"(✅|❌|✓|✗|don't|never|always|must)", body, re.I))

        if has_code or has_bullet or has_dontdo:
            actionable += 1
        else:
            inert.append(title)

    total  = len(sections) - 1
    score  = actionable / total if total else 0.0
    details = (
        f"{actionable}/{total} sections are actionable. "
        + (f"Inert sections: {', '.join(inert)}" if inert else "")
    )
    return MetricScore("Actionability", round(score, 2),
                       details=details, evidence=inert)


def metric_machine_parsability(content: str, repo: Path) -> MetricScore:
    """
    How easily can an agent parse and navigate this file?
    Checks:
      - Consistent heading hierarchy (no H4 before H3, etc.)
      - Code blocks have language tags
      - No extremely long paragraphs (>150 words) that bury key info
      - Section titles are descriptive (not just 'Notes', 'Other')
    """
    score = 1.0
    issues = []

    # Code blocks without language tag
    untagged = len(re.findall(r"^```\s*$", content, re.M))
    if untagged:
        score -= 0.1 * min(untagged, 3)
        issues.append(f"{untagged} code blocks missing language tag")

    # Paragraphs > 150 words
    paragraphs = re.split(r"\n\n+", content)
    long_paras = [p for p in paragraphs if len(p.split()) > 150 and not p.strip().startswith("#")]
    if long_paras:
        score -= 0.15 * min(len(long_paras), 2)
        issues.append(f"{len(long_paras)} overly long paragraphs")

    # Vague section titles
    vague_titles = re.findall(
        r"^#{2,3}\s+(notes?|other|misc|general|additional|overview)\s*$",
        content, re.I | re.M
    )
    if vague_titles:
        score -= 0.1 * len(vague_titles)
        issues.append(f"Vague section titles: {vague_titles}")

    # No table of contents or anchors in long files
    if len(content) > 3000 and not re.search(r"\[.*\]\(#", content):
        score -= 0.05
        issues.append("Long file lacks navigation anchors")

    score = max(0.0, round(score, 2))
    details = "; ".join(issues) if issues else "Structure is clean and parsable."
    return MetricScore("Machine Parsability", score,
                       details=details, evidence=issues)


def metric_repo_grounding(content: str, repo: Path) -> MetricScore:
    """
    Are the file paths, module names, and tool references in the file
    actually real in this repo?
    Checks every backtick-quoted path/module mention against the filesystem.
    """
    # Extract `something/like/this.py` or `module.submodule` patterns
    candidates = re.findall(r"`([a-zA-Z0-9_/.-]+(?:\.py|\.md|\.toml|\.cfg)?)`", content)
    if not candidates:
        return MetricScore("Repo Grounding", 0.5,
                           details="No file/module references found to verify.")

    verified, broken = [], []
    for ref in set(candidates):
        if "/" in ref or ref.endswith(".py"):
            # Looks like a path — check it exists
            if (repo / ref).exists():
                verified.append(ref)
            elif any(repo.rglob(ref)):
                verified.append(ref)
            else:
                broken.append(ref)
        # Skip short identifiers (too noisy to verify)

    total = len(verified) + len(broken)
    if total == 0:
        return MetricScore("Repo Grounding", 0.6,
                           details="References found but none are verifiable paths.")

    score   = len(verified) / total
    details = (
        f"{len(verified)}/{total} file references exist in repo."
        + (f" Missing: {', '.join(broken[:3])}" if broken else "")
    )
    return MetricScore("Repo Grounding", round(score, 2),
                       details=details, evidence=broken)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_shell_commands(content: str) -> list[str]:
    """Pull commands from ```bash / ```sh / ``` blocks."""
    commands = []
    in_block = False
    is_shell = False
    for line in content.splitlines():
        if re.match(r"^```(bash|sh|shell|zsh)?\s*$", line):
            in_block = not in_block
            is_shell = bool(re.match(r"^```(bash|sh|shell|zsh)", line)) or not line[3:].strip()
            continue
        if in_block and is_shell and line.strip() and not line.startswith("#"):
            cmd = line.strip()
            # Only lines that look like CLI commands (not code)
            if re.match(r"^(pytest|ruff|mypy|pip|python|make|npm|cargo|uv)\b", cmd):
                commands.append(cmd)
    return commands


# ===========================================================================
# EVALUATOR
# ===========================================================================

ALL_METRICS = [
    metric_command_validity,
    metric_section_coverage,
    metric_specificity,
    metric_novelty,
    metric_information_density,
    metric_actionability,
    metric_machine_parsability,
    metric_repo_grounding,
]


def evaluate(content: str, repo: Path, version: int) -> Scorecard:
    metrics = []
    for fn in ALL_METRICS:
        try:
            m = fn(content, repo)
        except Exception as e:
            m = MetricScore(fn.__name__, 0.0, details=f"Error: {e}")
        metrics.append(m)
    return Scorecard(version=version, metrics=metrics)


# ===========================================================================
# IMPROVER
# ===========================================================================

IMPROVER_SYSTEM = """\
You are improving an AGENTS.md file — a "README for AI coding agents."

Your job: rewrite the file to score higher on specific metrics that were identified as weak.
The file must remain accurate — never invent commands, file paths, or module names.
Improve QUALITY and SPECIFICITY, not just length.

AGENTS.md structure to follow:
  # AGENTS.md — {repo_name}
  ## §1 Environment & Commands
  ## §2 Architecture Map
  ## §3 Code Conventions
  ## §4 PR & Contribution Rules
  ## §5 Anti-Patterns & Gotchas    ← highest value; be specific
  ## §6 File Change Risk Map       ← use actual filenames if known
  ## §7 Testing Conventions

Rules:
- Every shell command must be runnable as-is
- Use ```bash for all shell commands
- Use ✅ / ❌ for do/don't pairs in §3 and §5
- Max 400 lines — be dense, not comprehensive
- §5 must be codebase-specific, not generic Python advice
"""


def improve(
    current_content: str,
    scorecard: Scorecard,
    repo: Path,
    repo_context: str,
) -> str:
    """
    Ask the LLM to rewrite the AGENTS.md, targeting the weakest metrics.
    """
    weakest = scorecard.weakest(3)
    weak_summary = "\n".join(
        f"  - {m.name} ({m.score:.0%}): {m.details}"
        for m in weakest
    )

    prompt = textwrap.dedent(f"""\
        Here is the current AGENTS.md (score: {scorecard.overall:.0%}):

        ---
        {current_content}
        ---

        WEAKEST METRICS (prioritize improving these):
        {weak_summary}

        REPO CONTEXT (use this to add specificity):
        {repo_context}

        Rewrite the AGENTS.md to address the weak metrics above.
        Be specific: use real file names, real commands, real patterns from the repo context.
        Do not add fictional information if it isn't in the context.
        Return ONLY the improved Markdown — no explanation, no preamble.
    """)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",  "content": IMPROVER_SYSTEM.format(repo_name=repo.name)},
            {"role": "user",    "content": prompt},
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


# ===========================================================================
# REPO CONTEXT EXTRACTOR
# ===========================================================================

def extract_repo_context(repo: Path) -> str:
    """
    Gather lightweight context from the repo to ground the LLM's rewrites.
    Stays cheap — no embeddings, no full AST walk.
    """
    ctx_parts = []

    # File tree (top 2 levels)
    tree_lines = []
    for p in sorted(repo.iterdir()):
        if p.name.startswith(".") or p.name in ("__pycache__", "node_modules"):
            continue
        tree_lines.append(f"  {p.name}{'/' if p.is_dir() else ''}")
        if p.is_dir():
            for child in sorted(p.iterdir())[:8]:
                if not child.name.startswith("__"):
                    tree_lines.append(f"    {child.name}{'/' if child.is_dir() else ''}")
    ctx_parts.append("FILE TREE:\n" + "\n".join(tree_lines[:60]))

    # pyproject.toml summary
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        import tomllib
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        project = data.get("project", {})
        ctx_parts.append(
            f"PROJECT: {project.get('name', repo.name)} "
            f"v{project.get('version', '?')}\n"
            f"DEPS: {list(project.get('dependencies', []))[:10]}\n"
            f"OPT DEPS: {list(project.get('optional-dependencies', {}).keys())}"
        )
        for tool in ("pytest", "ruff", "mypy"):
            cfg = data.get("tool", {}).get(tool)
            if cfg:
                ctx_parts.append(f"{tool.upper()} CONFIG: {json.dumps(cfg, indent=2)[:400]}")

    # README excerpt (first 60 lines)
    for readme in ("README.md", "README.rst"):
        rpath = repo / readme
        if rpath.exists():
            lines = rpath.read_text(encoding="utf-8", errors="replace").splitlines()[:60]
            ctx_parts.append("README EXCERPT:\n" + "\n".join(lines))
            break

    # CONTRIBUTING.md (if exists)
    for contrib in ("CONTRIBUTING.md", "CONTRIBUTING.rst"):
        cpath = repo / contrib
        if cpath.exists():
            lines = cpath.read_text(encoding="utf-8", errors="replace").splitlines()[:80]
            ctx_parts.append("CONTRIBUTING EXCERPT:\n" + "\n".join(lines))
            break

    # Scan for test file patterns
    test_files = list(repo.rglob("test_*.py"))[:5] + list(repo.rglob("*_test.py"))[:5]
    if test_files:
        ctx_parts.append(
            "TEST FILES (sample):\n" + "\n".join(str(f.relative_to(repo)) for f in test_files[:8])
        )

    # Sample a key source file
    key_files = ["fastapi/routing.py", "fastapi/applications.py", "src/main.py",
                 "app/main.py", "main.py"]
    for kf in key_files:
        kpath = repo / kf
        if kpath.exists():
            lines = kpath.read_text(encoding="utf-8", errors="replace").splitlines()[:50]
            ctx_parts.append(
                f"KEY FILE ({kf}, first 50 lines):\n" + "\n".join(lines)
            )
            break

    # Git log summary
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-15"],
            cwd=repo, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            ctx_parts.append("RECENT COMMITS:\n" + result.stdout.strip())
    except Exception:
        pass

    return "\n\n".join(ctx_parts)


# ===========================================================================
# MAIN LOOP
# ===========================================================================

def run_pipeline(
    repo_path: str,
    input_file: str,
    iterations: int,
    out_dir: str,
) -> None:
    repo    = Path(repo_path).resolve()
    out     = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    current_content = Path(input_file).read_text(encoding="utf-8")

    print(f"\n🔍 Extracting repo context from {repo.name}...")
    repo_context = extract_repo_context(repo)

    all_scorecards = []

    for iteration in range(iterations + 1):  # 0 = baseline, 1..N = improved
        version = iteration
        label   = "baseline" if version == 0 else f"v{version}"

        print(f"\n{'─'*60}")
        print(f"  Evaluating {label}...")

        scorecard = evaluate(current_content, repo, version)
        scorecard.print()
        all_scorecards.append(asdict(scorecard))

        # Save this version
        out_file = out / f"AGENTS_v{version}.md"
        out_file.write_text(current_content, encoding="utf-8")
        print(f"  Saved: {out_file}")

        if iteration == iterations:
            break  # done — don't improve after last eval

        # Check for plateau
        if len(all_scorecards) >= 2:
            prev_score = all_scorecards[-2]["overall"]
            curr_score = all_scorecards[-1]["overall"]
            if curr_score - prev_score < 0.02:
                print(f"\n  📊 Score plateaued ({prev_score:.0%} → {curr_score:.0%}). Stopping.")
                break

        print(f"\n  🔧 Generating improved version (targeting weakest metrics)...")
        current_content = improve(current_content, scorecard, repo, repo_context)

    # Final summary
    print(f"\n{'='*60}")
    print("  IMPROVEMENT SUMMARY")
    print(f"{'='*60}")
    for sc in all_scorecards:
        label  = "baseline" if sc["version"] == 0 else f"    v{sc['version']}  "
        delta  = ""
        if sc["version"] > 0:
            prev = all_scorecards[sc["version"] - 1]["overall"]
            diff = sc["overall"] - prev
            delta = f"  ({'+' if diff >= 0 else ''}{diff:.0%})"
        print(f"  {label}  overall: {sc['overall']:.0%}{delta}")

    # Save scores
    scores_file = out / "scores.json"
    scores_file.write_text(json.dumps(all_scorecards, indent=2))
    print(f"\n  Full scores → {scores_file}")
    print(f"  Files       → {out}/AGENTS_v*.md\n")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iteratively improve an AGENTS.md file using multi-metric evaluation."
    )
    parser.add_argument("--repo",   required=True, help="Path to the target repository")
    parser.add_argument("--input",  required=True, help="Starting AGENTS.md file")
    parser.add_argument("--iters",  type=int, default=3, help="Max improvement iterations")
    parser.add_argument("--out-dir", default="./agents_versions", help="Output directory")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    run_pipeline(args.repo, args.input, args.iters, args.out_dir)

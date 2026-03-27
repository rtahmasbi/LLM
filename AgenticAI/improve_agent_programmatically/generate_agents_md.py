#!/usr/bin/env python3
"""
generate_agents_md.py — Programmatically generate a high-quality AGENTS.md
for any Python repository using a multi-signal pipeline:

  Stage 1: Static Analysis  — AST parsing, pyproject.toml, Makefile
  Stage 2: Git History      — churn, co-change coupling, branch patterns
  Stage 3: Semantic Cluster — embed + DBSCAN cluster -> architecture map
  Stage 4: LLM Synthesis    — extract implicit conventions & anti-patterns
  Stage 5: Validation       — run every command, drop failures
  Stage 6: Render           — assemble into structured Markdown

Usage:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...        # or ANTHROPIC_API_KEY=sk-ant-...

    python generate_agents_md.py --repo /path/to/repo --out AGENTS.md
    python generate_agents_md.py --repo /path/to/repo --provider claude
    python generate_agents_md.py --repo /path/to/repo --no-llm

Notes:
    - Stage 3 (semantic clustering) requires OpenAI — Claude has no embeddings API.
      With --provider claude, stages 1, 2, 4, 5, 6 all run; only clustering is skipped.
    - llm_client.py must be in the same directory as this script.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional heavy deps for clustering — degrade gracefully if missing
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[warn] scikit-learn not installed — semantic clustering disabled", file=sys.stderr)

from llm_client import build_client, LLMClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENTS_MD_SCHEMA = """\
You are writing an AGENTS.md file — a "README for AI coding agents."

STRUCTURE (use exactly these section headers):
  ## §1 Environment & Commands
  ## §2 Architecture Map
  ## §3 Code Conventions
  ## §4 PR & Contribution Rules
  ## §5 Anti-Patterns & Gotchas
  ## §6 File Change Risk Map
  ## §7 Testing Conventions

RULES:
- Max 500 lines total. Prioritize density over completeness.
- §1 must only contain commands from the VALIDATED list provided.
- §5 must only contain codebase-SPECIFIC items, not general Python advice.
- Mark low-confidence items with a warning symbol.
- Use fenced code blocks for all commands and code examples.
- Use checkmark / X to mark correct / incorrect patterns in §3 and §5.
- Every sentence should be actionable. No filler prose.
"""

MAX_CHUNK_TOKENS = 400
MAX_EMBED_BATCH  = 100


# ===========================================================================
# STAGE 1 — STATIC ANALYSIS
# ===========================================================================

def extract_commands(repo: Path) -> dict[str, list[str]]:
    """
    Pull exact commands from pyproject.toml, setup.cfg, Makefile, tox.ini.
    Returns a dict keyed by category: 'test', 'lint', 'typecheck', 'build', 'install'.
    """
    commands: dict[str, list[str]] = defaultdict(list)

    # ---- pyproject.toml ---------------------------------------------------
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        # Hatch scripts
        for _env_name, env in data.get("tool", {}).get("hatch", {}).get("envs", {}).items():
            for script_name, script_cmd in env.get("scripts", {}).items():
                cmd = script_cmd if isinstance(script_cmd, str) else " ".join(script_cmd)
                commands[_categorize(script_name)].append(cmd)

        # pytest config -> infer test command
        pytest_cfg = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
        if pytest_cfg:
            addopts   = pytest_cfg.get("addopts", "")
            testpaths = " ".join(pytest_cfg.get("testpaths", ["tests/"]))
            commands["test"].append(f"pytest {testpaths} {addopts}".strip())

        # ruff -> infer lint command
        if "ruff" in data.get("tool", {}):
            commands["lint"].append("ruff check .")
            commands["lint"].append("ruff format --check .")

        # mypy -> infer typecheck command
        if "mypy" in data.get("tool", {}):
            src = list(data["tool"]["mypy"].get(
                "files", [data.get("project", {}).get("name", "src")]
            ))
            commands["typecheck"].append(f"mypy {' '.join(src)}")

        # install
        if data.get("project"):
            extras = list(data.get("project", {}).get("optional-dependencies", {}).keys())
            if extras:
                commands["install"].append(f"pip install -e '.[{','.join(extras)}]'")
            else:
                commands["install"].append("pip install -e .")

    # ---- Makefile ---------------------------------------------------------
    makefile = repo / "Makefile"
    if makefile.exists():
        for target, recipe in _parse_makefile(makefile).items():
            cat = _categorize(target)
            if cat != "other":
                commands[cat].extend(recipe)

    # ---- tox.ini / setup.cfg ----------------------------------------------
    for cfg_file in [repo / "tox.ini", repo / "setup.cfg"]:
        if cfg_file.exists():
            for cat, cmd_list in _parse_ini_commands(cfg_file).items():
                commands[cat].extend(cmd_list)

    # Deduplicate while preserving order
    return {k: list(dict.fromkeys(v)) for k, v in commands.items() if v}


def extract_ast_structure(repo: Path) -> dict:
    """
    Walk the repo with AST parsing.
    Returns per-file: classes, functions, decorators, imports, annotation coverage.
    """
    structure = {}
    for py_file in sorted(repo.rglob("*.py")):
        if _should_skip(py_file, repo):
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree   = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        rel = str(py_file.relative_to(repo))
        structure[rel] = {
            "classes":             _extract_classes(tree),
            "functions":           _extract_functions(tree),
            "decorators":          _extract_decorators(tree),
            "imports":             _extract_imports(tree),
            "annotation_coverage": _annotation_coverage(tree),
            "lines":               source.count("\n"),
        }
    return structure


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _extract_classes(tree: ast.AST) -> list[dict]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            out.append({"name": node.name, "bases": bases, "line": node.lineno})
    return out


def _extract_functions(tree: ast.AST) -> list[dict]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append({
                "name":   node.name,
                "async":  isinstance(node, ast.AsyncFunctionDef),
                "args":   [a.arg for a in node.args.args],
                "line":   node.lineno,
                "public": not node.name.startswith("_"),
            })
    return out


def _extract_decorators(tree: ast.AST) -> list[str]:
    decorators = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                decorators.append(ast.unparse(dec))
    return list(set(decorators))


def _extract_imports(tree: ast.AST) -> list[str]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return list(set(imports))


def _annotation_coverage(tree: ast.AST) -> float:
    total = annotated = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.args
            total     += len(args) + 1
            annotated += sum(1 for a in args if a.annotation is not None)
            annotated += (1 if node.returns is not None else 0)
    return round(annotated / total, 2) if total else 0.0


def _parse_makefile(path: Path) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = {}
    current = None
    for line in path.read_text().splitlines():
        if re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*\s*:", line) and not line.startswith("\t"):
            current = line.split(":")[0].strip()
            targets[current] = []
        elif line.startswith("\t") and current:
            cmd = line.strip()
            if cmd and not cmd.startswith("#"):
                targets[current].append(cmd)
    return targets


def _parse_ini_commands(path: Path) -> dict[str, list[str]]:
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(path)
    commands: dict[str, list[str]] = defaultdict(list)
    for section in cfg.sections():
        for key, val in cfg[section].items():
            if key in ("commands", "deps"):
                for line in val.strip().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        commands[_categorize(section)].append(line)
    return dict(commands)


def _categorize(name: str) -> str:
    name = name.lower()
    if any(k in name for k in ("test", "pytest", "spec")):
        return "test"
    if any(k in name for k in ("lint", "ruff", "flake", "pylint")):
        return "lint"
    if any(k in name for k in ("type", "mypy", "pyright")):
        return "typecheck"
    if any(k in name for k in ("build", "dist", "pack")):
        return "build"
    if any(k in name for k in ("install", "setup", "bootstrap")):
        return "install"
    if any(k in name for k in ("doc", "mkdoc", "sphinx")):
        return "docs"
    return "other"


def _should_skip(path: Path, repo: Path) -> bool:
    rel = path.relative_to(repo)
    skip_dirs = {
        ".git", "__pycache__", ".venv", "venv", "env",
        "node_modules", "dist", "build", ".tox", ".mypy_cache", ".ruff_cache",
    }
    return any(part in skip_dirs for part in rel.parts)


# ===========================================================================
# STAGE 2 — GIT HISTORY
# ===========================================================================

def extract_git_signals(repo: Path) -> dict:
    """
    Mine git log for file churn, co-change coupling, branch naming, merge strategy.
    """
    signals: dict = {}

    # File churn
    result = _git(repo, ["log", "--name-only", "--format=", "--since=1 year ago"])
    if result:
        files  = [l for l in result.splitlines() if l.endswith(".py") and l.strip()]
        churn  = Counter(files)
        signals["high_churn"] = [f for f, _ in churn.most_common(10)]
        signals["stable"]     = [f for f, c in churn.items() if c <= 2]
        signals["churn_map"]  = dict(churn.most_common(30))

    # Co-change coupling
    result = _git(repo, ["log", "--name-only", "--format=%H", "--since=1 year ago"])
    if result:
        signals["coupling"] = _compute_coupling(result)

    # Branch naming patterns
    result = _git(repo, ["branch", "-r", "--format=%(refname:short)"])
    if result:
        branches = [b.strip() for b in result.splitlines() if b.strip()]
        signals["branch_patterns"] = _infer_branch_pattern(branches)

    # Merge strategy
    result = _git(repo, ["log", "--oneline", "-100"])
    if result:
        lines            = result.splitlines()
        squash_hits      = sum(1 for l in lines if re.search(r"\(#\d+\)", l))
        merge_hits       = sum(1 for l in lines if l.lower().startswith("merge"))
        signals["merge_strategy"] = "squash" if squash_hits > merge_hits else "merge"

    # Conventional commits ratio
    result = _git(repo, ["log", "--format=%s", "-50"])
    if result:
        subjects     = result.splitlines()
        conventional = sum(
            1 for s in subjects
            if re.match(r"^(feat|fix|docs|refactor|test|chore|ci|build|style|perf)[\(:!]", s)
        )
        signals["conventional_commits"] = conventional / len(subjects) if subjects else 0

    return signals


def _git(repo: Path, args: list[str]) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git"] + args, cwd=repo,
            capture_output=True, text=True, timeout=30
        )
        return r.stdout if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _compute_coupling(log_output: str) -> list[dict]:
    commits: dict[str, list[str]] = {}
    current_hash = None
    for line in log_output.splitlines():
        line = line.strip()
        if re.match(r"^[0-9a-f]{40}$", line):
            current_hash = line
            commits[current_hash] = []
        elif line.endswith(".py") and current_hash:
            commits[current_hash].append(line)

    pair_counts: Counter = Counter()
    for files in commits.values():
        py_files = [f for f in files if f.endswith(".py")]
        for i, a in enumerate(py_files):
            for b in py_files[i + 1:]:
                pair_counts[tuple(sorted([a, b]))] += 1

    return [
        {"files": list(pair), "count": count}
        for pair, count in pair_counts.most_common(10)
        if count >= 3
    ]


def _infer_branch_pattern(branches: list[str]) -> str:
    patterns: Counter = Counter()
    for b in branches:
        b = b.replace("origin/", "")
        if re.match(r"^(feat|feature)/\d+", b):
            patterns["<type>/<issue-num>-description"] += 1
        elif re.match(r"^(feat|feature)/", b):
            patterns["feat/<description>"] += 1
        elif re.match(r"^fix/", b):
            patterns["fix/<description>"] += 1
        elif re.match(r"^[A-Z]+-\d+", b):
            patterns["<JIRA-123>-description"] += 1
    return patterns.most_common(1)[0][0] if patterns else "feature/<description>"


# ===========================================================================
# STAGE 3 — SEMANTIC CLUSTERING
# ===========================================================================

def chunk_repo(repo: Path, structure: dict) -> list[dict]:
    """
    Produce one chunk per significant AST node (class or top-level function).
    """
    chunks = []
    for rel_path, _info in structure.items():
        abs_path = repo / rel_path
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
            tree   = ast.parse(source)
        except (SyntaxError, FileNotFoundError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("__") and node.name.endswith("__"):
                continue

            try:
                lines   = source.splitlines()
                end     = getattr(node, "end_lineno", node.lineno + 20)
                snippet = "\n".join(lines[node.lineno - 1 : min(end, node.lineno + 40)])
                snippet = snippet[:MAX_CHUNK_TOKENS * 4]
            except Exception:
                continue

            chunks.append({
                "file": rel_path,
                "name": node.name,
                "kind": type(node).__name__,
                "line": node.lineno,
                "text": snippet,
            })

    return chunks


def embed_and_cluster(chunks: list[dict], llm: LLMClient) -> list[dict]:
    """
    Embed all chunks with OpenAI, DBSCAN-cluster, name each cluster with LLM.
    Skipped automatically when using Claude provider (no embeddings API).
    """
    if not HAS_SKLEARN or not chunks:
        return []

    if not llm.supports_embeddings:
        print(
            "[warn] Semantic clustering requires OpenAI embeddings. "
            "Skipping for Claude provider — use --provider openai to enable.",
            file=sys.stderr,
        )
        return []

    texts = [_sanitize(c["text"]) for c in chunks]
    print(f"      Embedding {len(texts)} chunks...", file=sys.stderr)
    embeddings = llm.embed(texts, batch_size=MAX_EMBED_BATCH)

    X      = normalize(np.array(embeddings))
    labels = DBSCAN(eps=0.25, min_samples=3, metric="cosine").fit_predict(X)

    cluster_map: dict[int, list[dict]] = defaultdict(list)
    for chunk, label in zip(chunks, labels):
        if label != -1:
            cluster_map[label].append(chunk)

    # Cap at top-20 largest — smaller clusters are usually noise
    top_clusters = sorted(cluster_map.items(), key=lambda kv: len(kv[1]), reverse=True)[:20]

    named_clusters = []
    for _label, cluster_chunks in top_clusters:
        files  = list({c["file"] for c in cluster_chunks})
        sample = cluster_chunks[:5]

        name = llm.chat(
            user=(
                "Name this cluster of related code in 2-4 words "
                "(e.g. 'Authentication Layer', 'Router Definitions', 'Dependency Injection').\n\n"
                "Files: " + ", ".join(files[:5]) + "\n\n"
                "Sample code:\n" + "\n\n".join(_sanitize(c["text"][:300]) for c in sample) + "\n\n"
                "Respond with ONLY the cluster name, nothing else."
            ),
            max_tokens=20,
        ).strip().strip('"')

        named_clusters.append({
            "name":                 name,
            "files":                files,
            "representative_chunks": sample,
        })

    return named_clusters


# ===========================================================================
# STAGE 4 — LLM SYNTHESIS
# ===========================================================================

def _sanitize(text: str) -> str:
    """
    Strip characters that cause JSON serialization failures:
    null bytes, ASCII control chars, lone surrogates, lines > 500 chars.
    """
    text  = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text  = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    lines = [l[:500] for l in text.splitlines()]
    return "\n".join(lines)


def synthesize_conventions(
    clusters: list[dict],
    structure: dict,
    git_signals: dict,
    llm: LLMClient,
) -> dict:
    """
    For each cluster, extract implicit conventions, anti-patterns, and gotchas.
    Returns confidence-filtered results.
    """
    all_conventions: dict[str, list[dict]] = defaultdict(list)

    for i, cluster in enumerate(clusters):
        print(f"  [synthesis] cluster {i+1}/{len(clusters)}: {cluster['name']}", file=sys.stderr)

        sample_code = _sanitize("\n\n---\n\n".join(
            f"# {c['file']} — {c['name']}\n{c['text']}"
            for c in cluster["representative_chunks"]
        ))

        try:
            raw = llm.chat(
                system=textwrap.dedent("""\
                    You analyze source code to extract implicit conventions for an AGENTS.md file.
                    Focus ONLY on things that:
                    1. Are actually evidenced in the code samples provided
                    2. Are specific to this codebase, not general Python best practices
                    3. Would trip up an AI agent making its first contribution

                    Return valid JSON only — no prose outside the JSON object.
                """),
                user=textwrap.dedent(f"""\
                    Cluster: "{cluster['name']}"
                    Files: {", ".join(cluster["files"][:8])}

                    Code samples:
                    {sample_code[:6000]}

                    Return JSON with this exact structure:
                    {{
                      "conventions": [
                        {{"rule": "...", "example": "...", "confidence": 0.0}}
                      ],
                      "antipatterns": [
                        {{"rule": "...", "wrong_example": "...", "right_example": "...", "confidence": 0.0}}
                      ],
                      "gotchas": [
                        {{"description": "...", "confidence": 0.0}}
                      ]
                    }}

                    Only include items with confidence >= 0.6.
                    If nothing is noteworthy, return empty lists.
                """),
                max_tokens=1024,
                json_mode=True,
            )
        except Exception as e:
            print(f"  [synthesis] cluster {i+1} skipped: {e}", file=sys.stderr)
            continue

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        for key in ("conventions", "antipatterns", "gotchas"):
            items     = parsed.get(key, [])
            high_conf = [item for item in items if item.get("confidence", 0) >= 0.7]
            low_conf  = [item for item in items if 0.6 <= item.get("confidence", 0) < 0.7]
            for item in low_conf:
                item["low_confidence"] = True
            all_conventions[key].extend(high_conf + low_conf)

    return dict(all_conventions)


def self_critique_pass(conventions: dict, llm: LLMClient) -> dict:
    """
    Second LLM pass: remove hallucinated, vague, or non-specific items.
    """
    try:
        raw = llm.chat(
            system="You review extracted code conventions for quality and accuracy.",
            user=textwrap.dedent(f"""\
                Review these extracted conventions and remove any that:
                1. Are general Python advice, not codebase-specific
                2. Are too vague to be actionable (e.g. "use good variable names")
                3. Contradict each other
                4. Are about commands or tools that might not exist in this project

                Input:
                {json.dumps(conventions, indent=2)[:8000]}

                Return the same JSON structure with low-quality items removed.
                Keep the "low_confidence" flag on items that had it.
            """),
            max_tokens=4096,
            json_mode=True,
        )
        return json.loads(raw)
    except Exception as e:
        print(f"[warn] self-critique pass failed: {e} — using unfiltered conventions",
              file=sys.stderr)
        return conventions


# ===========================================================================
# STAGE 5 — COMMAND VALIDATION
# ===========================================================================

def validate_commands(commands: dict[str, list[str]], repo: Path) -> dict[str, list[str]]:
    """
    Run every non-install command. Drop failures before writing to AGENTS.md.
    """
    validated: dict[str, list[str]] = {}

    safe_check = {
        "lint":      lambda cmd: cmd + " --no-fix"          if "ruff"  in cmd else cmd,
        "typecheck": lambda cmd: cmd + " --no-error-summary" if "mypy"  in cmd else cmd,
    }

    skip_prefixes = (
        "pip install", "npm install", "make install",
        "python setup", "conda ", "uv install",
    )

    for category, cmd_list in commands.items():
        validated[category] = []
        for cmd in cmd_list:
            if any(cmd.startswith(p) for p in skip_prefixes):
                validated[category].append(cmd)
                continue

            check_cmd = safe_check.get(category, lambda c: c)(cmd)
            try:
                result = subprocess.run(
                    check_cmd, shell=True, cwd=repo,
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    validated[category].append(cmd)
                else:
                    print(f"  [validation] DROPPED '{cmd}' -> exit {result.returncode}",
                          file=sys.stderr)
            except subprocess.TimeoutExpired:
                validated[category].append(f"{cmd}  # [timeout during validation]")
            except Exception as e:
                print(f"  [validation] ERROR '{cmd}': {e}", file=sys.stderr)

    return {k: v for k, v in validated.items() if v}


# ===========================================================================
# STAGE 6 — RENDER TO MARKDOWN
# ===========================================================================

def render_agents_md(
    repo: Path,
    commands: dict,
    validated_commands: dict,
    clusters: list[dict],
    conventions: dict,
    git_signals: dict,
    structure: dict,
    llm: Optional[LLMClient],
) -> str:
    if llm is not None:
        return _render_with_llm(
            repo, validated_commands, clusters, conventions, git_signals, structure, llm
        )
    return _render_template(
        repo, validated_commands, clusters, conventions, git_signals, structure
    )


def _render_with_llm(
    repo: Path,
    validated_commands: dict,
    clusters: list[dict],
    conventions: dict,
    git_signals: dict,
    structure: dict,
    llm: LLMClient,
) -> str:
    arch_summary = [
        {"name": c["name"], "files": c["files"][:6]}
        for c in clusters[:12]
    ]
    risk_map = {
        "high_churn": git_signals.get("high_churn", [])[:8],
        "stable":     git_signals.get("stable",     [])[:8],
        "coupling":   git_signals.get("coupling",   [])[:5],
    }
    low_ann_files = [
        path for path, info in structure.items()
        if info["annotation_coverage"] < 0.5
    ][:5]

    branch_pattern = git_signals.get("branch_patterns", "feat/<description>")
    merge_strategy = git_signals.get("merge_strategy", "squash")
    conventional   = git_signals.get("conventional_commits", 0)

    return llm.chat(
        system=AGENTS_MD_SCHEMA,
        user=textwrap.dedent(f"""\
            Write an AGENTS.md for the repository: {repo.name}

            Use ONLY these validated commands in §1 (do not invent any):
            {json.dumps(validated_commands, indent=2)}

            Architecture clusters (use for §2):
            {json.dumps(arch_summary, indent=2)}

            Extracted conventions (use for §3 and §5):
            {json.dumps(conventions, indent=2)[:4000]}

            Git risk signals (use for §6):
            {json.dumps(risk_map, indent=2)}

            Branch pattern: {branch_pattern}
            Merge strategy: {merge_strategy}
            Conventional commits: {"yes" if conventional > 0.6 else "not observed"}

            Files with low type annotation coverage (agents should be careful):
            {", ".join(low_ann_files) or "none"}

            Write the complete AGENTS.md now.
        """),
        max_tokens=4096,
    )


def _render_template(
    repo: Path,
    validated_commands: dict,
    clusters: list[dict],
    conventions: dict,
    git_signals: dict,
    structure: dict,
) -> str:
    """Fallback renderer — no LLM required."""
    lines = [
        f"# AGENTS.md — {repo.name}",
        "",
        "> Auto-generated. Commands have been validated against the repo.",
        "",
        "---",
        "",
    ]

    lines += ["## §1 Environment & Commands", ""]
    for category, cmds in validated_commands.items():
        lines += [f"### {category.title()}", "", "```bash"]
        lines += cmds
        lines += ["```", ""]

    if clusters:
        lines += ["## §2 Architecture Map", ""]
        for c in clusters[:10]:
            lines += [f"**{c['name']}**"]
            for f in c["files"][:5]:
                lines += [f"  - `{f}`"]
        lines += [""]

    if conventions.get("conventions"):
        lines += ["## §3 Code Conventions", ""]
        for item in conventions["conventions"][:10]:
            flag = " [low confidence]" if item.get("low_confidence") else ""
            lines += [f"- {item['rule']}{flag}"]
            if item.get("example"):
                lines += [f"  ```python\n  {item['example']}\n  ```"]
        lines += [""]

    if conventions.get("antipatterns"):
        lines += ["## §5 Anti-Patterns & Gotchas", ""]
        for item in conventions["antipatterns"][:10]:
            flag = " [low confidence]" if item.get("low_confidence") else ""
            lines += [f"- {item['rule']}{flag}"]
        lines += [""]

    if git_signals.get("high_churn"):
        lines += ["## §6 File Change Risk Map", ""]
        lines += ["**High churn — change carefully:**"]
        for f in git_signals["high_churn"][:6]:
            lines += [f"  - `{f}`"]
        if git_signals.get("stable"):
            lines += ["", "**Stable — low risk:**"]
            for f in git_signals["stable"][:6]:
                lines += [f"  - `{f}`"]
        lines += [""]

    return "\n".join(lines)


# ===========================================================================
# MAIN
# ===========================================================================

def generate(
    repo_path: str,
    output_path: str = "AGENTS.md",
    provider: Optional[str] = None,
    no_llm: bool = False,
) -> str:
    repo = Path(repo_path).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Repo not found: {repo}")

    # Build LLM client
    llm = None if no_llm else build_client(provider=provider)
    if llm is None and not no_llm:
        print("[warn] No LLM client — stages 3 and 4 disabled. "
              "Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable.",
              file=sys.stderr)

    # Stage 1
    print(f"[1/6] Static analysis: {repo.name}", file=sys.stderr)
    commands  = extract_commands(repo)
    structure = extract_ast_structure(repo)
    print(f"      {len(structure)} Python files parsed", file=sys.stderr)

    # Stage 2
    print("[2/6] Git history mining", file=sys.stderr)
    git_signals = extract_git_signals(repo)

    # Stage 3 — needs OpenAI embeddings
    if llm and HAS_SKLEARN and llm.supports_embeddings:
        print("[3/6] Semantic clustering", file=sys.stderr)
        chunks   = chunk_repo(repo, structure)
        print(f"      {len(chunks)} AST chunks to embed", file=sys.stderr)
        clusters = embed_and_cluster(chunks, llm)
        print(f"      {len(clusters)} clusters found", file=sys.stderr)
    else:
        if llm and not llm.supports_embeddings:
            print("[3/6] Semantic clustering skipped (Claude provider — no embeddings API)",
                  file=sys.stderr)
        else:
            print("[3/6] Semantic clustering skipped", file=sys.stderr)
        clusters = []

    # Stage 4
    if llm:
        print("[4/6] LLM synthesis", file=sys.stderr)
        conventions = synthesize_conventions(clusters, structure, git_signals, llm)
        print("[4b]  Self-critique pass", file=sys.stderr)
        conventions = self_critique_pass(conventions, llm)
    else:
        print("[4/6] LLM synthesis skipped", file=sys.stderr)
        conventions = {}

    # Stage 5
    print("[5/6] Validating commands", file=sys.stderr)
    validated = validate_commands(commands, repo)
    print(f"      {sum(len(v) for v in validated.values())} commands passed validation",
          file=sys.stderr)

    # Stage 6
    print("[6/6] Rendering AGENTS.md", file=sys.stderr)
    content = render_agents_md(
        repo, commands, validated, clusters, conventions, git_signals, structure, llm
    )

    output = Path(output_path)
    output.write_text(content, encoding="utf-8")
    print(f"[done] Written to {output}", file=sys.stderr)
    return content


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a high-quality AGENTS.md for a Python repository."
    )
    parser.add_argument(
        "--repo", required=True,
        help="Path to the repository root"
    )
    parser.add_argument(
        "--out", default="AGENTS.md",
        help="Output file path (default: AGENTS.md)"
    )
    parser.add_argument(
        "--provider", choices=["openai", "claude"], default=None,
        help="LLM provider: 'openai' or 'claude' (default: auto-detect from env vars)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip all LLM calls — static analysis + git only, no API cost"
    )
    args = parser.parse_args()

    generate(
        repo_path   = args.repo,
        output_path = args.out,
        provider    = args.provider,
        no_llm      = args.no_llm,
    )

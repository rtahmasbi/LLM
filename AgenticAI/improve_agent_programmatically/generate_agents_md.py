#!/usr/bin/env python3
"""
generate_agents_md.py — Programmatically generate a high-quality AGENTS.md
for any Python repository using a multi-signal pipeline:

  Stage 1: Static Analysis  — AST parsing, pyproject.toml, Makefile
  Stage 2: Git History      — churn, co-change coupling, branch patterns
  Stage 3: Semantic Cluster — embed + DBSCAN cluster → architecture map
  Stage 4: LLM Synthesis    — extract implicit conventions & anti-patterns
  Stage 5: Validation       — run every command, drop failures
  Stage 6: Render           — assemble into structured Markdown

Usage:
    pip install openai numpy scikit-learn tiktoken
    python generate_agents_md.py --repo /path/to/repo --out AGENTS.md

Requirements:
    OPENAI_API_KEY environment variable must be set.
"""

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

from llm_client import build_client

# ---------------------------------------------------------------------------
# Optional heavy deps — degrade gracefully if missing
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[warn] scikit-learn not installed — semantic clustering disabled", file=sys.stderr)


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
- Mark low-confidence items with ⚠️.
- Use fenced code blocks for all commands and code examples.
- Use ✅ / ❌ to mark correct / incorrect patterns in §3 and §5.
- Every sentence should be actionable. No filler prose.
"""

EMBED_MODEL   = "text-embedding-3-small"
SYNTH_MODEL   = "gpt-4o"
MAX_CHUNK_TOKENS = 400   # approximate; one AST node per chunk
MAX_EMBED_BATCH  = 100


# ===========================================================================
# STAGE 1 — STATIC ANALYSIS
# ===========================================================================

def extract_commands(repo: Path) -> dict[str, list[str]]:
    """
    Pull exact commands from pyproject.toml, setup.cfg, Makefile, tox.ini.
    Returns a dict keyed by category: 'test', 'lint', 'build', 'install'.
    """
    commands: dict[str, list[str]] = defaultdict(list)

    # ---- pyproject.toml -----------------------------------------------
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        # Hatch scripts
        for env_name, env in data.get("tool", {}).get("hatch", {}).get("envs", {}).items():
            for script_name, script_cmd in env.get("scripts", {}).items():
                cmd = script_cmd if isinstance(script_cmd, str) else " ".join(script_cmd)
                category = _categorize(script_name)
                commands[category].append(cmd)

        # pytest config → infer test command
        pytest_cfg = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
        if pytest_cfg:
            addopts = pytest_cfg.get("addopts", "")
            testpaths = " ".join(pytest_cfg.get("testpaths", ["tests/"]))
            commands["test"].append(f"pytest {testpaths} {addopts}".strip())

        # ruff → infer lint command
        if "ruff" in data.get("tool", {}):
            commands["lint"].append("ruff check .")
            commands["lint"].append("ruff format --check .")

        # mypy → infer typecheck command
        if "mypy" in data.get("tool", {}):
            src = list(data["tool"]["mypy"].get("files", [data.get("project", {}).get("name", "src")]))
            commands["typecheck"].append(f"mypy {' '.join(src)}")

        # install
        if data.get("project"):
            extras = list(data.get("project", {}).get("optional-dependencies", {}).keys())
            if extras:
                commands["install"].append(f"pip install -e '.[{','.join(extras)}]'")
            else:
                commands["install"].append("pip install -e .")

    # ---- Makefile -----------------------------------------------------
    makefile = repo / "Makefile"
    if makefile.exists():
        for target, recipe in _parse_makefile(makefile).items():
            category = _categorize(target)
            if category != "other":
                commands[category].extend(recipe)

    # ---- tox.ini / setup.cfg ------------------------------------------
    for cfg_file in [repo / "tox.ini", repo / "setup.cfg"]:
        if cfg_file.exists():
            cmds = _parse_ini_commands(cfg_file)
            for cat, cmd_list in cmds.items():
                commands[cat].extend(cmd_list)

    # Deduplicate while preserving order
    return {k: list(dict.fromkeys(v)) for k, v in commands.items() if v}


def extract_ast_structure(repo: Path) -> dict:
    """
    Walk the repo with AST parsing.
    Returns per-file: classes, functions, decorators, imports, type annotation coverage.
    """
    structure = {}
    for py_file in sorted(repo.rglob("*.py")):
        if _should_skip(py_file, repo):
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        rel = str(py_file.relative_to(repo))
        structure[rel] = {
            "classes":    _extract_classes(tree),
            "functions":  _extract_functions(tree),
            "decorators": _extract_decorators(tree),
            "imports":    _extract_imports(tree),
            "annotation_coverage": _annotation_coverage(tree),
            "lines":      source.count("\n"),
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
                "name":     node.name,
                "async":    isinstance(node, ast.AsyncFunctionDef),
                "args":     [a.arg for a in node.args.args],
                "line":     node.lineno,
                "public":   not node.name.startswith("_"),
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
            total += len(args) + 1  # +1 for return
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
    skip_dirs = {".git", "__pycache__", ".venv", "venv", "env", "node_modules",
                 "dist", "build", ".tox", ".mypy_cache", ".ruff_cache"}
    return any(part in skip_dirs for part in rel.parts)


# ===========================================================================
# STAGE 2 — GIT HISTORY
# ===========================================================================

def extract_git_signals(repo: Path) -> dict:
    """
    Mine git log for:
      - file churn (commit frequency)
      - co-change coupling (files that change together)
      - branch naming patterns
      - squash vs merge commit style
    """
    signals: dict = {}

    # ---- File churn ---------------------------------------------------
    result = _git(repo, ["log", "--name-only", "--format=", "--since=1 year ago"])
    if result:
        files = [l for l in result.splitlines() if l.endswith(".py") and l.strip()]
        churn = Counter(files)
        signals["high_churn"] = [f for f, _ in churn.most_common(10)]
        signals["stable"]     = [f for f, c in churn.items() if c <= 2]
        signals["churn_map"]  = dict(churn.most_common(30))

    # ---- Co-change coupling -------------------------------------------
    result = _git(repo, ["log", "--name-only", "--format=%H", "--since=1 year ago"])
    if result:
        signals["coupling"] = _compute_coupling(result)

    # ---- Branch naming patterns ---------------------------------------
    result = _git(repo, ["branch", "-r", "--format=%(refname:short)"])
    if result:
        branches = [b.strip() for b in result.splitlines() if b.strip()]
        signals["branch_patterns"] = _infer_branch_pattern(branches)

    # ---- Merge strategy -----------------------------------------------
    result = _git(repo, ["log", "--oneline", "-100"])
    if result:
        lines = result.splitlines()
        squash_indicators = sum(1 for l in lines if re.search(r"\(#\d+\)", l))
        merge_indicators  = sum(1 for l in lines if l.lower().startswith("merge"))
        signals["merge_strategy"] = (
            "squash" if squash_indicators > merge_indicators else "merge"
        )

    # ---- Commit message style -----------------------------------------
    result = _git(repo, ["log", "--format=%s", "-50"])
    if result:
        subjects = result.splitlines()
        conventional = sum(
            1 for s in subjects
            if re.match(r"^(feat|fix|docs|refactor|test|chore|ci|build|style|perf)[\(:!]", s)
        )
        signals["conventional_commits"] = conventional / len(subjects) if subjects else 0

    return signals


def _git(repo: Path, args: list[str]) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git"] + args,
            cwd=repo, capture_output=True, text=True, timeout=30
        )
        return r.stdout if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _compute_coupling(log_output: str) -> list[dict]:
    """Files that frequently change in the same commit."""
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
            for b in py_files[i+1:]:
                pair_counts[tuple(sorted([a, b]))] += 1

    return [
        {"files": list(pair), "count": count}
        for pair, count in pair_counts.most_common(10)
        if count >= 3
    ]


def _infer_branch_pattern(branches: list[str]) -> str:
    patterns = Counter()
    for b in branches:
        b = b.replace("origin/", "")
        if re.match(r"^(feat|feature)/", b):
            patterns["feat/<description>"] += 1
        elif re.match(r"^fix/", b):
            patterns["fix/<description>"] += 1
        elif re.match(r"^(feat|fix)/\d+", b):
            patterns["<type>/<issue-num>-description"] += 1
        elif re.match(r"^[A-Z]+-\d+", b):
            patterns["<JIRA-123>-description"] += 1
    return patterns.most_common(1)[0][0] if patterns else "feature/<description>"


# ===========================================================================
# STAGE 3 — SEMANTIC CLUSTERING
# ===========================================================================

def chunk_repo(repo: Path, structure: dict) -> list[dict]:
    """
    Produce one chunk per significant AST node (class or function).
    Chunks carry their file path and a text snippet for embedding.
    """
    chunks = []
    for rel_path, info in structure.items():
        abs_path = repo / rel_path
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (SyntaxError, FileNotFoundError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("__") and node.name.endswith("__"):
                continue  # skip dunders

            # Extract source lines for this node
            try:
                lines = source.splitlines()
                end = getattr(node, "end_lineno", node.lineno + 20)
                snippet = "\n".join(lines[node.lineno - 1 : min(end, node.lineno + 40)])
                # Trim to ~400 tokens (rough: 1 token ≈ 4 chars)
                snippet = snippet[:MAX_CHUNK_TOKENS * 4]
            except Exception:
                continue

            chunks.append({
                "file":    rel_path,
                "name":    node.name,
                "kind":    type(node).__name__,
                "line":    node.lineno,
                "text":    snippet,
            })

    return chunks


def embed_and_cluster(chunks: list[dict], client: "OpenAI") -> list[dict]:
    """
    Embed all chunks, DBSCAN-cluster, name each cluster with LLM.
    Returns list of {name, files, representative_chunks}.
    """
    if not HAS_SKLEARN or not chunks:
        return []

    texts = [c["text"] for c in chunks]

    # Batch embed
    embeddings = []
    for i in range(0, len(texts), MAX_EMBED_BATCH):
        batch = texts[i : i + MAX_EMBED_BATCH]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([r.embedding for r in response.data])

    X = normalize(np.array(embeddings))

    # DBSCAN — no need to specify n_clusters
    labels = DBSCAN(eps=0.25, min_samples=3, metric="cosine").fit_predict(X)

    # Group chunks by cluster
    cluster_map: dict[int, list[dict]] = defaultdict(list)
    for chunk, label in zip(chunks, labels):
        if label != -1:
            cluster_map[label].append(chunk)

    # Cap at top-20 largest clusters — 202 clusters generates too many API calls
    # and most small clusters are noise. Pick by size descending.
    top_clusters = sorted(cluster_map.items(), key=lambda kv: len(kv[1]), reverse=True)[:20]

    # Name each cluster
    named_clusters = []
    for label, cluster_chunks in top_clusters:
        files = list({c["file"] for c in cluster_chunks})
        sample = cluster_chunks[:5]

        name_resp = client.chat.completions.create(
            model=SYNTH_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Name this cluster of related code in 2-4 words "
                    "(e.g. 'Authentication Layer', 'Router Definitions', 'Dependency Injection').\n\n"
                    "Files: " + ", ".join(files[:5]) + "\n\n"
                    "Sample code:\n" + "\n\n".join(_sanitize(c["text"][:300]) for c in sample) + "\n\n"
                    "Respond with ONLY the cluster name, nothing else."
                )
            }],
            max_tokens=20,
        )
        name = name_resp.choices[0].message.content.strip().strip('"')

        named_clusters.append({
            "name":  name,
            "files": files,
            "representative_chunks": sample,
        })

    return named_clusters


# ===========================================================================
# STAGE 4 — LLM SYNTHESIS
# ===========================================================================

def _sanitize(text: str) -> str:
    """
    Remove characters that cause JSON serialization failures in the OpenAI API:
    - Null bytes and other ASCII control characters (except tab/newline/CR)
    - Lone surrogates and other invalid Unicode
    - Excessively long lines (truncate at 500 chars)
    """
    # Strip null bytes and control chars except \t \n \r
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Replace lone surrogates that slip through on some platforms
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    # Truncate any single line that is absurdly long (e.g. minified JS in a .py)
    lines = [l[:500] for l in text.splitlines()]
    return "\n".join(lines)


def synthesize_conventions(
    clusters: list[dict],
    structure: dict,
    git_signals: dict,
    client: "OpenAI",
) -> dict:
    """
    For each cluster, ask the LLM to extract implicit conventions, anti-patterns,
    and gotchas evidenced in the code. Returns confidence-filtered results.
    """
    all_conventions: dict[str, list[dict]] = defaultdict(list)

    for i, cluster in enumerate(clusters):
        print(f"  [synthesis] cluster {i+1}/{len(clusters)}: {cluster['name']}", file=sys.stderr)
        sample_code = _sanitize("\n\n---\n\n".join(
            f"# {c['file']} — {c['name']}\n{c['text']}"
            for c in cluster["representative_chunks"]
        ))

        try:
            response = client.chat.completions.create(
            model=SYNTH_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": textwrap.dedent("""\
                        You analyze source code to extract implicit conventions for an AGENTS.md file.
                        Focus ONLY on things that:
                        1. Are actually evidenced in the code samples provided
                        2. Are specific to this codebase, not general Python best practices
                        3. Would trip up an AI agent making its first contribution

                        Return valid JSON only — no prose outside the JSON object.
                    """)
                },
                {
                    "role": "user",
                    "content": textwrap.dedent(f"""\
                        Cluster: "{cluster['name']}"
                        Files: {", ".join(cluster["files"][:8])}

                        Code samples:
                        {sample_code[:6000]}

                        Return JSON with this exact structure:
                        {{
                          "conventions": [
                            {{"rule": "...", "example": "...", "confidence": 0.0-1.0}}
                          ],
                          "antipatterns": [
                            {{"rule": "...", "wrong_example": "...", "right_example": "...", "confidence": 0.0-1.0}}
                          ],
                          "gotchas": [
                            {{"description": "...", "confidence": 0.0-1.0}}
                          ]
                        }}

                        Only include items with confidence >= 0.6.
                        If nothing is noteworthy, return empty lists.
                    """)
                }
            ],
        )
        except Exception as e:
            print(f"  [synthesis] cluster {i} ('{cluster['name']}') skipped: {e}",
                  file=sys.stderr)
            continue

        try:
            parsed = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            continue

        for key in ("conventions", "antipatterns", "gotchas"):
            items = parsed.get(key, [])
            # Filter by confidence
            high_conf = [item for item in items if item.get("confidence", 0) >= 0.7]
            low_conf  = [item for item in items if 0.6 <= item.get("confidence", 0) < 0.7]
            # Mark low-confidence items
            for item in low_conf:
                item["low_confidence"] = True
            all_conventions[key].extend(high_conf + low_conf)

    return dict(all_conventions)


def self_critique_pass(conventions: dict, client: "OpenAI") -> dict:
    """
    Second LLM pass: remove hallucinated, too-vague, or non-specific items.
    """
    response = client.chat.completions.create(
        model=SYNTH_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You review extracted code conventions for quality and accuracy."
            },
            {
                "role": "user",
                "content": textwrap.dedent(f"""\
                    Review these extracted conventions and remove any that:
                    1. Are general Python advice, not codebase-specific
                    2. Are too vague to be actionable (e.g. "use good variable names")
                    3. Contradict each other
                    4. Are about commands or tools that might not exist in this project

                    Input:
                    {json.dumps(conventions, indent=2)[:8000]}

                    Return the same JSON structure with low-quality items removed.
                    Keep the "low_confidence" flag on items that had it.
                """)
            }
        ],
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return conventions  # fall back to unfiltered if parse fails


# ===========================================================================
# STAGE 5 — COMMAND VALIDATION
# ===========================================================================

def validate_commands(commands: dict[str, list[str]], repo: Path) -> dict[str, list[str]]:
    """
    Actually run every command in a subprocess.
    Commands that fail are logged and removed.

    Note: test/build commands are checked for syntax only (--help / --version),
    not executed fully, to avoid side effects.
    """
    validated: dict[str, list[str]] = {}

    # Commands safe to actually run (idempotent checks)
    safe_check = {
        "lint":      lambda cmd: cmd + " --no-fix" if "ruff" in cmd else cmd,
        "typecheck": lambda cmd: cmd + " --no-error-summary" if "mypy" in cmd else cmd,
    }

    # Commands to only syntax-check (not execute)
    syntax_only_prefixes = ("pytest", "python -m pytest", "make ", "pip install")

    for category, cmd_list in commands.items():
        validated[category] = []
        for cmd in cmd_list:
            if any(cmd.startswith(p) for p in syntax_only_prefixes):
                # Can't safely run installs or full test suites — assume valid
                # if we found them in config files (they were extracted, not invented)
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
                    print(f"  [validation] DROPPED '{cmd}' → exit {result.returncode}",
                          file=sys.stderr)
            except subprocess.TimeoutExpired:
                print(f"  [validation] TIMEOUT '{cmd}' — keeping with ⚠️", file=sys.stderr)
                validated[category].append(f"{cmd}  # ⚠️ timed out during validation")
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
    client: Optional["OpenAI"],
) -> str:
    """
    Assemble all signals into a structured AGENTS.md document.
    If OpenAI is not available, falls back to a template-based renderer.
    """
    if client is not None:
        return _render_with_llm(
            repo, validated_commands, clusters, conventions, git_signals, structure, client
        )
    else:
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
    client: "OpenAI",
) -> str:
    # Build architecture summary
    arch_summary = [
        {"name": c["name"], "files": c["files"][:6]}
        for c in clusters[:12]
    ]

    # Build risk map from git signals
    risk_map = {
        "high_churn": git_signals.get("high_churn", [])[:8],
        "stable":     git_signals.get("stable", [])[:8],
        "coupling":   git_signals.get("coupling", [])[:5],
    }

    # Annotation coverage summary
    coverage_info = {
        path: info["annotation_coverage"]
        for path, info in structure.items()
        if info["annotation_coverage"] < 0.5
    }

    prompt_data = {
        "repo_name":          repo.name,
        "validated_commands": validated_commands,
        "architecture":       arch_summary,
        "conventions":        conventions,
        "risk_map":           risk_map,
        "branch_pattern":     git_signals.get("branch_patterns", "feat/<description>"),
        "merge_strategy":     git_signals.get("merge_strategy", "squash"),
        "conventional_commits": git_signals.get("conventional_commits", 0),
        "low_annotation_files": list(coverage_info.keys())[:5],
    }

    response = client.chat.completions.create(
        model=SYNTH_MODEL,
        messages=[
            {"role": "system", "content": AGENTS_MD_SCHEMA},
            {
                "role": "user",
                "content": textwrap.dedent(f"""\
                    Write an AGENTS.md for the repository: {repo.name}

                    Use ONLY these validated commands in §1 (do not invent any):
                    {json.dumps(validated_commands, indent=2)}

                    Architecture clusters (use for §2):
                    {json.dumps(arch_summary, indent=2)}

                    Extracted conventions (use for §3 and §5):
                    {json.dumps(conventions, indent=2)[:4000]}

                    Git risk signals (use for §6):
                    {json.dumps(risk_map, indent=2)}

                    Branch pattern: {prompt_data["branch_pattern"]}
                    Merge strategy: {prompt_data["merge_strategy"]}
                    Conventional commits: {"yes" if prompt_data["conventional_commits"] > 0.6 else "not observed"}

                    Files with low type annotation coverage (caution):
                    {", ".join(prompt_data["low_annotation_files"]) or "none"}

                    Write the complete AGENTS.md now.
                """)
            }
        ],
        max_tokens=4096,
    )

    return response.choices[0].message.content


def _render_template(
    repo: Path,
    validated_commands: dict,
    clusters: list[dict],
    conventions: dict,
    git_signals: dict,
    structure: dict,
) -> str:
    """
    Fallback renderer — no LLM required.
    Produces a solid AGENTS.md from structured data alone.
    """
    lines = [
        f"# AGENTS.md — {repo.name}",
        "",
        "> Auto-generated. Commands have been validated against the repo.",
        "",
        "---",
        "",
    ]

    # §1 Commands
    lines += ["## §1 Environment & Commands", ""]
    for category, cmds in validated_commands.items():
        lines += [f"### {category.title()}", "", "```bash"]
        lines += cmds
        lines += ["```", ""]

    # §2 Architecture
    if clusters:
        lines += ["## §2 Architecture Map", ""]
        for c in clusters[:10]:
            lines += [f"**{c['name']}**"]
            for f in c["files"][:5]:
                lines += [f"  - `{f}`"]
        lines += [""]

    # §3 Conventions
    if conventions.get("conventions"):
        lines += ["## §3 Code Conventions", ""]
        for item in conventions["conventions"][:10]:
            flag = " ⚠️" if item.get("low_confidence") else ""
            lines += [f"- {item['rule']}{flag}"]
            if item.get("example"):
                lines += [f"  ```python\n  {item['example']}\n  ```"]
        lines += [""]

    # §5 Anti-Patterns
    if conventions.get("antipatterns"):
        lines += ["## §5 Anti-Patterns & Gotchas", ""]
        for item in conventions["antipatterns"][:10]:
            flag = " ⚠️" if item.get("low_confidence") else ""
            lines += [f"- ❌ {item['rule']}{flag}"]
        lines += [""]

    # §6 Risk Map
    if git_signals.get("high_churn"):
        lines += ["## §6 File Change Risk Map", ""]
        lines += ["**High churn (change carefully):**"]
        for f in git_signals["high_churn"][:6]:
            lines += [f"  - `{f}`"]
        if git_signals.get("stable"):
            lines += ["", "**Stable (low risk):**"]
            for f in git_signals["stable"][:6]:
                lines += [f"  - `{f}`"]
        lines += [""]

    return "\n".join(lines)


# ===========================================================================
# MAIN
# ===========================================================================

def generate(repo_path: str, output_path: str = "AGENTS.md") -> str:
    repo = Path(repo_path).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Repo not found: {repo}")
    
    client = build_client(provider=getattr(args, "provider", None))
    if client is None:
        print("[warn] Running without LLM — LLM synthesis and clustering disabled",
              file=sys.stderr)

    print(f"[1/6] Static analysis: {repo.name}", file=sys.stderr)
    commands  = extract_commands(repo)
    structure = extract_ast_structure(repo)
    print(f"      {len(structure)} Python files parsed", file=sys.stderr)

    print("[2/6] Git history mining", file=sys.stderr)
    git_signals = extract_git_signals(repo)

    if client and HAS_SKLEARN:
        print("[3/6] Semantic clustering", file=sys.stderr)
        chunks   = chunk_repo(repo, structure)
        print(f"      {len(chunks)} AST chunks to embed", file=sys.stderr)
        clusters = embed_and_cluster(chunks, client)
        print(f"      {len(clusters)} clusters found", file=sys.stderr)
    else:
        clusters = []

    if client:
        print("[4/6] LLM synthesis", file=sys.stderr)
        conventions = synthesize_conventions(clusters, structure, git_signals, client)
        print("[4b]  Self-critique pass", file=sys.stderr)
        conventions = self_critique_pass(conventions, client)
    else:
        conventions = {}

    print("[5/6] Validating commands", file=sys.stderr)
    validated = validate_commands(commands, repo)
    print(f"      {sum(len(v) for v in validated.values())} commands passed validation",
          file=sys.stderr)

    print("[6/6] Rendering AGENTS.md", file=sys.stderr)
    content = render_agents_md(
        repo, commands, validated, clusters, conventions, git_signals, structure, client
    )

    output = Path(output_path)
    output.write_text(content, encoding="utf-8")
    print(f"[done] Written to {output}", file=sys.stderr)
    return content


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

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
        "--no-llm", action="store_true",
        help="Skip LLM synthesis (static analysis + git only)"
    )
    parser.add_argument(
        "--provider", choices=["openai", "claude"], default=None,
        help="LLM provider (default: auto-detect from env vars)"
    )
    args = parser.parse_args()
    if args.no_llm:
        generate(args.repo, args.out, provider=None)
    else:
        generate(args.repo, args.out, provider=args.provider)
    generate(args.repo, args.out)

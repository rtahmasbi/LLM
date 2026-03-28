# Better AGENTS.md — Generation & Iterative Improvement

> Programmatically generate and improve `AGENTS.md` files — READMEs for AI coding agents.

---

## What is AGENTS.md?

An `AGENTS.md` file sits at the root of a repository and tells AI coding agents (Cursor, Copilot, Claude Code, etc.) how to work within the codebase effectively. While a `README.md` is written for humans, `AGENTS.md` is written for machines:

| README.md | AGENTS.md |
|-----------|-----------|
| Setup & installation | Exact, runnable test/lint commands |
| What the project does | Architecture map — where things live |
| How to contribute (prose) | Specific code conventions with examples |
| — | Anti-patterns & gotchas |
| — | File change risk map (what's safe to touch) |

A naive `AGENTS.md` just says "run pytest". A good one tells an agent which files are dangerous to modify, what the Pydantic v2 migration patterns look like, and that route order matters in FastAPI. This project generates the good kind.

---

## Files

| File | Description |
|------|-------------|
| `generate_agents_md.py` | Full generation pipeline from scratch (static analysis → git → embeddings → LLM → validated output) |
| `improve_agents_md.py` | Iterative improvement loop: score on 8 metrics → LLM rewrites targeting weak spots → repeat |
| `AGENTS_v0.md` | Minimal baseline — what a naive generator produces |
| `AGENTS.md` | Hand-crafted reference output for FastAPI showing the target quality |
| `agents_md_presentation.pptx` | slide deck covering approach, evaluation, tradeoffs |

---

## Quickstart

### Requirements

```bash
pip install openai numpy scikit-learn
export OPENAI_API_KEY=sk-...
```

### Option A — Generate from scratch

Runs the full 6-stage pipeline against any Python repo:

```bash
python generate_agents_md.py --repo /path/to/your/repo --out AGENTS.md
```

Without an OpenAI key (static analysis + git only, no LLM):

```bash
python generate_agents_md.py --repo /path/to/your/repo --out AGENTS.md --no-llm
```

### Option B — Start from a baseline and iteratively improve

```bash
python improve_agents_md.py \
    --repo    /path/to/your/repo \
    --input   AGENTS_v0.md \
    --iters   3 \
    --out-dir ./versions
```

This produces:

```
versions/
  AGENTS_v0.md    ← baseline (scored)
  AGENTS_v1.md    ← after 1 improvement pass
  AGENTS_v2.md    ← after 2 passes
  AGENTS_v3.md    ← final
  scores.json     ← full metric trajectory
```

---

## How It Works

### Generation Pipeline (`generate_agents_md.py`)

```
Stage 1 — Static Analysis     AST parsing, pyproject.toml, Makefile → exact commands
Stage 2 — Git History          Churn map, co-change coupling, branch patterns
Stage 3 — Semantic Clustering  Embed chunks → DBSCAN → named architecture clusters
Stage 4 — LLM Synthesis        Extract implicit conventions, anti-patterns, gotchas
Stage 5 — Validation           Run every command; drop failures before writing
Stage 6 — Render               Assemble into structured Markdown
```

The key principle: **static analysis and git history are ground truth** — they can't hallucinate. The LLM only synthesizes *prose* around those anchors, and its output is validated back against the filesystem before the file is written.

### Iterative Improvement Loop (`improve_agents_md.py`)

```
v0 (baseline) ──► evaluate ──► scorecard ──► improve (target weakest 3) ──► v1
                                                                              │
v1            ──► evaluate ──► scorecard ──► improve (target weakest 3) ──► v2
                                                                              │
v2            ──► evaluate ──► [plateau? stop] ─────────────────────────────►
```

The improver is given the **exact weak metric scores plus the evidence** (e.g. "Specificity 20%: found 4 vague phrases, 0 file paths") — not just a number. This produces targeted rewrites rather than generic improvements.

---

## The 8 Evaluation Metrics

Each metric is a pure function `(content: str, repo: Path) -> MetricScore`.

| Metric | What it catches | How measured |
|--------|----------------|--------------|
| **Command Validity** | Commands that don't actually run | `subprocess.run()` every shell command |
| **Section Coverage** | Missing high-value sections (anti-patterns, risk map) | Weighted regex over section headers |
| **Specificity** | Generic advice ("follow PEP 8") vs. concrete examples | Count file paths, vague phrases, code blocks |
| **Novelty vs README** | Restating what's already in README/CONTRIBUTING | 6-word window overlap against reference docs |
| **Information Density** | Filler lines, padding, empty headers | Useful lines / total lines ratio |
| **Actionability** | Sections with no concrete action an agent can take | Every H2/H3 must have code, bullets, or ✅/❌ |
| **Machine Parsability** | Untagged code blocks, walls of prose | Structural analysis of headings and blocks |
| **Repo Grounding** | Invented file paths and module names | Check every backtick path against the filesystem |

### Illustrative Score Trajectory (FastAPI)

| Version | Overall | Key improvement |
|---------|---------|----------------|
| v0 baseline | 42% | Minimal: only `pytest` and `ruff check .` |
| v1 | 67% | Added architecture map, specific commands |
| v2 | 82% | Added anti-patterns, ✅/❌ conventions |
| v3 | 90% | Added risk map, repo-grounded file paths |

---

## Defining "Better"

The proxies above are useful, but the only metric that truly matters is:

> **Does an agent make fewer mistakes on real tasks when using this file?**

The practical evaluation approach:
1. Pick 10–20 realistic tasks (add endpoint, fix bug, add test, refactor module)
2. Run the same agent on each task with your AGENTS.md vs. a baseline
3. Measure **first-attempt pass rate** — did the output pass tests + linter without human correction?

Static metrics (the 8 above) are necessary conditions. Task completion rate is the sufficient condition.

---

## Key Design Decisions

**Metrics are pure functions.** Each one is independently testable and swappable. Adding a new metric is a one-function addition with no changes to the rest of the pipeline.

**The improver sees evidence, not just scores.** Passing "Specificity: 20%" to the LLM is less useful than "Specificity: 20% — found 4 vague phrases ('where possible', 'best practices'), 0 backtick file paths, 1 code block." The latter produces targeted rewrites.

**Commands are validated before writing.** A command in §1 that doesn't run is actively harmful — it causes the agent to fail with false confidence. Every command is executed (or syntax-checked for install commands) before inclusion.

**Plateau detection stops early.** If the overall score improves by less than 2% between iterations, the loop stops. This avoids burning API budget on rewrites with diminishing returns.

**Graceful degradation.** No OpenAI key? The pipeline still runs stages 1 and 2 (static analysis + git) and produces a solid template-based AGENTS.md. Useful for CI environments or cost-sensitive runs.

---

## Limitations

- **Evaluation sample is illustrative.** The v0→v3 scores shown are directional estimates. Real scores depend on the repo and require human spot-checking of the anti-patterns section.
- **LLM synthesis can hallucinate.** The self-critique pass and command validation reduce this, but the anti-patterns section in particular should be reviewed by someone who knows the codebase.
- **Novelty ≠ correctness.** A high novelty score means the content isn't in the README — not that it's accurate. Novel content that's wrong is worse than no content.
- **Staleness.** An AGENTS.md without a refresh mechanism will rot. Consider triggering regeneration in CI on structural diffs (new files, changed public APIs), not every commit.
- **Model-dependent.** A/B test results are specific to the agent and model tested. GPT-4o, Claude, and Cursor's built-in agent may use the file differently.

---

## Project Target: FastAPI

This project was developed and tested against [FastAPI](https://github.com/tiangolo/fastapi) (80k+ GitHub stars, Python, active community). FastAPI was chosen because:

- Rich implicit conventions (Pydantic v2 migration, `Annotated[]` injection style) that aren't in the README
- Large active test suite makes command validation reliable
- Enough architectural complexity to stress-test the generation pipeline
- Familiar to many developers — easy to human-evaluate whether the output is accurate


# my run
```sh
cd /home/ras/temp/
git clone https://github.com/tiangolo/fastapi.git


cd ~/GITHUB/LLM/AgenticAI/improve_agent_programmatically
conda create -n improve_agent_programmatically python=3.11
conda activate improve_agent_programmatically
pip install -r requirements.txt

# OpenAI (with clustering)
python generate_agents_md.py --repo /home/ras/temp/fastapi --out out_AGENTS_fastapi_openai.md  --provider openai

# Claude (no clustering, all other stages run)
python generate_agents_md.py --repo /home/ras/temp/fastapi --out out_AGENTS_fastapi_claude.md --provider claude

# No LLM at all (free, static only)
python generate_agents_md.py --repo /home/ras/temp/fastapi --out out_AGENTS_fastapi_no_llm.md --no-llm



python improve_agents_md.py \
    --repo   /home/ras/temp/fastapi \
    --input  AGENTS_v0.md \
    --iters  3 \
    --out-dir ./versions


# Auto-detect (uses whichever key is set; OpenAI takes priority if both set)
python improve_agents_md.py --repo /home/ras/temp/fastapi --input AGENTS_v0.md --iters 3

# Force Claude
export ANTHROPIC_API_KEY=sk-ant-...
python improve_agents_md.py --repo /home/ras/temp/fastapi --input AGENTS_v0.md --provider claude

# Force OpenAI
export OPENAI_API_KEY=sk-...
python improve_agents_md.py --repo /home/ras/temp/fastapi --input AGENTS_v0.md --provider openai


```

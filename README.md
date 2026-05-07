# GAN Arena: Adversarial Backtrader MCQ Evaluation

This repository contains a self-contained arena for evaluating coding agents on multiple-choice questions that require executable reasoning. The primary task is Backtrader-based financial backtesting: a solver receives a question and must answer exactly in the form `<<< X >>>`, where `X` is one of `A`, `B`, `C`, or `D`.

The repo includes both an adversarial two-agent arena and a vendored deterministic Backtrader MCQ pipeline used to generate and verify benchmark questions.

## Repository Contents

- `arena.py` -- command-line entry point for matches, agent listing, database inspection, and exports.
- `agent_backends.py` -- local wrappers for supported non-interactive agent CLIs.
- `arena_core.py`, `arena_invoke.py`, `arena_match.py` -- arena logic, invocation, scoring, logging, and match orchestration.
- `backtrader_source.py` -- adapter between the arena and the vendored Backtrader MCQ pipeline.
- `run_cursor_mcq.py` -- standalone Cursor Agent CLI runner for the bundled MCQ dataset, with tools enabled.
- `run_cursor_sdk_mcq_notool.mjs` -- standalone Cursor SDK runner for the bundled MCQ dataset, with tool-use detection and cancellation.
- `Backtrader_MCQ/` -- standalone Backtrader question builder, checker, strategy implementations, Croissant metadata, and bundled JSONL datasets.
- `prompts/backtrader_agent.txt` -- generator prompt for LLM-authored Backtrader questions.
- `format.txt` -- solver prompt template.
- `requirements.txt` -- Python dependencies for the arena and Backtrader pipeline.

Bundled benchmark files:

- `Backtrader_MCQ/backtrader_mcq_balanced_30_all_strategies.jsonl` -- balanced 30-question evaluation set.
- `Backtrader_MCQ/backtrader_mcq_base_pool_all_strategies.jsonl` -- larger 160-question reference pool.
- `Backtrader_MCQ/croissant.json` -- dataset metadata.

## Setup

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Agent-based arena runs also require at least one supported agent CLI installed and authenticated. The current backend registry supports `cursor`, `claude`, `codex`, and `copilot`, depending on what is available on the machine.

```bash
.venv/bin/python arena.py agents
```

The no-tool Cursor SDK runner additionally needs Node.js, a Cursor API key, and the SDK package:

```bash
npm install @cursor/sdk
export CURSOR_API_KEY="cursor_..."
```

## Standalone Checks

These commands do not require access to the original development directory.

Compile all Python modules:

```bash
.venv/bin/python -m py_compile \
  arena.py arena_core.py arena_invoke.py arena_match.py backtrader_source.py \
  agent_backends.py db.py \
  Backtrader_MCQ/backtrader_MCQ.py \
  Backtrader_MCQ/backtrader_MCQ_builder.py \
  Backtrader_MCQ/backtrader_MCQ_checker.py \
  Backtrader_MCQ/backtrader_MCQ_pipeline.py \
  Backtrader_MCQ/backtrader_info.py \
  Backtrader_MCQ/strategies.py
```

Generate and verify one Backtrader MCQ:

```bash
mkdir -p Backtrader_MCQ/output
.venv/bin/python Backtrader_MCQ/backtrader_MCQ_builder.py \
  --symbol AAPL \
  --start-date 2020-01-01 \
  --end-date 2020-06-01 \
  --initial-cash 50000 \
  --commission-rate 0.001 \
  --stake 1000 \
  --strategy sma_crossover \
  --num-questions 1 \
  --difficulty easy \
  --seed 42 \
  -o Backtrader_MCQ/output/standalone_smoke.json

.venv/bin/python Backtrader_MCQ/backtrader_MCQ_checker.py \
  Backtrader_MCQ/output/standalone_smoke.json
```

`Backtrader_MCQ/output/` is ignored by git.

## Standalone Cursor MCQ Runners

The repo includes two standalone runners for comparing Cursor on the bundled Backtrader MCQ benchmark without running the full adversarial arena.

### Cursor Agent CLI With Tools

Use `run_cursor_mcq.py` when you want Cursor's local `agent -p` CLI to solve each MCQ with tools enabled. Each question runs in its own workspace under `cursor_agent_runs/`, and the runner writes per-question prompts, responses, stderr, `summary.jsonl`, and `report.json`.

```bash
.venv/bin/python run_cursor_mcq.py \
  --input Backtrader_MCQ/backtrader_mcq_balanced_30_all_strategies.jsonl \
  --model composer-2 \
  --timeout 180
```

Useful options:

```bash
# Show models supported by the installed Cursor Agent CLI, if available.
.venv/bin/python run_cursor_mcq.py --list-models

# Use a custom Cursor Agent binary and pass additional CLI flags.
.venv/bin/python run_cursor_mcq.py \
  --agent-bin agent \
  --extra-arg=--some-cursor-flag \
  --extra-arg some-value
```

The tool-enabled prompt explicitly allows code execution and workspace inspection, so this mode is intended to measure tool-using problem solving.

### Cursor SDK No Tools

Use `run_cursor_sdk_mcq_notool.mjs` when you want a Cursor SDK local agent run that should answer from the prompt only. The runner disables ambient settings, passes no MCP servers or subagents, enables sandboxing, watches the event stream for tool calls, and cancels the run if tool use appears.

```bash
export CURSOR_API_KEY="cursor_..."

node run_cursor_sdk_mcq_notool.mjs \
  --input Backtrader_MCQ/backtrader_mcq_balanced_30_all_strategies.jsonl \
  --model composer-2 \
  --timeout 120
```

Outputs are written under `cursor_sdk_notools_runs/`. Check `report.json` for aggregate accuracy and `tool_attempts`; each question directory includes `event_summaries.jsonl` and `tool_events.json` for auditing whether the model tried to use tools.

## Running the Arena

The default arena mode uses an LLM generator and an LLM solver:

```bash
.venv/bin/python arena.py --rounds 10
```

For a guided interactive match configuration, run:

```bash
.venv/bin/python arena.py --backtrader-interactive
```

The interactive flow first asks for arena-level settings such as round count, model, generator and solver agent backends, and memory modes. For Backtrader-backed modes it then opens the Backtrader config editor, where you can choose the symbol, date range, initial cash, commission, stake, strategy, seed, and difficulty. Press Enter to keep each shown default.

To use the deterministic Backtrader pipeline as the question source, run:

```bash
.venv/bin/python arena.py \
  --generator-source backtrader \
  --backtrader-difficulty easy \
  --no-backtrader-prompt \
  --rounds 10
```

The deterministic source uses `Backtrader_MCQ/backtrader_MCQ_builder.py` from this repository. It checks the project `.venv` first for `backtrader`, `pandas`, and `yfinance`, so no external pipeline path is required.

You can also combine deterministic Backtrader questions with the interactive editor:

```bash
.venv/bin/python arena.py \
  --generator-source backtrader \
  --backtrader-interactive
```

If you only want the quick difficulty picker for deterministic Backtrader questions, omit `--backtrader-interactive` and leave off `--backtrader-difficulty`. For non-interactive scripts, pass `--backtrader-difficulty easy` or `--no-backtrader-prompt`.

Common options:

```bash
# Cross-agent run.
.venv/bin/python arena.py --rounds 10 \
  --generator-agent claude \
  --solver-agent cursor

# Use the same memory regime for both roles.
.venv/bin/python arena.py --rounds 10 --memory sandbox

# Override Backtrader generation settings.
.venv/bin/python arena.py \
  --generator-source backtrader \
  --backtrader-difficulty hard \
  --backtrader-symbol MSFT \
  --backtrader-start-date 2018-01-01 \
  --backtrader-end-date 2024-01-01 \
  --backtrader-seed 42 \
  --rounds 20
```

## Outputs

Runtime outputs are intentionally excluded from version control:

- `logs/` -- per-match JSONL summaries, prompts, stdout/stderr, and scratch directories.
- `cursor_agent_runs/` -- standalone Cursor Agent CLI MCQ runs.
- `cursor_sdk_notools_runs/` -- standalone Cursor SDK no-tool MCQ runs.
- `db/questions.jsonl` -- deduplicated question records.
- `db/attempts.jsonl` -- solver attempts and optional usage metadata.
- `.venv/`, Python caches, generated Backtrader outputs, local env files, and editor metadata.

Inspect or export the local question database:

```bash
.venv/bin/python arena.py list --limit 20
.venv/bin/python arena.py export bank.md --well-defined-only
```

## Notes for Review

The repository is intended to run as a standalone artifact. The Backtrader pipeline, strategy definitions, benchmark JSONL files, prompt templates, and arena backend registry are included locally. Network access may be needed when generating fresh Backtrader questions because `yfinance` downloads historical market data.

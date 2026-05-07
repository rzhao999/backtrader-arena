# GAN Arena: Adversarial Backtrader MCQ Evaluation

This repository evaluates coding agents on multiple-choice questions that require executable reasoning. The main benchmark is Backtrader-based financial backtesting: each solver receives a question and must answer exactly in the form `<<< X >>>`, where `X` is one of `A`, `B`, `C`, or `D`.

The repo supports three main workflows:

- Run Cursor on the bundled MCQ dataset with tools enabled.
- Run Cursor on the same dataset with tool use blocked and audited.
- Run a two-agent GAN-style arena where one agent generates questions and another solves them.

## Dataset

The bundled benchmark files are:

- `Backtrader_MCQ/backtrader_mcq_balanced_30_all_strategies.jsonl` -- balanced 30-question evaluation set.
- `Backtrader_MCQ/backtrader_mcq_base_pool_all_strategies.jsonl` -- larger 160-question reference pool.
- `Backtrader_MCQ/croissant.json` -- dataset metadata.

Evaluating the bundled JSONL files does not require downloading market data. Network access is only needed if you regenerate or re-verify questions from raw Backtrader runs, because that generation path uses `yfinance` to fetch historical prices.

## Setup

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Agent-based runs require the relevant agent CLI to be installed and authenticated. The arena backend registry currently supports `cursor`, `claude`, `codex`, and `copilot`, depending on what is available on the machine.

```bash
.venv/bin/python arena.py agents
```

The no-tool Cursor runner also requires Node.js, a Cursor API key, and the Cursor SDK package:

```bash
npm install @cursor/sdk
export CURSOR_API_KEY="cursor_..."
```

## Run Cursor With Tools

Use `run_cursor_mcq.py` to evaluate Cursor's local `agent -p` CLI with tools enabled. The prompt explicitly allows code execution and workspace inspection, so this mode measures tool-using problem solving.

```bash
.venv/bin/python run_cursor_mcq.py \
  --input Backtrader_MCQ/backtrader_mcq_balanced_30_all_strategies.jsonl \
  --model composer-2 \
  --timeout 180
```

Each question runs in its own workspace under `cursor_agent_runs/`. The runner writes prompts, responses, stderr, `summary.jsonl`, and `report.json`.

Useful options:

```bash
# Show models supported by the installed Cursor Agent CLI.
.venv/bin/python run_cursor_mcq.py --list-models

# Use a custom Cursor Agent binary and pass additional CLI flags.
.venv/bin/python run_cursor_mcq.py \
  --agent-bin agent \
  --extra-arg=--some-cursor-flag \
  --extra-arg some-value
```

## Run Cursor Without Tools

Use `run_cursor_sdk_mcq_notool.mjs` to evaluate Cursor in a no-tool setting. This runner uses the Cursor SDK local agent path, disables ambient settings, passes no MCP servers or subagents, enables sandboxing, watches the event stream for tool calls, and cancels the run if tool use appears.

```bash
export CURSOR_API_KEY="cursor_..."

node run_cursor_sdk_mcq_notool.mjs \
  --input Backtrader_MCQ/backtrader_mcq_balanced_30_all_strategies.jsonl \
  --model composer-2 \
  --timeout 120
```

Outputs are written under `cursor_sdk_notools_runs/`. Check `report.json` for aggregate accuracy and `tool_attempts`; each question directory includes `event_summaries.jsonl` and `tool_events.json` for auditing whether the model tried to use tools.

## Run the GAN Arena

The arena runs a generator agent against a solver agent. By default, both roles use the configured LLM agent backends:

```bash
.venv/bin/python arena.py --rounds 10
```

For a guided interactive setup, run:

```bash
.venv/bin/python arena.py --backtrader-interactive
```

The interactive flow asks for arena settings such as round count, model, generator and solver backends, and memory modes. For Backtrader-backed modes, it also opens a config editor for symbol, date range, initial cash, commission, stake, strategy, seed, and difficulty.

To use the deterministic Backtrader pipeline as the question source:

```bash
.venv/bin/python arena.py \
  --generator-source backtrader \
  --backtrader-difficulty easy \
  --no-backtrader-prompt \
  --rounds 10
```

To run a cross-agent match:

```bash
.venv/bin/python arena.py --rounds 10 \
  --generator-agent claude \
  --solver-agent cursor
```

Common arena options:

```bash
# Use the same memory regime for both roles.
.venv/bin/python arena.py --rounds 10 --memory sandbox

# Combine deterministic Backtrader questions with the interactive editor.
.venv/bin/python arena.py \
  --generator-source backtrader \
  --backtrader-interactive

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

## Repository Layout

- `arena.py` -- command-line entry point for matches, agent listing, database inspection, and exports.
- `agent_backends.py` -- local wrappers for supported non-interactive agent CLIs.
- `arena_core.py`, `arena_invoke.py`, `arena_match.py` -- arena logic, invocation, scoring, logging, and match orchestration.
- `backtrader_source.py` -- adapter between the arena and the vendored Backtrader MCQ pipeline.
- `run_cursor_mcq.py` -- Cursor Agent CLI runner for the bundled MCQ dataset, with tools enabled.
- `run_cursor_sdk_mcq_notool.mjs` -- Cursor SDK runner for the bundled MCQ dataset, with tool-use detection and cancellation.
- `Backtrader_MCQ/` -- Backtrader question builder, checker, strategy implementations, Croissant metadata, and bundled JSONL datasets.
- `prompts/backtrader_agent.txt` -- generator prompt for LLM-authored Backtrader questions.
- `format.txt` -- solver prompt template.
- `requirements.txt` -- Python dependencies for the arena and Backtrader pipeline.

## Outputs

Runtime outputs are intentionally excluded from version control:

- `logs/` -- per-match JSONL summaries, prompts, stdout/stderr, and scratch directories.
- `cursor_agent_runs/` -- Cursor Agent CLI runs with tools enabled.
- `cursor_sdk_notools_runs/` -- Cursor SDK runs with tool use blocked and audited.
- `db/questions.jsonl` -- deduplicated question records.
- `db/attempts.jsonl` -- solver attempts and optional usage metadata.
- `.venv/`, Python caches, generated Backtrader outputs, local env files, and editor metadata.

Inspect or export the local question database:

```bash
.venv/bin/python arena.py list --limit 20
.venv/bin/python arena.py export bank.md --well-defined-only
```

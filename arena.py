"""
Adversarial MCQ arena: a zero-sum game between two AI coding agents.

  - GENERATOR: produces (or drives production of) a multiple-choice
    question + self-contained Python verification code.
  - SOLVER:    receives the question rendered through `format.txt` and must
    answer with `<<< X >>>`. Runs in an isolated per-round scratch directory
    with full execution permissions.

Generator sources (``--generator-source``):
  - ``backtrader``        -- pre-generate the whole queue from the
                             deterministic Backtrader MCQ pipeline; each
                             round replays one pre-verified question.
                             No LLM in the generator loop.
  - ``backtrader_agent``  -- LLM-authored Backtrader MCQ mode (default).
                             An LLM generator CLI agent is invoked per
                             round and writes the full MCQ plus executable
                             verification code, using Backtrader reference
                             material and match history.

Ground truth for each round is taken from the verification code's printed
`ANSWER: X` line (this overrides the generator's own declaration).

Scoring (per round):
    solver correct  -> solver +1, generator -1
    solver wrong    -> solver -1, generator +1
    generator code fails / unparseable -> generator forfeits (gen -1, solver +1)

The generator and solver can be ANY LLM CLI agent registered in the local
arena backend registry: currently ``claude``, ``cursor`` (Cursor's ``agent``
CLI, the default for BOTH roles), ``codex``, and ``copilot``. Mix and match
freely, e.g.::

    python arena.py --rounds 10 --generator-agent claude --solver-agent cursor

With no flags, ``python arena.py`` runs ``backtrader_agent`` with the Cursor
LLM CLI as both the generator driver and solver.

Per-round and per-match token / cost usage is auto-captured for each role
via ``agent_backends.extract_usage`` -- no per-agent plumbing is required.

This file is the CLI entry point. The implementation is split across::

    arena_core.py     -- constants, paths, RoundResult, prompts, parsing,
                         console helpers, and verification-code execution
    arena_invoke.py   -- AgentInvocation + agent invocation + interactive cfg
    arena_match.py    -- LLM Backtrader generator, play_round, run_match,
                         and end-of-match summaries
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import subprocess
import sys
from pathlib import Path

import arena_core as _core
from arena_core import (
    AGENT_TIMEOUT_S,
    CODE_TIMEOUT_S,
    DEFAULT_GENERATOR_AGENT,
    DEFAULT_GENERATOR_MEMORY,
    DEFAULT_GENERATOR_SOURCE,
    DEFAULT_MODEL,
    DEFAULT_SOLVER_AGENT,
    DEFAULT_SOLVER_MEMORY,
    GENERATOR_SOURCES,
    MEMORY_MODES,
)
import db  # noqa: E402
from agent_backends import get_agent, list_agents  # noqa: E402
import backtrader_source as bt_source  # noqa: E402
from backtrader_source import BacktraderGenConfig  # noqa: E402
from arena_core import (
    _BOLD,
    _DIM,
    _GREEN,
    _RED,
    _YELLOW,
    _c,
)
from arena_match import run_match


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_match_args(p: argparse.ArgumentParser) -> None:
    agents_available = ", ".join(list_agents()) or "(none registered)"
    p.add_argument("--rounds", type=int, default=10,
                   help="Number of rounds in the match (default: 10).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=(
                       f"Default model slug used for BOTH generator and "
                       f"solver (default: {DEFAULT_MODEL}). Override per "
                       "role with --generator-model / --solver-model. "
                       "Forwarded automatically as ``--model X`` when using "
                       "the cursor agent; ignored for other agents (pass "
                       "your own model via --generator-extra/--solver-extra)."
                   ))
    p.add_argument("--generator-model", default=None,
                   help=(
                       "Override the generator-side model. Falls back to "
                       "--model when omitted."
                   ))
    p.add_argument("--solver-model", default=None,
                   help=(
                       "Override the solver-side model. Falls back to "
                       "--model when omitted."
                   ))
    p.add_argument("--generator-agent", default=DEFAULT_GENERATOR_AGENT,
                   choices=list_agents() or [DEFAULT_GENERATOR_AGENT],
                   help=(
                       f"Registered LLM CLI agent used as GENERATOR "
                       f"(default: {DEFAULT_GENERATOR_AGENT}; "
                       f"available: {agents_available})."
                   ))
    p.add_argument("--solver-agent", default=DEFAULT_SOLVER_AGENT,
                   choices=list_agents() or [DEFAULT_SOLVER_AGENT],
                   help=(
                       f"Registered LLM CLI agent used as SOLVER "
                       f"(default: {DEFAULT_SOLVER_AGENT}; "
                       f"available: {agents_available})."
                   ))
    p.add_argument("--generator-extra", action="append", default=[],
                   metavar="ARG",
                   help="Extra argv token forwarded verbatim to the generator "
                        "agent CLI (repeatable).")
    p.add_argument("--solver-extra", action="append", default=[],
                   metavar="ARG",
                   help="Extra argv token forwarded verbatim to the solver "
                        "agent CLI (repeatable).")
    p.add_argument("--generator-memory", default=DEFAULT_GENERATOR_MEMORY,
                   choices=list(MEMORY_MODES),
                   help=(
                       f"Information regime for the GENERATOR "
                       f"(default: {DEFAULT_GENERATOR_MEMORY}). "
                       "'sandbox': every round is an independent session with "
                       "no memory of prior rounds. 'history': the generator "
                       "sees a recap of prior rounds and can adapt."
                   ))
    p.add_argument("--solver-memory", default=DEFAULT_SOLVER_MEMORY,
                   choices=list(MEMORY_MODES),
                   help=(
                       f"Information regime for the SOLVER "
                       f"(default: {DEFAULT_SOLVER_MEMORY}). "
                       "'sandbox': every round is an independent session with "
                       "no memory of prior rounds. 'history': the solver "
                       "sees its own past attempts (question preview, its "
                       "answer, ground truth, correctness)."
                   ))
    p.add_argument("--memory", default=None, choices=list(MEMORY_MODES),
                   help="Shortcut: set both --generator-memory and "
                        "--solver-memory to the same value (symmetric "
                        "information regime). Overrides the per-role flags.")
    # ---- Question-source selection (Backtrader vs LLM-driven Backtrader) ----
    p.add_argument("--generator-source", default=DEFAULT_GENERATOR_SOURCE,
                   choices=list(GENERATOR_SOURCES),
                   help=(
                       f"Where questions come from each round (default: "
                       f"{DEFAULT_GENERATOR_SOURCE}). "
                       "'backtrader' = pre-generate the full queue from "
                       "the Backtrader MCQ pipeline; each round replays "
                       "one pre-verified question (deterministic; no LLM "
                       "in the generator loop). In this mode you'll be "
                       "prompted for the difficulty level unless "
                       "--backtrader-difficulty is set. "
                       "'backtrader_agent' = LLM-authored Backtrader MCQ "
                       "mode (default). The generator CLI writes the full "
                       "MCQ plus executable verification code each round. "
                       "Combine with --generator-memory=history so the "
                       "LLM can learn from the solver's past answers."
                   ))
    p.add_argument("--backtrader-difficulty", default=None,
                   help=(
                       "Skip the interactive difficulty prompt and use this "
                       f"value directly. One of: "
                       f"{', '.join(bt_source.DIFFICULTY_CHOICES)}."
                   ))
    p.add_argument("--no-backtrader-prompt", action="store_true",
                   help="Never prompt interactively for backtrader "
                        "difficulty. With no --backtrader-difficulty, "
                        "defaults to 'easy'. Useful for scripted runs.")
    p.add_argument("--backtrader-interactive", "-I", action="store_true",
                   help="Open the FULL interactive match config editor "
                        "at match start. Stage 1 (always): arena-level "
                        "knobs -- rounds, model, generator/solver agent "
                        "backend, generator/solver memory mode. Stage 2 "
                       "(if --generator-source is backtrader or "
                       "backtrader_agent): every BacktraderGenConfig "
                        "field (symbol / dates / cash / commission / "
                        "stake / strategy / seed) plus difficulty. For "
                       "backtrader_agent the stage-2 values are reference "
                        "context for the LLM-authored MCQ prompt. "
                        "Without this flag, backtrader mode only shows "
                        "the small difficulty picker. Auto-skipped if "
                        "stdin is not a TTY.")
    p.add_argument("--backtrader-dir", default=None,
                   help="Override the path to the Backtrader MCQ pipeline "
                        f"(default: {bt_source.BACKTRADER_DIR_DEFAULT}, "
                        "or $BACKTRADER_MCQ_DIR).")
    p.add_argument("--backtrader-python", default=None,
                   help="Python interpreter used to run the Backtrader "
                        "pipeline. Defaults to auto-detection of an "
                        "interpreter with backtrader/pandas/yfinance "
                        "installed; override with $BACKTRADER_MCQ_PYTHON "
                        "or this flag.")
    p.add_argument("--backtrader-symbol", default=None,
                   help="Ticker symbol used by the Backtrader source or as "
                        "reference context for backtrader_agent (e.g. AAPL, MSFT).")
    p.add_argument("--backtrader-start-date", default=None,
                   help="YYYY-MM-DD start date for the backtrader backtest.")
    p.add_argument("--backtrader-end-date", default=None,
                   help="YYYY-MM-DD end date for the backtrader backtest.")
    p.add_argument("--backtrader-seed", type=int, default=None,
                   help="Random seed for backtrader question sampling "
                        "(reproducibility).")
    p.add_argument(
        "--backtrader-strategy",
        default=None,
        choices=list(bt_source.REGISTERED_BACKTRADER_STRATEGIES),
        help=(
            "Trading strategy forwarded to the Backtrader MCQ pipeline "
            f"(default: {bt_source.DEFAULT_STRATEGY}). "
            f"Choices: {', '.join(bt_source.REGISTERED_BACKTRADER_STRATEGIES)}."
        ),
    )
    p.add_argument("--topic-hint", default=None,
                   help="Optional non-binding hint shown to the generator "
                        "(e.g. 'numerical linear algebra').")
    p.add_argument("--agent-timeout", type=int, default=AGENT_TIMEOUT_S,
                   help=f"Per-agent-invocation timeout in seconds "
                        f"(default: {AGENT_TIMEOUT_S}).")
    p.add_argument("--code-timeout", type=int, default=CODE_TIMEOUT_S,
                   help=f"Per-round verification-code timeout in seconds "
                        f"(default: {CODE_TIMEOUT_S}).")


def cmd_list(args: argparse.Namespace) -> None:
    """Print recent questions from the database."""
    qs = list(db.filter_questions(
        well_defined_only=args.well_defined_only,
        ill_defined_only=args.ill_defined_only,
    ))
    if args.limit:
        qs = qs[-args.limit:]

    if not qs:
        print(_c("No questions match those filters.", _YELLOW))
        return

    n_total = sum(1 for _ in db.iter_questions())
    n_well = sum(1 for q in db.iter_questions() if q.get("well_defined"))
    print(_c(
        f"Database: {n_total} unique questions  "
        f"({n_well} well-defined, {n_total - n_well} ill-defined)",
        _BOLD,
    ))
    print(_c(f"Showing {len(qs)} entr{'y' if len(qs) == 1 else 'ies'}\n", _DIM))

    for q in qs:
        wd = q.get("well_defined")
        wd_color = _GREEN if wd else _RED
        wd_label = "WELL-DEFINED" if wd else "ILL-DEFINED"
        first_line = (q.get("question") or "").strip().splitlines()
        first_line = first_line[0][:140] if first_line else "(no question text)"
        stats = db.question_stats(q["id"])
        print(_c(f"[{q['id'][:12]}] {wd_label}", wd_color + _BOLD), end="  ")
        print(_c(
            f"first_seen={q.get('first_seen','?')[:19]}  "
            f"truth={q.get('ground_truth') or '?'}  "
            f"declared={q.get('declared_answer') or '?'}  "
            f"attempts={stats['attempts']} solver_correct={stats['solver_correct']}",
            _DIM,
        ))
        print(f"  Q: {first_line}")
        if not wd:
            for r in q.get("ill_defined_reasons", []):
                print(_c(f"     - {r}", _RED))
        print()


def cmd_export(args: argparse.Namespace) -> None:
    """Write a Markdown dump of the question DB."""
    out_path = Path(args.output).expanduser().resolve()
    qs = list(db.filter_questions(
        well_defined_only=args.well_defined_only,
        ill_defined_only=args.ill_defined_only,
    ))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write("# Adversarial MCQ -- question bank\n\n")
        f.write(f"_Exported: {_dt.datetime.now().isoformat(timespec='seconds')}_\n\n")
        f.write(f"Total entries in this export: **{len(qs)}**\n\n")
        f.write("---\n\n")
        for i, q in enumerate(qs, 1):
            stats = db.question_stats(q["id"])
            wd = "WELL-DEFINED" if q.get("well_defined") else "ILL-DEFINED"
            f.write(f"## {i}. `{q['id'][:12]}` -- {wd}\n\n")
            f.write(f"- First seen: `{q.get('first_seen','?')}` "
                    f"(round {q.get('first_seen_round','?')} of "
                    f"`{q.get('first_seen_match','?')}`)\n")
            f.write(f"- Generator model: `{q.get('generator_model','?')}`\n")
            if q.get("topic_hint"):
                f.write(f"- Topic hint: {q['topic_hint']}\n")
            f.write(f"- Ground truth (from code): **{q.get('ground_truth') or '?'}**  "
                    f"Declared: {q.get('declared_answer') or '?'}\n")
            f.write(f"- Solver attempts: {stats['attempts']}  "
                    f"correct: {stats['solver_correct']}\n")
            if not q.get("well_defined"):
                f.write("- Ill-defined reasons:\n")
                for r in q.get("ill_defined_reasons", []):
                    f.write(f"  - {r}\n")
            f.write("\n### Question\n\n")
            f.write(q.get("question", "").strip() + "\n\n")
            f.write("### Options\n\n")
            for k in ("A", "B", "C", "D"):
                f.write(f"- **{k}.** {q['options'].get(k,'').strip()}\n")
            f.write("\n### Verification code\n\n")
            f.write("```python\n")
            f.write(q.get("code", "").rstrip() + "\n")
            f.write("```\n\n")
            if q.get("code_stdout"):
                f.write("### Code stdout\n\n```\n")
                f.write(q["code_stdout"].rstrip() + "\n```\n\n")
            f.write("---\n\n")

    print(_c(f"Wrote {len(qs)} question(s) to {out_path}", _GREEN))


# Per-agent capability matrix used by ``cmd_agents``. Keys mirror the
# names registered in agent_backends.py. Token tracking depends on what
# JSON / NDJSON the underlying CLI emits (extracted by
# ``agent_backends.extract_usage``):
_AGENT_CAPABILITIES = {
    "claude":  {"non_interactive": True,  "tokens": True,
                "note": "claude -p ... --output-format json (single JSON object on stdout)"},
    "cursor":  {"non_interactive": True,  "tokens": True,
                "note": "agent -p --output-format json --yolo --trust (forward-compat token adapter)"},
    "codex":   {"non_interactive": True,  "tokens": True,
                "note": "codex exec --json --skip-git-repo-check (NDJSON token_count events)"},
    "copilot": {"non_interactive": True,  "tokens": False,
                "note": "copilot --prompt/-p ... --output-format json (usage capture best-effort; many CLIs still omit tokens)"},
}


def cmd_agents(args: argparse.Namespace) -> None:
    """Print the agents registered in the local backend registry, including
    each one's binary location, non-interactive support, and whether
    token / cost usage is auto-captured by agent_backends.extract_usage.

    All agents listed here are valid choices for both ``--solver-agent``
    and ``--generator-agent``.
    """
    names = list_agents()
    if not names:
        print(_c("No agents registered.", _YELLOW))
        return
    print(_c("Registered agents (local registry):", _BOLD))
    print(_c(
        "  use as --solver-agent <name>  or  --generator-agent <name>",
        _DIM,
    ))
    print()
    header = (
        f"  {'NAME':<9} {'BIN':<10} {'STATUS':<8} {'NON-INT':<8} "
        f"{'TOKENS':<7} PATH"
    )
    print(_c(header, _BOLD))
    print(_c("  " + "-" * (len(header) + 30), _DIM))
    for name in names:
        ag = get_agent(name)
        assert ag is not None
        where = shutil.which(ag.bin_default)
        status_text = "found" if where else "missing"
        caps = _AGENT_CAPABILITIES.get(name, {})
        ni_text = "yes" if caps.get("non_interactive") else "?"
        tok_text = (
            "yes" if caps.get("tokens")
            else "no" if caps.get("tokens") is False
            else "?"
        )
        # Build the row with ALIGNMENT first (no ANSI in widths), then
        # color individual cells via inline replace -- this keeps the
        # columns lined up regardless of terminal color support.
        row = (
            f"  {name:<9} {ag.bin_default:<10} {status_text:<8} "
            f"{ni_text:<8} {tok_text:<7} {where or '(not on PATH)'}"
        )
        # Colorize the status / tokens / path cells in place.
        row = row.replace(
            f" {status_text:<8} ",
            " " + _c(f"{status_text:<8}", _GREEN if where else _RED) + " ",
            1,
        )
        if caps.get("tokens") is True:
            row = row.replace(
                f" {tok_text:<7} ",
                " " + _c(f"{tok_text:<7}", _GREEN) + " ",
                1,
            )
        elif caps.get("tokens") is False:
            row = row.replace(
                f" {tok_text:<7} ",
                " " + _c(f"{tok_text:<7}", _YELLOW) + " ",
                1,
            )
        if not where:
            row = row.replace("(not on PATH)", _c("(not on PATH)", _DIM))
        print(row)
        note = caps.get("note")
        if note:
            print(_c(f"             -> {note}", _DIM))
    print()
    print(_c(
        "Token totals are accumulated per role and shown at end-of-match. "
        "Agents marked 'tokens=no' have their CLI emit no usage payload, "
        "so their per-round records simply omit the 'usage' block.",
        _DIM,
    ))

    if not getattr(args, "list_models", False):
        return

    # Best-effort model listing (CLI has multiple variants).
    target = getattr(args, "agent", None) or "claude"
    if target not in {"claude", "cursor", "copilot", "codex"}:
        print(_c(
            f"\nModel listing currently supported for 'claude', 'cursor', 'copilot', and 'codex' only (got {target!r}).",
            _YELLOW,
        ))
        return

    ag = get_agent(target)
    if ag is None:
        print(_c(f"\nAgent {target!r} is not registered.", _YELLOW))
        return
    bin_name = ag.bin_default
    if shutil.which(bin_name) is None:
        print(_c(f"\nAgent binary not found on PATH: {bin_name!r}", _YELLOW))
        return

    label = (
        "Claude" if target == "claude"
        else "Cursor Agent" if target == "cursor"
        else "Copilot" if target == "copilot"
        else "Codex"
    )
    print(_c(f"\n{label} models (from CLI):", _BOLD))
    candidates: list[list[str]] = [
        [bin_name, "models"],
        [bin_name, "models", "--json"],
        [bin_name, "--list-models"],
        [bin_name, "list-models"],
        [bin_name, "model", "list"],
    ]
    out = ""
    rc = 1
    for cmd in candidates:
        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            continue
        stdout = (cp.stdout or "").strip()
        stderr = (cp.stderr or "").strip()
        combined = "\n".join(s for s in (stdout, stderr) if s).strip()
        if combined:
            out = combined
            rc = cp.returncode
            break
    if out:
        print(out)
        if rc != 0:
            print(_c("\n(note: CLI returned non-zero, but printed output above)", _DIM))
    else:
        print(_c(
            f"Could not list models via the {label} CLI. "
            "Tried: `<bin> models`, `<bin> --list-models`, `<bin> model list`.",
            _YELLOW,
        ))


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run a zero-sum adversarial MCQ match between two AI coding "
            "agents, or inspect the question database. Both roles can be "
            "filled by any locally registered backend (claude, cursor, "
            "codex, copilot)."
        ),
    )
    sub = p.add_subparsers(dest="cmd")

    pm = sub.add_parser("match", help="Run a match (default).")
    _add_match_args(pm)

    pl = sub.add_parser("list", help="List questions in the database.")
    pl.add_argument("--limit", type=int, default=20,
                    help="Show the most recent N entries (default: 20). "
                         "Use 0 for unlimited.")
    g = pl.add_mutually_exclusive_group()
    g.add_argument("--well-defined-only", action="store_true",
                   help="Only show well-defined questions.")
    g.add_argument("--ill-defined-only", action="store_true",
                   help="Only show ill-defined questions.")

    pe = sub.add_parser("export", help="Export the database to Markdown.")
    pe.add_argument("output", help="Output path (e.g. bank.md).")
    g2 = pe.add_mutually_exclusive_group()
    g2.add_argument("--well-defined-only", action="store_true",
                    help="Only export well-defined questions.")
    g2.add_argument("--ill-defined-only", action="store_true",
                    help="Only export ill-defined questions.")

    pa = sub.add_parser("agents", help="List locally registered agent backends.")
    pa.add_argument("--list-models", action="store_true",
                    help="Also list available models for a given agent (best-effort).")
    pa.add_argument("--agent", default="claude",
                    help="Agent name for --list-models (default: claude).")

    # Allow legacy: `python arena.py --rounds N` -> implicit `match` subcommand.
    raw = sys.argv[1:]
    known = {"match", "list", "export", "agents", "-h", "--help"}
    if raw and raw[0] not in known and not raw[0].startswith("-h"):
        if raw[0].startswith("-"):
            raw = ["match", *raw]
    elif not raw:
        raw = ["match"]

    args = p.parse_args(raw)

    if args.cmd == "list":
        cmd_list(args)
        return
    if args.cmd == "export":
        cmd_export(args)
        return
    if args.cmd == "agents":
        cmd_agents(args)
        return

    # Default: run a match. Mutate the module-level timeouts on
    # ``arena_core`` so submodules that read them via attribute access
    # (``_core.AGENT_TIMEOUT_S``) pick up the override.
    _core.AGENT_TIMEOUT_S = args.agent_timeout
    _core.CODE_TIMEOUT_S = args.code_timeout
    # OS-level sandboxing is now tied to the memory regime:
    # role memory="sandbox" => role runs in OS sandbox (best-effort, macOS).

    gen_memory = args.generator_memory
    sol_memory = args.solver_memory
    if args.memory is not None:
        gen_memory = sol_memory = args.memory

    # Only forward fields the user actually set on the CLI; otherwise
    # ``BacktraderGenConfig`` falls back to its own arena-side defaults
    # (AAPL 2020-2024 etc., see backtrader_source.py).
    bt_overrides: dict = {}
    if args.backtrader_symbol:
        bt_overrides["symbol"] = args.backtrader_symbol
    if args.backtrader_start_date:
        bt_overrides["start_date"] = args.backtrader_start_date
    if args.backtrader_end_date:
        bt_overrides["end_date"] = args.backtrader_end_date
    if args.backtrader_seed is not None:
        bt_overrides["seed"] = args.backtrader_seed
    if args.backtrader_strategy:
        bt_overrides["strategy"] = args.backtrader_strategy
    bt_cfg = BacktraderGenConfig(**bt_overrides)
    bt_dir = Path(args.backtrader_dir).expanduser() if args.backtrader_dir else None

    gen_model = args.generator_model or args.model
    sol_model = args.solver_model or args.model

    run_match(
        args.rounds,
        generator_model=gen_model,
        solver_model=sol_model,
        generator_agent_name=args.generator_agent,
        solver_agent_name=args.solver_agent,
        generator_extra=list(args.generator_extra),
        solver_extra=list(args.solver_extra),
        generator_memory=gen_memory,
        solver_memory=sol_memory,
        topic_hint=args.topic_hint,
        generator_source=args.generator_source,
        backtrader_difficulty=args.backtrader_difficulty,
        backtrader_dir=bt_dir,
        backtrader_python=args.backtrader_python,
        backtrader_cfg=bt_cfg,
        backtrader_interactive=not args.no_backtrader_prompt,
        backtrader_full_interactive=args.backtrader_interactive,
    )


if __name__ == "__main__":
    main()

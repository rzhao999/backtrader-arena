"""
``play_round``: orchestrate a single round (generator phase, code execution,
solver phase, scoring, DB persistence).

``run_match``: drive the full multi-round match including initial config +
per-round logging + final summary with token totals and per-difficulty
solver-accuracy breakdown.

The end-of-match summary helpers (``_usage_totals``, ``_print_usage_totals``,
``_difficulty_breakdown``, ``_print_difficulty_breakdown``) live here too
since they're only consumed by ``run_match``.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import arena_core as _core  # first: defines shared arena paths/defaults
from arena_core import (
    BACKTRADER_AGENT_TEMPLATE,
    DEFAULT_GENERATOR_MEMORY,
    DEFAULT_GENERATOR_SOURCE,
    FORMAT_TEMPLATE,
    GENERATOR_SOURCES,
    LOGS_DIR,
    MEMORY_MODES,
    RoundResult,
    VENV_PY,
    _BLUE,
    _BOLD,
    _CYAN,
    _DIM,
    _GREEN,
    _MAGENTA,
    _RED,
    _YELLOW,
    _c,
    _fmt_usage,
    _logical_text,
    SOLVER_WORK_PROTOCOL,
    banner,
    build_solver_prompt,
    choose_solver_answer_with_options,
    extract_block,
    parse_code_answer,
    render_history,
    render_solver_history,
    run_verification_code,
)

import db  # noqa: E402
from agent_backends import AgentDef, aggregate_usage  # noqa: E402
import backtrader_source as bt_source  # noqa: E402
from backtrader_source import BacktraderGenConfig, PreloadedQuestion  # noqa: E402
from arena_invoke import (
    AgentInvocation,
    _invoke_agent,
    _prompt_arena_config,
    _resolve_and_check_agents,
)


# ---------------------------------------------------------------------------
# LLM-authored backtrader_agent generator
# ---------------------------------------------------------------------------

_SYMBOL_ROTATION = (
    "AAPL", "MSFT", "NVDA", "TSLA", "SPY",
    "QQQ", "AMZN", "GOOG", "JPM", "XOM",
)


def _serialize_baseline_cfg(cfg: BacktraderGenConfig) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in asdict(cfg).items())


def _smart_round_baseline(
    baseline: BacktraderGenConfig,
    *,
    round_num: int,
    history: list[RoundResult],
) -> BacktraderGenConfig:
    """Vary safe defaults so omitted LLM fields do not repeat round 1."""
    values = asdict(baseline)
    offset = max(round_num - 1, 0)

    if baseline.seed is not None:
        values["seed"] = int(baseline.seed) + offset * 9973

    if baseline.symbol in _SYMBOL_ROTATION:
        start = _SYMBOL_ROTATION.index(baseline.symbol)
        values["symbol"] = _SYMBOL_ROTATION[(start + offset) % len(_SYMBOL_ROTATION)]

    strategies = tuple(bt_source.REGISTERED_BACKTRADER_STRATEGIES)
    if baseline.strategy in strategies:
        start = strategies.index(baseline.strategy)
        values["strategy"] = strategies[(start + offset) % len(strategies)]

    try:
        return BacktraderGenConfig(**values)
    except Exception:
        return baseline


def build_backtrader_agent_prompt(
    history: list[RoundResult],
    baseline_cfg: BacktraderGenConfig,
    *,
    memory: str = DEFAULT_GENERATOR_MEMORY,
    topic_hint: Optional[str] = None,
    base_pool_path: Optional[Path] = None,
) -> str:
    """Ask the LLM generator to author this round's Backtrader MCQ."""
    effective_history = history if memory == "history" else []
    base_pool_path = base_pool_path or (
        bt_source.BACKTRADER_DIR_DEFAULT / bt_source.BACKTRADER_BASE_POOL_FILENAME
    )
    template = BACKTRADER_AGENT_TEMPLATE.read_text()
    prompt = (
        template
        .replace("{BASELINE_CFG}", _serialize_baseline_cfg(baseline_cfg))
        .replace("{BASE_POOL_PATH}", str(base_pool_path))
        .replace("{ARENA_ROOT}", str(_core.ROOT))
        .replace("{BACKTRADER_DIR}", str(base_pool_path.parent))
        .replace(
            "{DIFFICULTY_CHOICES}",
            ", ".join(bt_source.DIFFICULTY_CHOICES),
        )
        .replace("{HISTORY}", render_history(effective_history))
    )
    if topic_hint:
        prompt += (
            "\n\nAdditional designer note for this round (non-binding "
            f"hint):\n  {topic_hint}\n"
        )
    return prompt


def _parse_declared_answer(answer_block: str) -> str:
    answer = (answer_block or "").strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(
            "ANSWER block must contain exactly one of A, B, C, or D."
        )
    return answer


def parse_backtrader_agent_mcq(raw_text: str) -> tuple[PreloadedQuestion, Optional[str]]:
    """Parse an LLM-authored Backtrader MCQ into the round input object."""
    question_block = (extract_block(raw_text, "QUESTION") or "").strip()
    answer_block = extract_block(raw_text, "ANSWER") or ""
    code_block = (extract_block(raw_text, "CODE") or "").strip()
    rationale = extract_block(raw_text, "RATIONALE")

    if not question_block:
        raise ValueError("missing QUESTION block")
    if not code_block:
        raise ValueError("missing CODE block")

    declared = _parse_declared_answer(answer_block)
    question, options = bt_source.parse_question(question_block)
    missing_options = [k for k in ("A", "B", "C", "D") if not options.get(k)]
    if missing_options:
        raise ValueError(f"missing option text for: {', '.join(missing_options)}")

    return (
        PreloadedQuestion(
            question=question,
            options=options,
            answer=declared,
            raw_prompt=question_block,
            code=code_block,
            source="backtrader_agent_llm",
            difficulty="llm",
            qtype="llm_authored",
            metadata={
                "rationale": (rationale or "").strip(),
            },
        ),
        rationale,
    )


def run_backtrader_agent_generator(
    *,
    round_num: int,
    history: list[RoundResult],
    generator_agent: AgentDef,
    generator_model: Optional[str],
    generator_extra: list[str],
    generator_memory: str,
    baseline_cfg: BacktraderGenConfig,
    backtrader_dir: Optional[Path],
    backtrader_python: Optional[str],
    round_dir: Path,
    topic_hint: Optional[str],
) -> tuple[Optional[PreloadedQuestion], AgentInvocation, Optional[str], str]:
    """Invoke the LLM generator and parse its authored Backtrader MCQ."""
    round_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = round_dir / "generator"
    gen_dir.mkdir(parents=True, exist_ok=True)

    banner(
        f"ROUND {round_num} -- GENERATOR "
        f"({generator_agent.name}+backtrader"
        f"{f' / {generator_model}' if generator_model else ''}, "
        f"memory={generator_memory})",
        _MAGENTA,
    )

    round_baseline_cfg = _smart_round_baseline(
        baseline_cfg,
        round_num=round_num,
        history=history,
    )
    base_pool_path = (
        (backtrader_dir or bt_source.BACKTRADER_DIR_DEFAULT)
        / bt_source.BACKTRADER_BASE_POOL_FILENAME
    )

    prompt = build_backtrader_agent_prompt(
        history, round_baseline_cfg,
        memory=generator_memory,
        topic_hint=topic_hint,
        base_pool_path=base_pool_path,
    )
    (gen_dir / "prompt.txt").write_text(prompt)

    invo = _invoke_agent(
        generator_agent, prompt,
        cwd=gen_dir,
        timeout=_core.AGENT_TIMEOUT_S,
        role="generator",
        model=generator_model,
        user_extra=generator_extra,
    )
    (gen_dir / "stdout.txt").write_text(invo.stdout)
    (gen_dir / "stderr.txt").write_text(invo.stderr)
    if invo.returncode != 0:
        print(_c(
            f"[generator exited with code {invo.returncode}]", _YELLOW,
        ))
        if invo.stderr.strip():
            print(_c(invo.stderr.strip()[:2000], _DIM))
    if invo.usage:
        print(_c(f"Generator usage: {_fmt_usage(invo.usage)}", _DIM))

    text = _logical_text(invo.stdout)
    try:
        preloaded, rationale = parse_backtrader_agent_mcq(text)
    except ValueError as exc:
        print(_c(
            f"[backtrader_agent] failed to parse authored MCQ: {exc}",
            _YELLOW,
        ))
        return None, invo, None, "llm"
    bt_dir = (backtrader_dir or bt_source.BACKTRADER_DIR_DEFAULT).resolve()
    preloaded.metadata.update({
        "base_pool_path": str(base_pool_path),
        "backtrader_dir": str(bt_dir),
    })

    print(_c(
        "[backtrader_agent] authored MCQ parsed; verifier will be run "
        "before solver phase.",
        _DIM,
    ))
    if rationale:
        first_line = rationale.strip().splitlines()[0][:200]
        print(_c(f"[backtrader_agent] rationale: {first_line}", _DIM))

    (gen_dir / "driver_config.json").write_text(
        json.dumps({
            "round_baseline_cfg": asdict(round_baseline_cfg),
            "base_pool_path": str(base_pool_path),
            "declared_answer": preloaded.answer,
            "source": preloaded.source,
            "difficulty": preloaded.difficulty,
            "qtype": preloaded.qtype,
            "rationale": rationale,
        }, ensure_ascii=False, indent=2),
    )
    return preloaded, invo, rationale, "llm"


def _build_forfeit_result(
    round_num: int,
    reason: str,
    *,
    driver_invocation: Optional[AgentInvocation],
    generator_agent: Optional[AgentDef],
    solver_agent: AgentDef,
    generator_model: Optional[str],
    solver_model: Optional[str],
    difficulty: Optional[str] = None,
) -> RoundResult:
    """Build a synthesized generator-forfeit result when no MCQ exists."""
    print(_c(f"GENERATOR FORFEIT: {reason}", _RED))
    gen_usage = driver_invocation.usage if driver_invocation else None
    gen_model = None
    if gen_usage:
        gen_model = gen_usage.get("model")
    return RoundResult(
        round=round_num,
        question="",
        options={"A": "", "B": "", "C": "", "D": ""},
        declared_answer=None,
        ground_truth=None,
        solver_answer=None,
        solver_correct=None,
        generator_forfeit=True,
        gen_score_delta=-1,
        solver_score_delta=+1,
        code="",
        code_stdout="",
        code_stderr="",
        generator_raw=(driver_invocation.stdout if driver_invocation else ""),
        solver_raw="",
        generator_agent=generator_agent.name if generator_agent else "",
        solver_agent=solver_agent.name,
        generator_model=gen_model or generator_model,
        solver_model=solver_model,
        generator_usage=gen_usage,
        solver_usage=None,
        elapsed_sec=0.0,
        notes="forfeit: " + reason,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------


def play_round(
    round_num: int,
    history: list[RoundResult],
    *,
    generator_agent: Optional[AgentDef],
    solver_agent: AgentDef,
    generator_model: Optional[str],
    solver_model: Optional[str],
    generator_extra: list[str],
    solver_extra: list[str],
    generator_memory: str,
    solver_memory: str,
    match_dir: Path,
    topic_hint: Optional[str],
    match_log_name: str,
    db_index: dict,
    preloaded: Optional[PreloadedQuestion] = None,
    driver_invocation: Optional[AgentInvocation] = None,
) -> RoundResult:
    """Play one round.

    ``preloaded`` is required: supported generator sources provide a ready
    MCQ either from the deterministic Backtrader pipeline or from the
    LLM-driven ``backtrader_agent`` authoring path.

    ``driver_invocation`` is the ``AgentInvocation`` returned by the
    ``backtrader_agent`` LLM call (authoring this round's MCQ). When
    supplied it replaces the synthesized
    generator invocation in the preloaded branch, so token usage is
    tracked correctly.
    """
    round_dir = match_dir / f"round_{round_num:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    # Each role gets exactly ONE per-round subdir which is BOTH:
    #   - the agent's cwd (so any code/scratch the agent writes lands here)
    #   - the artifact dir (prompt.txt, stdout.txt, stderr.txt, role-specific extras)
    # The OS sandbox (auto-enabled in sandbox memory mode) blocks reads
    # of every other dir under the arena, so the role can only see its
    # own subdir + system paths needed for the binary to execute.
    gen_dir = round_dir / "generator"
    sol_dir = round_dir / "solver"
    gen_dir.mkdir(parents=True, exist_ok=True)
    sol_dir.mkdir(parents=True, exist_ok=True)
    round_start = time.time()

    notes_parts: list[str] = []
    gen_invo: AgentInvocation

    if preloaded is not None:
        # ---- Preloaded question (e.g. Backtrader pipeline) ----
        src = preloaded.source
        diff = preloaded.difficulty or "?"
        # Skip the banner if the caller (backtrader_agent path)
        # already printed one for this round.
        if driver_invocation is None:
            banner(
                f"ROUND {round_num} -- GENERATOR "
                f"(source={src}, difficulty={diff})",
                _MAGENTA,
            )
        notes_parts.append(f"source={src}")
        if preloaded.difficulty:
            notes_parts.append(f"difficulty={preloaded.difficulty}")
        if preloaded.qtype:
            notes_parts.append(f"qtype={preloaded.qtype}")

        question: Optional[str] = preloaded.question
        options = dict(preloaded.options)
        code: Optional[str] = preloaded.code
        declared = preloaded.answer
        opt_a, opt_b, opt_c, opt_d = [
            options.get(k, "") for k in ("A", "B", "C", "D")
        ]

        # For the pure-backtrader path we synthesize a fake
        # AgentInvocation so the downstream DB / logging paths keep a
        # uniform shape (stdout=raw prompt, usage=None -- deterministic
        # pipeline, no LLM call). For the backtrader_agent path the caller
        # supplies the real LLM invocation (question author), so we track
        # its token usage honestly.
        if driver_invocation is not None:
            gen_invo = driver_invocation
        else:
            gen_invo = AgentInvocation(
                stdout=preloaded.raw_prompt,
                stderr="",
                returncode=0,
                usage=None,
                cmd_ok=True,
            )
        missing: list[str] = []

        # Actually run the tiny ``print("ANSWER: X")`` stub -- takes ~10ms
        # and keeps the well-defined check honest (real stdout, real rc).
        print(_c(
            "Preloaded question. Running pre-verified ANSWER stub...", _DIM,
        ))
        with tempfile.TemporaryDirectory(prefix="arena_verify_") as tmp:
            verifier_python = None
            verifier_paths: list[Path] | None = None
            if preloaded.source == "backtrader_agent_llm":
                bt_dir_raw = preloaded.metadata.get("backtrader_dir")
                bt_dir = Path(bt_dir_raw) if bt_dir_raw else bt_source.BACKTRADER_DIR_DEFAULT
                verifier_python = bt_source.discover_backtrader_python()
                verifier_paths = [bt_dir]
            code_stdout, code_stderr, code_rc = run_verification_code(
                code,
                round_num,
                work_dir=Path(tmp),
                python_path=verifier_python,
                pythonpath=verifier_paths,
            )
        code_timed_out = (code_rc == 124 and "timed out" in code_stderr)
        ground_truth: Optional[str] = parse_code_answer(code_stdout)
    else:
        raise RuntimeError(
            "play_round requires a Backtrader preloaded question; "
            "supported generator sources are backtrader and backtrader_agent"
        )

    have_min_blocks = bool(question) and all(
        options[k] for k in ("A", "B", "C", "D")
    )

    well_defined, ill_reasons = db.evaluate_well_defined(
        question=question,
        options=options,
        code=code,
        code_stdout=code_stdout,
        code_stderr=code_stderr,
        code_returncode=code_rc,
        code_timed_out=code_timed_out,
    )

    forfeit = ground_truth is None
    if forfeit:
        if ill_reasons:
            notes_parts.append("forfeit: " + "; ".join(ill_reasons))
        print(_c(
            f"GENERATOR FORFEIT: {ill_reasons or ['no parseable ground truth']}",
            _RED,
        ))
        if code_stderr.strip():
            print(_c(textwrap.indent(code_stderr.strip()[:1500], "  "), _DIM))
    else:
        if declared and declared != ground_truth:
            notes_parts.append(
                f"declared answer {declared!r} != code-derived "
                f"{ground_truth!r} (using code as ground truth)"
            )
        print(_c(f"Ground truth (from code): {ground_truth}", _GREEN))
        print(_c(f"Well-defined: {well_defined}", _DIM))

    if question:
        print(_c("Question preview:", _DIM))
        preview = question.strip().splitlines()
        for line in preview[:6]:
            print("  " + line)
        if len(preview) > 6:
            print(_c(f"  ... ({len(preview)-6} more lines)", _DIM))

    # ---- Solver phase (only if we have something fair to ask) ----
    solver_stdout = ""
    solver_stderr = ""
    solver_answer: Optional[str] = None
    solver_parse_method: str = "none"
    correct: Optional[bool] = None
    solver_invo: Optional[AgentInvocation] = None
    solver_saved_solution = False

    if forfeit:
        gen_delta, solver_delta = -1, +1
    else:
        banner(
            f"ROUND {round_num} -- SOLVER "
            f"({solver_agent.name}"
            f"{f' / {solver_model}' if solver_model else ''}, "
            f"memory={solver_memory})",
            _BLUE,
        )
        if preloaded is not None and preloaded.raw_prompt:
            # Preloaded prompts (e.g. backtrader) already include their
            # own preamble + options + "<<< X >>>" instruction, tailored
            # to that source. Re-wrapping them through ``format.txt``
            # (which hardcodes "Use SciPy...") would be misleading.
            solver_prompt = preloaded.raw_prompt
            if solver_memory == "history":
                past = render_solver_history(history)
                solver_prompt = (
                    "You are continuing a multi-round adversarial MCQ "
                    "match. Summary of your previous attempts (oldest "
                    "first):\n\n"
                    f"{past}\n\n"
                    "---- Current round ----\n\n"
                ) + solver_prompt
            # Force every solver round (including backtrader) to actually
            # write + run code rather than answer from memory.
            solver_prompt = SOLVER_WORK_PROTOCOL + solver_prompt
        else:
            solver_prompt = build_solver_prompt(
                question or "", options,
                history=history, memory=solver_memory,
            )
        (sol_dir / "prompt.txt").write_text(solver_prompt)
        solver_invo = _invoke_agent(
            solver_agent, solver_prompt,
            cwd=sol_dir,
            timeout=_core.AGENT_TIMEOUT_S,
            role="solver",
            model=solver_model,
            user_extra=solver_extra,
        )
        solver_stdout = solver_invo.stdout
        solver_stderr = solver_invo.stderr
        (sol_dir / "stdout.txt").write_text(solver_stdout)
        (sol_dir / "stderr.txt").write_text(solver_stderr)
        solver_saved_solution = (sol_dir / "solution.py").exists()
        if not solver_saved_solution:
            notes_parts.append(
                "solver protocol violation: ./solution.py was not saved"
            )
            print(_c(
                "SOLVER PROTOCOL VIOLATION: ./solution.py was not saved",
                _YELLOW,
            ))

        if solver_invo.returncode != 0:
            print(_c(
                f"[solver exited with code {solver_invo.returncode}]", _YELLOW,
            ))
            if solver_stderr.strip():
                print(_c(solver_stderr.strip()[:2000], _DIM))
        if solver_invo.usage:
            print(_c(f"Solver usage: {_fmt_usage(solver_invo.usage)}", _DIM))

        # <<< X >>> may live either in raw stdout or inside the JSON
        # wrapper; check the logical text too. ``choose_solver_answer``
        # returns ``(letter_or_None, method)`` where method is "strict"
        # (literal ``<<< X >>>``), "heuristic" (agent_backends hint
        # patterns like "answer is C"), or "none".
        solver_answer, solver_parse_method = choose_solver_answer_with_options(
            solver_stdout,
            options=options,
        )
        if solver_answer is None:
            # Try the logical (JSON-unwrapped) text before giving up.
            solver_answer, solver_parse_method = choose_solver_answer_with_options(
                _logical_text(solver_stdout),
                options=options,
            )

        if solver_answer is None:
            notes_parts.append(
                "solver did not emit a parseable <<< X >>> answer"
            )
            correct = False
        elif not solver_saved_solution:
            correct = False
        else:
            correct = (solver_answer == ground_truth)
            if solver_parse_method == "heuristic":
                # Surface the fallback so we can tell at-a-glance which
                # solver answers were recovered loosely vs by the
                # canonical wrapper.
                notes_parts.append(
                    "solver answer recovered via heuristic fallback "
                    "(no <<< X >>> wrapper)"
                )

        method_tag = (
            "" if solver_parse_method == "strict"
            else f", parse={solver_parse_method}"
        )
        if correct:
            gen_delta, solver_delta = -1, +1
            print(_c(
                f"SOLVER CORRECT  (answered {solver_answer}, "
                f"truth {ground_truth}{method_tag})",
                _GREEN,
            ))
        else:
            gen_delta, solver_delta = +1, -1
            print(_c(
                f"SOLVER WRONG    (answered {solver_answer}, "
                f"truth {ground_truth}{method_tag})",
                _RED,
            ))

    elapsed = round(time.time() - round_start, 3)

    if preloaded is not None:
        # Persist answer-bearing generator artifacts only after the solver
        # has answered, so an unsandboxed solver cannot read sibling files.
        (gen_dir / "generator_preloaded.json").write_text(
            json.dumps({
                "source": src,
                "difficulty": preloaded.difficulty,
                "qtype": preloaded.qtype,
                "answer": preloaded.answer,
                "raw_prompt": preloaded.raw_prompt,
                "metadata": preloaded.metadata,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- Persist to the question + attempt database ----
    if have_min_blocks:
        try:
            qid, was_new = db.record_question(
                question=question or "",
                options=options,
                code=code or "",
                declared_answer=declared,
                ground_truth=ground_truth,
                code_stdout=code_stdout,
                code_stderr=code_stderr,
                code_returncode=code_rc,
                code_timed_out=code_timed_out,
                well_defined=well_defined,
                well_defined_reasons=ill_reasons,
                match_log=match_log_name,
                round_num=round_num,
                generator_model=(
                    (gen_invo.usage or {}).get("model")
                    or (preloaded.source if preloaded else None)
                    or generator_model
                    or (generator_agent.name if generator_agent else "unknown")
                ),
                topic_hint=topic_hint,
                existing_index=db_index,
            )
            tag = "new" if was_new else "duplicate of existing"
            print(_c(
                f"DB: question {qid[:12]} ({tag}); well_defined={well_defined}",
                _DIM,
            ))
            db.record_attempt(
                question_id=qid,
                solver_model=(
                    (solver_invo.usage or {}).get("model") if solver_invo else None
                ) or solver_model or solver_agent.name,
                solver_answer=solver_answer,
                ground_truth=ground_truth,
                correct=correct,
                generator_forfeit=forfeit,
                match_log=match_log_name,
                round_num=round_num,
                solver_raw_excerpt=solver_stdout,
                solver_usage=(solver_invo.usage if solver_invo else None),
                solver_parse_method=solver_parse_method,
            )
        except Exception as e:  # pragma: no cover -- defensive
            print(_c(f"[db] failed to persist round: {e!r}", _YELLOW))
    else:
        print(_c(
            "DB: skipped (generator did not produce question + 4 options)",
            _DIM,
        ))

    return RoundResult(
        round=round_num,
        question=question or "",
        options=options,
        declared_answer=declared,
        ground_truth=ground_truth,
        solver_answer=solver_answer,
        solver_correct=correct,
        generator_forfeit=forfeit,
        gen_score_delta=gen_delta,
        solver_score_delta=solver_delta,
        code=code or "",
        code_stdout=code_stdout,
        code_stderr=code_stderr,
        generator_raw=gen_invo.stdout,
        solver_raw=solver_stdout,
        generator_agent=(
            preloaded.source if preloaded
            else (generator_agent.name if generator_agent else "unknown")
        ),
        solver_agent=solver_agent.name,
        generator_model=(
            (gen_invo.usage or {}).get("model")
            or (preloaded.source if preloaded else None)
            or generator_model
        ),
        solver_model=(
            (solver_invo.usage or {}).get("model") if solver_invo else None
        ) or solver_model,
        generator_usage=gen_invo.usage,
        solver_usage=(solver_invo.usage if solver_invo else None),
        elapsed_sec=elapsed,
        notes="; ".join(notes_parts),
        difficulty=(preloaded.difficulty if preloaded else None),
        solver_parse_method=solver_parse_method,
    )


# ---------------------------------------------------------------------------
# End-of-match summary helpers (token usage + per-difficulty accuracy)
# ---------------------------------------------------------------------------


def _usage_totals(records: list[RoundResult], key: str) -> Optional[dict]:
    """Aggregate per-round usage records that live on a given ``key``."""
    fake = [{"usage": getattr(r, key)} for r in records if getattr(r, key)]
    if not fake:
        return None
    return aggregate_usage(fake)


def _print_usage_totals(label: str, totals: Optional[dict], color: str) -> None:
    if not totals:
        print(_c(f"  {label}: (no token usage reported)", _DIM))
        return
    t = totals["totals"]
    cost = t.get("total_cost_usd") or 0.0
    models = ", ".join(totals["models"]) or "?"
    print(_c(
        f"  {label}: in={int(t['input_tokens'])} out={int(t['output_tokens'])} "
        f"cache_r={int(t['cache_read_tokens'])} "
        f"cache_c={int(t['cache_creation_tokens'])}  "
        f"cost=${cost:.4f}  model={models}",
        color,
    ))


def _difficulty_breakdown(
    history: list[RoundResult],
) -> dict[str, dict[str, int]]:
    """Group rounds by ``difficulty`` and tally outcomes per bucket.

    Returns a mapping ``difficulty -> {rounds, solver_correct,
    solver_wrong, forfeits}``. Rounds without a difficulty are bucketed
    under ``"(unspecified)"``.
    """
    out: dict[str, dict[str, int]] = {}
    for r in history:
        key = r.difficulty or "(unspecified)"
        bucket = out.setdefault(
            key,
            {"rounds": 0, "solver_correct": 0,
             "solver_wrong": 0, "forfeits": 0},
        )
        bucket["rounds"] += 1
        if r.generator_forfeit:
            bucket["forfeits"] += 1
        elif r.solver_correct:
            bucket["solver_correct"] += 1
        else:
            bucket["solver_wrong"] += 1
    return out


def _ordered_difficulty_keys(breakdown: dict[str, dict[str, int]]) -> list[str]:
    """Stable display order: backtrader's canonical levels first
    (easy / medium / hard), then any other tags alphabetically, with
    ``"(unspecified)"`` last.
    """
    preferred = list(bt_source.DIFFICULTY_LEVELS)
    keys = [d for d in preferred if d in breakdown]
    extras = sorted(
        d for d in breakdown
        if d not in preferred and d != "(unspecified)"
    )
    keys.extend(extras)
    if "(unspecified)" in breakdown:
        keys.append("(unspecified)")
    return keys


def _print_difficulty_breakdown(history: list[RoundResult]) -> None:
    """Print solver accuracy grouped by per-round difficulty.

    Suppressed entirely when no round carries a difficulty tag so the
    summary stays uncluttered.
    """
    breakdown = _difficulty_breakdown(history)
    if not breakdown:
        return
    if set(breakdown) == {"(unspecified)"}:
        return
    print(_c("Per-difficulty solver accuracy:", _BOLD))
    for k in _ordered_difficulty_keys(breakdown):
        b = breakdown[k]
        scorable = b["rounds"] - b["forfeits"]
        if scorable > 0:
            acc = b["solver_correct"] / scorable
            acc_str = (
                f"{acc * 100:5.1f}%  "
                f"({b['solver_correct']}/{scorable})"
            )
        else:
            acc_str = "  n/a  (no scorable rounds)"
        forfeit_tag = (
            f"  forfeits={b['forfeits']}" if b["forfeits"] else ""
        )
        print(_c(
            f"  {k:<14} rounds={b['rounds']:<3}  "
            f"solver acc={acc_str}{forfeit_tag}",
            _BLUE,
        ))
    total_rounds = sum(b["rounds"] for b in breakdown.values())
    total_forfeits = sum(b["forfeits"] for b in breakdown.values())
    total_scorable = total_rounds - total_forfeits
    total_correct = sum(b["solver_correct"] for b in breakdown.values())
    if total_scorable > 0:
        overall_acc = total_correct / total_scorable
        overall_str = f"{overall_acc * 100:5.1f}%  ({total_correct}/{total_scorable})"
    else:
        overall_str = "  n/a  (no scorable rounds)"
    forfeit_tag = f"  forfeits={total_forfeits}" if total_forfeits else ""
    print(_c(f"  {'OVERALL':<14} rounds={total_rounds:<3}  solver acc={overall_str}{forfeit_tag}", _BLUE))


def _parse_method_breakdown(history: list[RoundResult]) -> dict[str, int]:
    """Count rounds by ``solver_parse_method`` (skipping forfeits).

    Returns an ordered dict keyed by ``"strict" / "heuristic" / "none"``
    with the count of (non-forfeit) rounds in each bucket.
    """
    counts = {"strict": 0, "heuristic": 0, "numeric_closest": 0, "none": 0}
    for r in history:
        if r.generator_forfeit:
            continue
        method = (r.solver_parse_method or "none")
        counts[method] = counts.get(method, 0) + 1
    return counts


def _print_parse_method_breakdown(history: list[RoundResult]) -> None:
    """Print how many solver answers were strict / heuristic / unparsed.

    Suppressed when there are no scorable rounds so the summary stays
    uncluttered for forfeit-only matches.
    """
    counts = _parse_method_breakdown(history)
    total = sum(counts.values())
    if total == 0:
        return
    print(_c("Solver answer parse method:", _BOLD))
    label_color = {
        "strict":          _GREEN,
        "heuristic":       _YELLOW,
        "numeric_closest": _CYAN,
        "none":            _RED,
    }
    for k in ("strict", "heuristic", "numeric_closest", "none"):
        n = counts.get(k, 0)
        pct = (n / total) * 100 if total else 0.0
        print(_c(
            f"  {k:<10} rounds={n:<3}  ({pct:5.1f}%  of {total} scorable)",
            label_color[k],
        ))


# ---------------------------------------------------------------------------
# Match driver
# ---------------------------------------------------------------------------


def run_match(
    rounds: int,
    *,
    generator_model: Optional[str],
    solver_model: Optional[str],
    generator_agent_name: str,
    solver_agent_name: str,
    generator_extra: list[str],
    solver_extra: list[str],
    generator_memory: str,
    solver_memory: str,
    topic_hint: Optional[str],
    generator_source: str = DEFAULT_GENERATOR_SOURCE,
    backtrader_difficulty: Optional[str] = None,
    backtrader_dir: Optional[Path] = None,
    backtrader_python: Optional[str] = None,
    backtrader_cfg: Optional[BacktraderGenConfig] = None,
    backtrader_interactive: bool = True,
    backtrader_full_interactive: bool = False,
) -> None:
    if generator_memory not in MEMORY_MODES:
        raise SystemExit(
            f"--generator-memory must be one of {MEMORY_MODES}, "
            f"got {generator_memory!r}"
        )
    if solver_memory not in MEMORY_MODES:
        raise SystemExit(
            f"--solver-memory must be one of {MEMORY_MODES}, "
            f"got {solver_memory!r}"
        )
    if generator_source not in GENERATOR_SOURCES:
        raise SystemExit(
            f"--generator-source must be one of {GENERATOR_SOURCES}, "
            f"got {generator_source!r}"
        )
    if not VENV_PY.exists():
        print(_c(
            f"ERROR: venv python not found at {VENV_PY}.\n"
            "Run:  python3 -m venv .venv && "
            ".venv/bin/pip install -r requirements.txt",
            _RED,
        ))
        sys.exit(1)
    if not FORMAT_TEMPLATE.exists():
        print(_c(f"ERROR: missing {FORMAT_TEMPLATE}", _RED))
        sys.exit(1)
    backtrader_cfg = backtrader_cfg or BacktraderGenConfig()

    generator_agent: Optional[AgentDef] = None
    solver_agent: Optional[AgentDef] = None
    if not backtrader_full_interactive:
        generator_agent, solver_agent, _ = _resolve_and_check_agents(
            generator_source=generator_source,
            generator_agent_name=generator_agent_name,
            solver_agent_name=solver_agent_name,
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"match_{ts}.jsonl"
    match_dir = LOGS_DIR / f"match_{ts}"
    match_dir.mkdir(parents=True, exist_ok=True)

    # Memory="sandbox" means no history in the prompt and, where the
    # selected backend supports it, an OS sandbox restricted to the role's
    # own per-round directory. The local wrappers honor these flags via
    # agent_backends.maybe_wrap_macos_sandbox_exec where available.
    _core.GENERATOR_OS_SANDBOX = (generator_memory == "sandbox")
    _core.SOLVER_OS_SANDBOX = (solver_memory == "sandbox")

    # ---- Arena-level interactive config (any generator source) ----
    # ``-I`` opens the full interactive editor. Stage 1 (here) covers the
    # arena-level knobs: round count, model, and per-role agent + memory.
    # Stage 2 (further down, only if source=backtrader) covers the
    # backtrader-specific knobs (symbol / dates / ... / difficulty).
    if backtrader_full_interactive:
        (new_rounds, generator_agent_name, solver_agent_name,
         generator_model, solver_model,
         generator_memory, solver_memory,
         generator_os_sandbox, solver_os_sandbox) = _prompt_arena_config(
            rounds=rounds,
            generator_agent_name=generator_agent_name,
            solver_agent_name=solver_agent_name,
            generator_model=generator_model,
            solver_model=solver_model,
            generator_memory=generator_memory,
            solver_memory=solver_memory,
            generator_source=generator_source,
            generator_os_sandbox=getattr(_core, "GENERATOR_OS_SANDBOX", False),
            solver_os_sandbox=getattr(_core, "SOLVER_OS_SANDBOX", False),
        )
        if new_rounds != rounds:
            print(_c(
                f"[arena] rounds overridden interactively: "
                f"{rounds} -> {new_rounds}",
                _DIM,
            ))
            rounds = new_rounds
        # Re-resolve agents in case the user picked a different backend.
        generator_agent, solver_agent, _ = _resolve_and_check_agents(
            generator_source=generator_source,
            generator_agent_name=generator_agent_name,
            solver_agent_name=solver_agent_name,
        )
        # In -I mode, choosing memory="sandbox" implies OS sandboxing;
        # preserve any explicit sandbox choice returned by the editor.
        _core.GENERATOR_OS_SANDBOX = (
            (generator_memory == "sandbox") or bool(generator_os_sandbox)
        )
        _core.SOLVER_OS_SANDBOX = (
            (solver_memory == "sandbox") or bool(solver_os_sandbox)
        )
    assert solver_agent is not None

    # ---- Backtrader-specific setup (baseline cfg + optional queue) -------
    # For ``backtrader``: we pre-generate the full queue up front.
    # For ``backtrader_agent``: we only collect reference Backtrader context
    # for the LLM-authored prompt; the MCQ itself is written per round.
    preloaded_queue: list[PreloadedQuestion] = []
    chosen_difficulty: Optional[str] = None
    if generator_source in ("backtrader", "backtrader_agent"):
        if backtrader_full_interactive:
            # Stage 2: backtrader knobs. CLI flags (and arena-side defaults
            # baked into ``backtrader_cfg``) seed each field. For
            # backtrader_agent these values are reference context for
            # the LLM-authored MCQ prompt.
            backtrader_cfg, chosen_difficulty = bt_source.prompt_full_config(
                defaults=backtrader_cfg,
                preselect_difficulty=backtrader_difficulty,
                allow_smart_defaults=(generator_source == "backtrader_agent"),
            )
        elif generator_source == "backtrader":
            chosen_difficulty = bt_source.prompt_difficulty(
                preselect=backtrader_difficulty,
                interactive=backtrader_interactive,
            )
        else:
            # backtrader_agent without -I: no Backtrader difficulty prompt.
            chosen_difficulty = backtrader_difficulty or "easy"

    if generator_source == "backtrader":
        print(_c(
            f"\n[backtrader] pre-generating {rounds} question(s) "
            f"at difficulty={chosen_difficulty!r} ...",
            _DIM,
        ))
        try:
            preloaded_queue = bt_source.generate_questions(
                difficulty=chosen_difficulty,
                num_questions=rounds,
                backtrader_dir=backtrader_dir,
                python_interpreter=backtrader_python,
                cfg=backtrader_cfg,
            )
        except RuntimeError as exc:
            print(_c(f"ERROR: backtrader generation failed: {exc}", _RED))
            sys.exit(1)
        print(_c(
            f"[backtrader] pre-generated {len(preloaded_queue)} question(s). "
            f"Difficulty breakdown: "
            f"{dict((d, sum(1 for q in preloaded_queue if q.difficulty == d)) for d in bt_source.DIFFICULTY_LEVELS)}",
            _DIM,
        ))
        # Generator-memory is moot when questions are pre-generated; warn
        # so the user understands the flag is being ignored.
        if generator_memory == "history":
            print(_c(
                "[backtrader] note: --generator-memory=history has no "
                "effect with a non-agent generator source (questions are "
                "drawn from a fixed queue).",
                _DIM,
            ))
    elif generator_source == "backtrader_agent":
        print(_c(
            f"\n[backtrader_agent] LLM generator will author a Backtrader "
            f"MCQ and verifier per round (reference baseline: symbol={backtrader_cfg.symbol}, "
            f"dates={backtrader_cfg.start_date}..{backtrader_cfg.end_date}, "
            f"strategy={backtrader_cfg.strategy}).",
            _DIM,
        ))

    symmetry = (
        "symmetric" if generator_memory == solver_memory else "asymmetric"
    )
    if generator_source == "backtrader":
        gen_label = f"backtrader:{chosen_difficulty}"
    elif generator_source == "backtrader_agent":
        gen_label = (
            f"{generator_agent.name}+backtrader ({generator_memory})"  # type: ignore[union-attr]
        )
    model_suffix = ""
    if generator_model and solver_model and generator_model == solver_model:
        model_suffix = f"  model={generator_model}"
    else:
        parts = []
        if generator_model:
            parts.append(f"gen_model={generator_model}")
        if solver_model:
            parts.append(f"sol_model={solver_model}")
        if parts:
            model_suffix = "  " + "  ".join(parts)
    banner(
        f"ADVERSARIAL MCQ MATCH  rounds={rounds}  "
        f"generator={gen_label}  "
        f"solver={solver_agent.name} ({solver_memory})  [{symmetry}]"
        + model_suffix,
        _CYAN,
    )
    print(_c(f"Logging to {log_path}", _DIM))
    print(_c(f"Round artifacts in {match_dir}", _DIM))
    db_index = db.load_question_index()
    print(_c(
        f"Question DB: {len(db_index)} unique questions on file at "
        f"{db.QUESTIONS_PATH}",
        _DIM,
    ))

    history: list[RoundResult] = []
    gen_score = 0
    solver_score = 0

    with log_path.open("w") as logf:
        meta = {
            "type": "match_meta",
            "timestamp": ts,
            "generator_source": generator_source,
            "generator_agent": (
                generator_agent.name if generator_agent else None
            ),
            "backtrader_difficulty": chosen_difficulty,
            "solver_agent": solver_agent.name,
            "generator_memory": generator_memory,
            "solver_memory": solver_memory,
            "symmetry": symmetry,
            "generator_model": generator_model,
            "solver_model": solver_model,
            "rounds": rounds,
            "topic_hint": topic_hint,
            "agent_timeout_s": _core.AGENT_TIMEOUT_S,
            "code_timeout_s": _core.CODE_TIMEOUT_S,
        }
        logf.write(json.dumps(meta) + "\n")
        logf.flush()

        for r in range(1, rounds + 1):
            try:
                preloaded_for_round: Optional[PreloadedQuestion] = None
                driver_invo: Optional[AgentInvocation] = None

                if generator_source == "backtrader":
                    if r - 1 < len(preloaded_queue):
                        preloaded_for_round = preloaded_queue[r - 1]
                elif generator_source == "backtrader_agent":
                    assert generator_agent is not None
                    (preloaded_for_round, driver_invo,
                     _rationale, chosen_diff_for_round) = (
                        run_backtrader_agent_generator(
                            round_num=r,
                            history=history,
                            generator_agent=generator_agent,
                            generator_model=generator_model,
                            generator_extra=generator_extra,
                            generator_memory=generator_memory,
                            baseline_cfg=backtrader_cfg,
                            backtrader_dir=backtrader_dir,
                            backtrader_python=backtrader_python,
                            round_dir=match_dir / f"round_{r:03d}",
                            topic_hint=topic_hint,
                        )
                    )
                    if preloaded_for_round is None:
                        # Hybrid generation failed -- forfeit this round
                        # without invoking the solver.
                        result = _build_forfeit_result(
                            r,
                            "backtrader_agent did not produce a valid MCQ",
                            driver_invocation=driver_invo,
                            generator_agent=generator_agent,
                            solver_agent=solver_agent,
                            generator_model=generator_model,
                            solver_model=solver_model,
                            difficulty=chosen_diff_for_round,
                        )
                        history.append(result)
                        gen_score += result.gen_score_delta
                        solver_score += result.solver_score_delta
                        record = {"type": "round", **asdict(result),
                                  "running_gen_score": gen_score,
                                  "running_solver_score": solver_score}
                        logf.write(json.dumps(record) + "\n")
                        logf.flush()
                        print(_c(
                            f"Running score after round {r}:  "
                            f"GENERATOR {gen_score:+d}   SOLVER {solver_score:+d}",
                            _BOLD,
                        ))
                        continue

                if preloaded_for_round is None:
                    result = _build_forfeit_result(
                        r,
                        "backtrader queue produced no question",
                        driver_invocation=driver_invo,
                        generator_agent=generator_agent,
                        solver_agent=solver_agent,
                        generator_model=generator_model,
                        solver_model=solver_model,
                        difficulty=chosen_difficulty,
                    )
                    history.append(result)
                    gen_score += result.gen_score_delta
                    solver_score += result.solver_score_delta
                    record = {"type": "round", **asdict(result),
                              "running_gen_score": gen_score,
                              "running_solver_score": solver_score}
                    logf.write(json.dumps(record) + "\n")
                    logf.flush()
                    print(_c(
                        f"Running score after round {r}:  "
                        f"GENERATOR {gen_score:+d}   SOLVER {solver_score:+d}",
                        _BOLD,
                    ))
                    continue

                result = play_round(
                    r, history,
                    generator_agent=generator_agent,
                    solver_agent=solver_agent,
                    generator_model=generator_model,
                    solver_model=solver_model,
                    generator_extra=generator_extra,
                    solver_extra=solver_extra,
                    generator_memory=generator_memory,
                    solver_memory=solver_memory,
                    match_dir=match_dir,
                    topic_hint=topic_hint,
                    match_log_name=log_path.name,
                    db_index=db_index,
                    preloaded=preloaded_for_round,
                    driver_invocation=driver_invo,
                )
            except KeyboardInterrupt:
                print(_c("\n[interrupted by user]", _YELLOW))
                break
            except Exception as e:  # pragma: no cover -- defensive
                print(_c(f"[round {r} crashed: {e!r}]", _RED))
                continue

            history.append(result)
            gen_score += result.gen_score_delta
            solver_score += result.solver_score_delta

            record = {"type": "round", **asdict(result),
                      "running_gen_score": gen_score,
                      "running_solver_score": solver_score}
            logf.write(json.dumps(record) + "\n")
            logf.flush()

            print(_c(
                f"Running score after round {r}:  "
                f"GENERATOR {gen_score:+d}   SOLVER {solver_score:+d}",
                _BOLD,
            ))

        gen_totals = _usage_totals(history, "generator_usage")
        solver_totals = _usage_totals(history, "solver_usage")
        difficulty_breakdown = _difficulty_breakdown(history)
        parse_method_breakdown = _parse_method_breakdown(history)
        final = {
            "type": "match_final",
            "rounds_played": len(history),
            "gen_score": gen_score,
            "solver_score": solver_score,
            "generator_usage_totals": gen_totals,
            "solver_usage_totals": solver_totals,
            "difficulty_breakdown": difficulty_breakdown,
            "parse_method_breakdown": parse_method_breakdown,
        }
        logf.write(json.dumps(final) + "\n")

    banner("MATCH COMPLETE", _CYAN)
    print(_c(f"Rounds played: {len(history)}   ({symmetry} information)", _BOLD))
    if generator_source == "backtrader":
        gen_role_label = f"source=backtrader:{chosen_difficulty}"
    elif generator_source == "backtrader_agent":
        gen_role_label = (
            f"{generator_agent.name}+backtrader, "  # type: ignore[union-attr]
            f"memory={generator_memory}"
        )
    gen_model_tag = f", model={generator_model}" if generator_model else ""
    sol_model_tag = f", model={solver_model}" if solver_model else ""
    print(_c(
        f"  GENERATOR ({gen_role_label}{gen_model_tag}) "
        f"final score: {gen_score:+d}",
        _MAGENTA,
    ))
    print(_c(
        f"  SOLVER    ({solver_agent.name}, memory={solver_memory}"
        f"{sol_model_tag}) "
        f"final score: {solver_score:+d}",
        _BLUE,
    ))
    if gen_score > solver_score:
        print(_c("Winner: GENERATOR", _MAGENTA + _BOLD))
    elif solver_score > gen_score:
        print(_c("Winner: SOLVER", _BLUE + _BOLD))
    else:
        print(_c("Result: DRAW", _YELLOW + _BOLD))

    print(_c("Token / cost usage:", _BOLD))
    _print_usage_totals("GENERATOR", gen_totals, _MAGENTA)
    _print_usage_totals("SOLVER   ", solver_totals, _BLUE)

    _print_difficulty_breakdown(history)
    _print_parse_method_breakdown(history)

    print(_c(f"Full log: {log_path}", _DIM))
    print(_c(f"Round artifacts: {match_dir}", _DIM))

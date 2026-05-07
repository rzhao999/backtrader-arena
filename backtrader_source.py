"""Backtrader-pipeline-backed question generator for the adversarial arena.

Drives the vendored Backtrader MCQ pipeline in ``Backtrader_MCQ/`` and
feeds its pre-verified questions into the arena one per round.

Because the backtrader pipeline has heavy scientific dependencies
(``backtrader``, ``pandas``, ``yfinance``) that typically live outside
the arena's own venv, we invoke it as a **subprocess** with a
user-selectable Python interpreter. The pipeline writes a JSONL file;
this module parses that file back into :class:`PreloadedQuestion`
records that ``arena.py`` can replay as rounds.

The ``DIFFICULTY_LEVELS`` exposed by this module are the three the user
asked for (``easy`` / ``medium`` / ``hard``); ``all`` is offered as a
convenience for a balanced mix.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Defaults + discovery
# ---------------------------------------------------------------------------

# Canonical location of the vendored backtrader MCQ pipeline. Override via
# the ``BACKTRADER_MCQ_DIR`` env var or arena's ``--backtrader-dir`` flag.
BACKTRADER_DIR_DEFAULT = Path(
    os.environ.get(
        "BACKTRADER_MCQ_DIR",
        str(Path(__file__).resolve().parent / "Backtrader_MCQ"),
    )
)
BACKTRADER_BASE_POOL_FILENAME = "backtrader_mcq_base_pool_all_strategies.jsonl"

# The pipeline depends on pandas/backtrader/yfinance. Arena's own venv
# usually does not have them, so by default we search common system
# Python interpreters for one that does. The user can force a choice
# with the ``BACKTRADER_MCQ_PYTHON`` env var or ``--backtrader-python``.
_CANDIDATE_PYTHONS = [
    sys.executable,
    str(Path(__file__).resolve().parent / ".venv" / "bin" / "python"),
    "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
    "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
    "/usr/bin/python3",
]

# Three levels (per the user's project structure) plus a balanced "all".
DIFFICULTY_LEVELS: tuple[str, ...] = ("easy", "medium", "hard")
DIFFICULTY_CHOICES: tuple[str, ...] = DIFFICULTY_LEVELS + ("all",)


def discover_backtrader_python(user_override: Optional[str] = None) -> Optional[str]:
    """Return the first Python interpreter that can import the pipeline deps.

    Preference order:
      1. ``user_override`` (from ``--backtrader-python``),
      2. ``$BACKTRADER_MCQ_PYTHON``,
      3. the candidate list above.

    Returns ``None`` if no interpreter works -- callers should surface a
    helpful message telling the user to pip-install ``backtrader pandas
    yfinance`` somewhere or pass ``--backtrader-python``.
    """
    candidates: list[str] = []
    if user_override:
        candidates.append(user_override)
    env_py = os.environ.get("BACKTRADER_MCQ_PYTHON")
    if env_py:
        candidates.append(env_py)
    candidates.extend(_CANDIDATE_PYTHONS)

    probe = "import backtrader, pandas, yfinance"
    seen: set[str] = set()
    for py in candidates:
        if not py or py in seen:
            continue
        seen.add(py)
        resolved = shutil.which(py) if not Path(py).is_absolute() else py
        if not resolved or not Path(resolved).exists():
            continue
        try:
            proc = subprocess.run(
                [resolved, "-c", probe],
                capture_output=True, text=True, check=False, timeout=15,
            )
        except Exception:
            continue
        if proc.returncode == 0:
            return resolved
    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PreloadedQuestion:
    """A ready-to-play MCQ that did not come from an LLM agent.

    ``raw_prompt`` carries the full original prompt as written by the
    upstream pipeline (preamble + options + ``<<< X >>>`` instruction).
    When set, the arena uses it verbatim as the solver prompt -- we do
    NOT re-wrap it through ``format.txt`` because that template says
    "Use SciPy" which would contradict the backtrader instructions.

    ``question`` + ``options`` are the structured parse of ``raw_prompt``
    so the arena can still store a sensible ``{question, options}`` row
    in the question DB.
    """
    question: str
    options: dict[str, str]
    answer: str                   # "A" | "B" | "C" | "D"
    raw_prompt: str
    code: str                     # trivial pre-verified stub (see below)
    source: str                   # e.g. "backtrader"
    difficulty: Optional[str] = None
    qtype: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_OPTION_RE = re.compile(r"^([A-D])\.\s+(.+)$", re.MULTILINE)
_RESPOND_RE = re.compile(
    r"\n*\s*Respond in the form\b.*?$", re.IGNORECASE | re.DOTALL,
)


def parse_question(raw_text: str) -> tuple[str, dict[str, str]]:
    """Split a combined MCQ prompt into ``(core_question, options_dict)``.

    Expected input shape (as produced by the backtrader pipeline)::

        <preamble text + actual question body>
        A. option A text
        B. option B text
        C. option C text
        D. option D text

        Respond in the form <<< X >>> ...

    The preamble/body is everything up to the first ``A.`` line. The
    trailing "Respond in the form" sentence is dropped (it belongs to
    the solver instruction, not the core question).
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("empty raw_text")

    matches = list(_OPTION_RE.finditer(raw_text))
    if len(matches) < 4:
        raise ValueError(
            f"could not find all four A/B/C/D option lines "
            f"(found {len(matches)})"
        )

    # Use only the first occurrence per letter in case a preamble mentions
    # "A." in prose. We keep the letter with the smallest start offset.
    by_letter: dict[str, re.Match[str]] = {}
    for m in matches:
        letter = m.group(1)
        if letter not in by_letter:
            by_letter[letter] = m
    for letter in ("A", "B", "C", "D"):
        if letter not in by_letter:
            raise ValueError(f"option {letter} not found")

    first_option_start = by_letter["A"].start()
    core = raw_text[:first_option_start].rstrip()

    options = {letter: by_letter[letter].group(2).strip()
               for letter in ("A", "B", "C", "D")}
    return core, options


# ---------------------------------------------------------------------------
# Interactive difficulty prompt
# ---------------------------------------------------------------------------


def prompt_difficulty(
    *,
    preselect: Optional[str] = None,
    interactive: bool = True,
    stream=sys.stdin,
    out=sys.stdout,
) -> str:
    """Ask the user to pick a difficulty level. Returns one of
    ``"easy" | "medium" | "hard" | "all"``.

    - If ``preselect`` is a valid choice, we return it without prompting.
    - Otherwise, if ``interactive`` and stdin is a TTY, we prompt with a
      numbered menu. Blank input defaults to option 1 (``easy``).
    - If non-interactive and no preselect, we default to ``easy`` and
      print a notice.
    """
    if preselect is not None:
        choice = preselect.strip().lower()
        if choice in DIFFICULTY_CHOICES:
            return choice
        print(
            f"WARNING: --backtrader-difficulty {preselect!r} not recognized; "
            f"falling back to interactive prompt.",
            file=out,
        )

    tty_like = interactive and getattr(stream, "isatty", lambda: False)()
    if not tty_like:
        default = "easy"
        print(
            f"(non-interactive -- defaulting backtrader difficulty to {default!r}; "
            f"pass --backtrader-difficulty to override)",
            file=out,
        )
        return default

    menu = [
        ("1", "easy",   "easy   -- direct single-row lookups"),
        ("2", "medium", "medium -- conditional selection or aggregation"),
        ("3", "hard",   "hard   -- multi-step derived metrics"),
        ("4", "all",    "all    -- balanced mix of all three"),
    ]
    print("\nChoose difficulty for Backtrader-sourced questions:", file=out)
    for key, _val, desc in menu:
        print(f"  [{key}] {desc}", file=out)
    while True:
        try:
            raw = input("Choice [1-4, default=1]: ").strip()
        except EOFError:
            print("(EOF on stdin -- defaulting to easy)", file=out)
            return "easy"
        if raw == "":
            return "easy"
        for key, val, _desc in menu:
            if raw == key or raw.lower() == val:
                return val
        print(f"  ! {raw!r} not recognized. Enter 1, 2, 3, or 4.", file=out)


# ---------------------------------------------------------------------------
# Full interactive config editor (opt-in via ``--backtrader-interactive``)
# ---------------------------------------------------------------------------


def _ask_field(
    label: str,
    default,
    *,
    cast=str,
    choices: Optional[list[str]] = None,
    out=sys.stdout,
):
    """Prompt for a single config value. Blank input keeps the default;
    invalid input re-prompts (up to 3 retries, then keeps the default).

    ``cast`` is applied to non-blank input. ``choices`` (if given) is
    validated *before* casting.
    """
    hint = ""
    if choices:
        hint = f"  options: {' / '.join(choices)}"
    elif cast is int:
        hint = "  (integer)"
    elif cast is float:
        hint = "  (number)"

    for attempt in range(3):
        try:
            raw = input(f"  {label} [{default}]:{hint} ").strip()
        except EOFError:
            print("  (EOF on stdin -- keeping default)", file=out)
            return default
        if raw == "":
            return default
        if choices and raw.lower() not in [c.lower() for c in choices]:
            print(
                f"    ! {raw!r} not one of {choices}. Try again "
                f"(or press Enter to keep {default!r}).",
                file=out,
            )
            continue
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(
                f"    ! could not parse {raw!r} as {cast.__name__}. "
                f"Try again (or press Enter to keep {default!r}).",
                file=out,
            )
            continue
    print(f"  (too many invalid attempts -- keeping default {default!r})", file=out)
    return default


def _looks_like_iso_date(s: str) -> str:
    """Validator that accepts a YYYY-MM-DD string and returns it unchanged."""
    import datetime as _dt
    _dt.datetime.strptime(s, "%Y-%m-%d")  # raises ValueError if malformed
    return s


def prompt_full_config(
    defaults: Optional["BacktraderGenConfig"] = None,
    *,
    preselect_difficulty: Optional[str] = None,
    allow_smart_defaults: bool = False,
    stream=sys.stdin,
    out=sys.stdout,
) -> tuple["BacktraderGenConfig", str]:
    """Walk the user through every backtrader knob, then the difficulty.

    Returns a tuple ``(BacktraderGenConfig, difficulty)``. If stdin is
    not a TTY, we don't prompt -- we just return the supplied defaults
    (and ``preselect_difficulty`` if set, else ``"easy"``).

    This is opt-in via ``arena.py --backtrader-interactive``; the
    default arena flow only prompts for difficulty (or nothing if
    ``--no-backtrader-prompt``). For ``backtrader_agent`` callers can
    set ``allow_smart_defaults`` so the user may skip detailed fields and
    let the LLM-authored generator receive rotating reference defaults.
    """
    base = defaults or BacktraderGenConfig()

    tty_like = getattr(stream, "isatty", lambda: False)()
    if not tty_like:
        print(
            "(non-interactive -- skipping backtrader interactive config; "
            "using defaults / CLI overrides)",
            file=out,
        )
        diff = preselect_difficulty if preselect_difficulty in DIFFICULTY_CHOICES else "easy"
        return base, diff

    print("\n[backtrader] interactive config editor", file=out)
    print(
        "  Press Enter on any prompt to keep the default shown in [brackets].",
        file=out,
    )
    if allow_smart_defaults:
        print(
            "  Hybrid mode can auto-vary symbol / strategy / seed / "
            "difficulty per round from these defaults.",
            file=out,
        )
        while True:
            try:
                raw = input("Use smart per-round defaults and skip detailed Backtrader fields? [Y/n]: ").strip().lower()
            except EOFError:
                raw = ""
            if raw in ("", "y", "yes"):
                diff = preselect_difficulty if preselect_difficulty in DIFFICULTY_CHOICES else "auto"
                print("  (using smart per-round Backtrader defaults)\n", file=out)
                return base, diff
            if raw in ("n", "no"):
                break
            print("  Please answer y or n.", file=out)

    cfg_dict: dict = {
        "symbol":          _ask_field("Symbol",          base.symbol,          out=out),
        "start_date":      _ask_field("Start date YYYY-MM-DD", base.start_date,
                                       cast=_looks_like_iso_date, out=out),
        "end_date":        _ask_field("End date   YYYY-MM-DD", base.end_date,
                                       cast=_looks_like_iso_date, out=out),
        "initial_cash":    _ask_field("Initial cash",    base.initial_cash,    cast=float, out=out),
        "commission_rate": _ask_field("Commission rate", base.commission_rate, cast=float, out=out),
        "stake":           _ask_field("Stake (shares)",  base.stake,           cast=int,   out=out),
        "strategy":        _ask_field("Strategy",        base.strategy,
                                       choices=list(REGISTERED_BACKTRADER_STRATEGIES), out=out),
        "seed":            _ask_field("Random seed",     base.seed,            cast=int, out=out),
    }

    # Difficulty has its own dedicated picker UX (numbered menu) -- reuse it.
    difficulty = prompt_difficulty(
        preselect=preselect_difficulty,
        interactive=True,
        stream=stream,
        out=out,
    )

    cfg = BacktraderGenConfig(**cfg_dict)

    print("\n[backtrader] final config:", file=out)
    for k, v in cfg_dict.items():
        print(f"  {k:<16} = {v}", file=out)
    print(f"  {'difficulty':<16} = {difficulty}", file=out)

    while True:
        try:
            confirm = input("Proceed with this config? [Y/n]: ").strip().lower()
        except EOFError:
            confirm = ""
        if confirm in ("", "y", "yes"):
            print(file=out)
            return cfg, difficulty
        if confirm in ("n", "no"):
            print("  Restarting interactive config...\n", file=out)
            return prompt_full_config(
                defaults=cfg,  # let user iterate from their last attempt
                preselect_difficulty=difficulty,
                allow_smart_defaults=allow_smart_defaults,
                stream=stream, out=out,
            )
        print("  Please answer y or n.", file=out)


# ---------------------------------------------------------------------------
# Subprocess-driven generation
# ---------------------------------------------------------------------------


# Arena-side defaults for the backtrader subprocess. These are chosen
# to be a reasonable, reproducible setting for adversarial MCQ matches:
# AAPL over 2020-01-01..2024-01-01 spans a full bull run + 2022 drawdown
# + 2023 recovery, which gives the question generator enough variety
# (clear trends, drawdowns, multiple crossovers) without depending on
# whatever the upstream pipeline happens to ship as its hardcoded
# fallback. They can still be overridden per-flag from ``arena.py``
# (``--backtrader-symbol``, ``--backtrader-start-date``, etc.).
DEFAULT_SYMBOL = "AAPL"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-01-01"
DEFAULT_SEED = 42
DEFAULT_INITIAL_CASH = 50000.0
DEFAULT_COMMISSION_RATE = 0.001
DEFAULT_STAKE = 1000
DEFAULT_STRATEGY = "sma_crossover"

# Keep aligned with ``Backtrader_MCQ/strategies.py`` STRATEGIES keys.
REGISTERED_BACKTRADER_STRATEGIES: tuple[str, ...] = (
    "sma_crossover",
    "rolling_window_mean",
    "exponential_weighted_moving_average",
    "rsi_strategy",
    "macd_crossover",
)


@dataclass
class BacktraderGenConfig:
    """Non-difficulty knobs forwarded to the pipeline subprocess.

    Defaults are arena-side (NOT the upstream pipeline's
    ``backtrader_info.CONFIG`` -- which ships with MSFT 2011-2013 and
    isn't a great match for an adversarial benchmark). Pass ``None`` to
    explicitly defer to the upstream default for that field.
    """
    symbol: Optional[str] = DEFAULT_SYMBOL
    start_date: Optional[str] = DEFAULT_START_DATE
    end_date: Optional[str] = DEFAULT_END_DATE
    initial_cash: Optional[float] = DEFAULT_INITIAL_CASH
    commission_rate: Optional[float] = DEFAULT_COMMISSION_RATE
    stake: Optional[int] = DEFAULT_STAKE
    strategy: Optional[str] = DEFAULT_STRATEGY
    seed: Optional[int] = DEFAULT_SEED


def _build_builder_argv(
    *,
    difficulty: str,
    num_questions: int,
    output_path: Path,
    cfg: BacktraderGenConfig,
) -> list[str]:
    """Build argv for ``backtrader_MCQ_builder.py`` (the stage of the
    pipeline that just writes the JSONL -- we don't need the verifier
    stage again because the builder already self-verifies)."""
    argv = [
        "-o", str(output_path),
        "--num-questions", str(num_questions),
        "--difficulty", str(difficulty),
    ]
    if cfg.symbol:          argv += ["--symbol", cfg.symbol]
    if cfg.start_date:      argv += ["--start-date", cfg.start_date]
    if cfg.end_date:        argv += ["--end-date", cfg.end_date]
    if cfg.initial_cash is not None:
        argv += ["--initial-cash", str(cfg.initial_cash)]
    if cfg.commission_rate is not None:
        argv += ["--commission-rate", str(cfg.commission_rate)]
    if cfg.stake is not None:
        argv += ["--stake", str(cfg.stake)]
    if cfg.strategy:        argv += ["--strategy", cfg.strategy]
    if cfg.seed is not None: argv += ["--seed", str(cfg.seed)]
    return argv


def generate_questions(
    *,
    difficulty: str,
    num_questions: int,
    backtrader_dir: Optional[Path] = None,
    python_interpreter: Optional[str] = None,
    cfg: Optional[BacktraderGenConfig] = None,
    output_dir: Optional[Path] = None,
    timeout_s: int = 900,
    echo: bool = True,
) -> list[PreloadedQuestion]:
    """Run the backtrader MCQ builder via subprocess and parse its output.

    Raises ``RuntimeError`` with a clear message if anything goes wrong
    (missing directory, missing interpreter with deps, non-zero exit,
    unparseable output, fewer questions produced than requested).
    """
    cfg = cfg or BacktraderGenConfig()
    bt_dir = (backtrader_dir or BACKTRADER_DIR_DEFAULT).resolve()
    builder = bt_dir / "backtrader_MCQ_builder.py"
    if not builder.exists():
        raise RuntimeError(
            f"backtrader builder not found: {builder}. "
            "Pass --backtrader-dir or set $BACKTRADER_MCQ_DIR."
        )

    py = discover_backtrader_python(python_interpreter)
    if not py:
        raise RuntimeError(
            "No Python interpreter with backtrader/pandas/yfinance was "
            "found. Either pip-install those into the arena venv, or "
            "pass --backtrader-python <path>, or set "
            "$BACKTRADER_MCQ_PYTHON."
        )

    # Keep generated JSON alongside the arena's other logs when the caller
    # provides an output_dir; otherwise use a private tempdir.
    if output_dir is not None:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"backtrader_{difficulty.replace(',', '-')}_n{num_questions}.json"
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="arena_backtrader_")
        out_path = Path(tmp_ctx.name) / "questions.json"

    argv = [py, str(builder)] + _build_builder_argv(
        difficulty=difficulty,
        num_questions=num_questions,
        output_path=out_path,
        cfg=cfg,
    )
    if echo:
        print(f"[backtrader] running: {' '.join(argv)}", file=sys.stderr)
    try:
        proc = subprocess.run(
            argv,
            capture_output=True, text=True,
            cwd=str(bt_dir),
            timeout=timeout_s,
            check=False,
        )
    finally:
        pass

    if proc.returncode != 0:
        tail_out = (proc.stdout or "").strip().splitlines()[-20:]
        tail_err = (proc.stderr or "").strip().splitlines()[-20:]
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        raise RuntimeError(
            f"backtrader builder failed (exit {proc.returncode}).\n"
            f"stdout tail:\n  " + "\n  ".join(tail_out) + "\n" +
            f"stderr tail:\n  " + "\n  ".join(tail_err)
        )

    if not out_path.exists():
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        raise RuntimeError(
            f"backtrader builder finished but did not write {out_path}."
        )

    records = _load_jsonl(out_path)
    questions = [_record_to_preloaded(r) for r in records]

    # The builder may over-produce or under-produce slightly depending
    # on the difficulty pool. Trim or warn accordingly.
    if len(questions) < num_questions:
        print(
            f"[backtrader] WARNING: requested {num_questions} questions, "
            f"pipeline produced {len(questions)}. Proceeding with what we "
            f"have; extra rounds will trigger generator forfeit.",
            file=sys.stderr,
        )
    elif len(questions) > num_questions:
        questions = questions[:num_questions]

    if tmp_ctx is not None:
        # Only cleanup after we've fully read the file above.
        tmp_ctx.cleanup()
    return questions


def _load_jsonl(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    # The builder writes JSONL; be forgiving and also accept a JSON array.
    try:
        blob = json.loads(raw)
    except json.JSONDecodeError:
        blob = None
    if isinstance(blob, list):
        return [r for r in blob if isinstance(r, dict)]
    if isinstance(blob, dict):
        return [blob]
    out: list[dict] = []
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"unparseable JSONL line {i} in {path}: {e}"
            ) from e
    return out


def _record_to_preloaded(record: dict) -> PreloadedQuestion:
    raw_prompt = record.get("question") or ""
    answer = str(record.get("answer") or "").strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise RuntimeError(f"bad answer letter {answer!r} in record {record}")

    core, options = parse_question(raw_prompt)
    # The pipeline's own "Respond in the form" instruction is the same
    # as arena's solver protocol, so we can use the raw prompt verbatim.
    difficulty = record.get("difficulty")
    qtype = record.get("type")

    code = (
        f"# Pre-verified by the Backtrader MCQ pipeline.\n"
        f"# qtype={qtype or '?'}, difficulty={difficulty or '?'}.\n"
        f"print(\"ANSWER:\", {answer!r})\n"
    )
    return PreloadedQuestion(
        question=core,
        options=options,
        answer=answer,
        raw_prompt=raw_prompt,
        code=code,
        source="backtrader",
        difficulty=difficulty,
        qtype=qtype,
        metadata={
            "benchmark": record.get("benchmark"),
            "author": record.get("author"),
        },
    )

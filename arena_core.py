"""
Core constants, paths, and the per-round result dataclass shared by the
rest of the arena modules.

Importing this module keeps the arena's root path available to sibling
modules. Agent backends are vendored locally in :mod:`agent_backends`, so
the GitHub repo does not depend on the parent ``AI_Agents`` workspace.

The two timeout constants (``AGENT_TIMEOUT_S`` and ``CODE_TIMEOUT_S``)
are mutated from :mod:`arena`'s ``main`` after CLI parsing. Other
modules MUST read them via attribute access on this module (e.g.
``import arena_core as _core; _core.AGENT_TIMEOUT_S``) rather than
``from arena_core import AGENT_TIMEOUT_S`` so the override propagates.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VENV_PY = ROOT / ".venv" / "bin" / "python"
PROMPTS_DIR = ROOT / "prompts"
LOGS_DIR = ROOT / "logs"
FORMAT_TEMPLATE = ROOT / "format.txt"
BACKTRADER_AGENT_TEMPLATE = PROMPTS_DIR / "backtrader_agent.txt"


# ---------------------------------------------------------------------------
# Defaults: agents, model, memory mode, generator source
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "composer-2"
# Default LLM CLI backend for both roles. The "cursor" registry entry wraps
# Cursor's `agent` binary, so a plain `python arena.py` runs an LLM-driven
# generator and an LLM-driven solver without extra flags.
DEFAULT_GENERATOR_AGENT = "cursor"
DEFAULT_SOLVER_AGENT = "cursor"

# Memory modes available for each role:
#   - "sandbox": every round is independent. The agent does NOT see prior
#     rounds of this match. This is the default for both roles -- it keeps
#     rounds truly i.i.d. and prevents trivial single-model correlation
#     exploits (e.g. the generator memorizing which tricks worked).
#   - "history": the agent is shown a summary of past rounds, so it can
#     adapt (generator: hunt the solver's weaknesses; solver: recognize
#     recurring traps).
# Combining the two roles independently yields all four regimes --
# symmetric (both sandbox or both history) as well as the interesting
# asymmetric ones.
MEMORY_MODES = ("sandbox", "history")
DEFAULT_GENERATOR_MEMORY = "history"
DEFAULT_SOLVER_MEMORY = "sandbox"

# Where questions come from each round:
#   - "backtrader":       pre-generate the full queue from the Backtrader
#                         MCQ pipeline and replay one pre-verified
#                         question per round (no LLM in the generator
#                         loop; deterministic).
#   - "backtrader_agent": LLM-authored Backtrader MCQ mode (default). An LLM
#                         generator CLI is invoked each round and writes the
#                         full MCQ plus executable verification code.
#                         Combined with --generator-memory=history the LLM can
#                         adapt the authored question to the solver's past
#                         performance.
GENERATOR_SOURCES = ("backtrader", "backtrader_agent")
DEFAULT_GENERATOR_SOURCE = "backtrader_agent"


# ---------------------------------------------------------------------------
# Timeouts (mutated by ``arena.main`` from --agent-timeout / --code-timeout)
# ---------------------------------------------------------------------------

# How long to wait for each agent invocation. Composer 2 usually finishes
# a round in seconds; thinking / Opus variants can need many minutes, so
# override --agent-timeout when using those.
AGENT_TIMEOUT_S = 600
# How long the generator's verification code is allowed to run.
CODE_TIMEOUT_S = 60

# Whether to wrap role invocations in an OS sandbox (best-effort).
# Mutated by arena_match from each role's memory mode. "sandbox" memory
# enables this for wrappers that support agent_backends.maybe_wrap_macos_sandbox_exec.
GENERATOR_OS_SANDBOX = False
SOLVER_OS_SANDBOX = False


# ---------------------------------------------------------------------------
# Per-round result data model
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    round: int
    question: str
    options: dict           # {"A": ..., "B": ..., "C": ..., "D": ...}
    declared_answer: Optional[str]
    ground_truth: Optional[str]      # from verification code
    solver_answer: Optional[str]
    solver_correct: Optional[bool]
    generator_forfeit: bool
    gen_score_delta: int
    solver_score_delta: int
    code: str
    code_stdout: str
    code_stderr: str
    generator_raw: str
    solver_raw: str
    generator_agent: str = ""
    solver_agent: str = ""
    generator_model: Optional[str] = None
    solver_model: Optional[str] = None
    generator_usage: Optional[dict] = None
    solver_usage: Optional[dict] = None
    elapsed_sec: float = 0.0
    notes: str = ""
    difficulty: Optional[str] = None
    # How the solver's letter was recovered from its output:
    #   "strict"          -- a literal <<< X >>> was emitted (preferred).
    #   "heuristic"       -- recovered via agent_backends.ANSWER_HINT_RE
    #                        fallbacks (e.g. "answer is C").
    #   "numeric_closest" -- no letter found; recovered by parsing a
    #                        computed numeric value from solver output
    #                        and selecting the closest numeric option.
    #   "none"            -- nothing parseable; the solver effectively
    #                        forfeited this round on the answer side.
    solver_parse_method: str = "none"


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _c(s: str, color: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"{color}{s}{_RESET}"


def banner(text: str, color: str = _CYAN) -> None:
    line = "=" * max(8, len(text) + 4)
    print()
    print(_c(line, color))
    print(_c(f"  {text}", color))
    print(_c(line, color))


def _fmt_usage(u: Optional[dict]) -> str:
    if not u:
        return "(no usage reported)"
    parts: list[str] = []
    model = u.get("model")
    if model:
        parts.append(f"model={model}")
    in_t, out_t = u.get("input_tokens"), u.get("output_tokens")
    if in_t is not None or out_t is not None:
        parts.append(f"in={in_t or 0} out={out_t or 0}")
    cr = u.get("cache_read_tokens")
    cc = u.get("cache_creation_tokens")
    if cr or cc:
        parts.append(f"cache_r={cr or 0} cache_c={cc or 0}")
    cost = u.get("total_cost_usd")
    if isinstance(cost, (int, float)):
        parts.append(f"cost=${cost:.4f}")
    turns = u.get("num_turns")
    if turns is not None:
        parts.append(f"turns={turns}")
    return ", ".join(parts) or "(no usage reported)"


# ---------------------------------------------------------------------------
# Text and answer parsing
# ---------------------------------------------------------------------------

_TAG_RE_CACHE: dict[str, re.Pattern[str]] = {}
_ANSWER_LINE_RE = re.compile(r"^\s*ANSWER:\s*([ABCD])\s*$", re.MULTILINE)
_SOLVER_ANSWER_RE = re.compile(r"<<<\s*([ABCD])\s*>>>")
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _tag_re(name: str) -> re.Pattern[str]:
    if name not in _TAG_RE_CACHE:
        _TAG_RE_CACHE[name] = re.compile(
            rf"<<<{re.escape(name)}>>>\s*(.*?)\s*<<</{re.escape(name)}>>>",
            re.DOTALL,
        )
    return _TAG_RE_CACHE[name]


def extract_block(text: str, name: str) -> Optional[str]:
    m = _tag_re(name).search(text)
    return m.group(1) if m else None


def _mcq_hint_regexes() -> list:
    """Return ``agent_backends.ANSWER_HINT_RE`` without importing it eagerly."""
    try:
        from agent_backends import ANSWER_HINT_RE  # type: ignore
        return list(ANSWER_HINT_RE)
    except Exception:  # pragma: no cover -- defensive fallback
        return []


def parse_code_answer(stdout: str) -> Optional[str]:
    """Return the last ``ANSWER: X`` line from verification-code stdout."""
    matches = _ANSWER_LINE_RE.findall(stdout)
    return matches[-1] if matches else None


def parse_solver_answer(text: str) -> Optional[str]:
    """Return the last ``<<< X >>>`` letter the solver emitted."""
    matches = _SOLVER_ANSWER_RE.findall(text)
    return matches[-1] if matches else None


def choose_solver_answer(text: str) -> tuple[Optional[str], str]:
    """Parse a solver letter strictly first, then via shared MCQ hints."""
    strict = parse_solver_answer(text)
    if strict:
        return strict, "strict"
    body = text or ""
    for rx in _mcq_hint_regexes():
        m = rx.search(body)
        if m:
            return m.group(1).upper(), "heuristic"
    return None, "none"


def _parse_option_number(s: str) -> Optional[float]:
    if not s:
        return None
    m = _NUM_RE.search(str(s))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def choose_solver_answer_with_options(
    text: str,
    *,
    options: dict[str, str],
) -> tuple[Optional[str], str]:
    """Pick a solver answer using strict/heuristic, else numeric closest."""
    letter, method = choose_solver_answer(text)
    if letter is not None:
        return letter, method

    opt_vals: dict[str, float] = {}
    for k in ("A", "B", "C", "D"):
        v = _parse_option_number(options.get(k, ""))
        if v is None:
            return None, "none"
        opt_vals[k] = v

    nums = _extract_numbers(text)
    if not nums:
        return None, "none"

    target = nums[-1]
    best = min(opt_vals, key=lambda k: abs(opt_vals[k] - target))
    return best, "numeric_closest"


def _logical_text(stdout: str) -> str:
    """Peel off an agent CLI JSON wrapper when one surrounds the model text."""
    if not stdout:
        return stdout or ""
    stripped = stdout.strip()
    if not stripped:
        return stdout

    def _from_obj(obj: Any) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        result = obj.get("result")
        if isinstance(result, str) and result.strip():
            return result
        for key in ("content", "message", "output", "text"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return None

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        obj = None
    else:
        text = _from_obj(obj)
        if text is not None:
            return text

    best: Optional[str] = None
    for line in stripped.splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = _from_obj(obj)
        if text and (best is None or len(text) > len(best)):
            best = text
    return best if best is not None else stdout


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def load_format_template() -> str:
    """Load ``format.txt`` and extract the triple-quoted template body."""
    raw = FORMAT_TEMPLATE.read_text()
    m = re.search(r'"""\s*\n(.*?)\n"""', raw, re.DOTALL)
    if not m:
        raise RuntimeError(
            "Could not locate the triple-quoted template inside format.txt"
        )
    return m.group(1)


SOLVER_WORK_PROTOCOL = (
    "You MUST solve this MCQ by writing and running Python code -- do "
    "not answer from memory. Be FAST and direct: this is a short, "
    "single-question task, not an exploration session.\n"
    "\n"
    "Procedure (do these and nothing else):\n"
    "  1. Write your full solution code to ``./solution.py`` (relative to "
    "your current working directory). Do NOT inspect / list / search the "
    "workspace first -- it is intentionally empty except for this prompt.\n"
    "  2. Run it ONCE with ``python solution.py`` (or your bash/shell "
    "tool). Do not delegate to a subagent.\n"
    "  3. Read the output, then immediately reply with your final answer "
    "in the form ``<<< X >>>`` where X is A, B, C, or D.\n"
    "\n"
    "Do not skip steps 1-2. Do not retry the workflow on a different "
    "path -- ``./solution.py`` is writable. The arena grader reads the "
    "printed ``<<< X >>>`` letter; the saved ``solution.py`` is kept as "
    "evidence of how you arrived at it.\n\n"
)


def _truncate_block(text: str, *, limit: int = 1200) -> str:
    body = (text or "").strip()
    if len(body) <= limit:
        return body or "(empty)"
    return body[:limit].rstrip() + "\n    ... [truncated]"


def _render_question_with_options(r: RoundResult, *, limit: int = 1200) -> str:
    opts = r.options or {}
    option_lines = [
        f"    {letter}. {(opts.get(letter) or '').strip()}"
        for letter in ("A", "B", "C", "D")
        if (opts.get(letter) or "").strip()
    ]
    question = textwrap.indent(_truncate_block(r.question, limit=limit), "    ")
    if option_lines:
        question += "\n" + "\n".join(option_lines)
    return question


def render_history(history: list[RoundResult]) -> str:
    """Render past rounds for the generator's view."""
    if not history:
        return "(no rounds yet -- this is round 1)"
    chunks: list[str] = []
    for r in history:
        gt = r.ground_truth or "?"
        sa = r.solver_answer or "?"
        outcome = (
            "GENERATOR FORFEIT" if r.generator_forfeit
            else ("SOLVER CORRECT" if r.solver_correct else "SOLVER WRONG")
        )
        chunks.append(textwrap.dedent(f"""\
            Round {r.round}: {outcome}
              generated question:
{_render_question_with_options(r)}
              ground truth: {gt}    solver answered: {sa}
        """))
    return "\n".join(chunks)


def render_solver_history(history: list[RoundResult]) -> str:
    """Render past scorable rounds for the solver's view."""
    scorable = [r for r in history if not r.generator_forfeit]
    if not scorable:
        return "(no previous scorable rounds in this match)"

    chunks: list[str] = []
    for r in scorable:
        my_ans = r.solver_answer or "?"
        truth = r.ground_truth or "?"
        outcome = "CORRECT" if r.solver_correct else "WRONG"
        chunks.append(textwrap.dedent(f"""\
            Round {r.round}: your answer={my_ans}  truth={truth}  [{outcome}]
              generated question:
{_render_question_with_options(r)}
        """))
    return "\n".join(chunks)


def build_solver_prompt(
    question: str,
    options: dict[str, str],
    *,
    history: Optional[list[RoundResult]] = None,
    memory: str = DEFAULT_SOLVER_MEMORY,
) -> str:
    """Render the solver prompt for this round."""
    base = load_format_template().format(
        QUESTION=question.strip(),
        OPTION_A=options["A"].strip(),
        OPTION_B=options["B"].strip(),
        OPTION_C=options["C"].strip(),
        OPTION_D=options["D"].strip(),
    )
    if memory != "history":
        return SOLVER_WORK_PROTOCOL + base

    past = render_solver_history(history or [])
    preamble = (
        "You are continuing a multi-round adversarial MCQ match against "
        "the same generator agent. Summary of your previous attempts in "
        "this match (oldest first) -- use them only to anticipate the "
        "generator's style; each round is still judged on its own "
        "question.\n\n"
        f"{past}\n\n"
        "---- Current round ----\n\n"
    )
    return SOLVER_WORK_PROTOCOL + preamble + base


# ---------------------------------------------------------------------------
# Verification-code execution
# ---------------------------------------------------------------------------


def run_verification_code(
    code: str,
    round_num: int,
    *,
    work_dir: Path | None = None,
    python_path: str | Path | None = None,
    pythonpath: list[str | Path] | None = None,
) -> tuple[str, str, int]:
    """Run verification code. Returns stdout/stderr/rc."""
    work_dir = (work_dir or (LOGS_DIR / "code_runs")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / f"round_{round_num:03d}.py"
    script_path.write_text(code)

    py = Path(python_path) if python_path is not None else VENV_PY
    if not py.exists():
        return "", f"[arena] verifier python missing at {py}", 127
    env = os.environ.copy()
    if pythonpath:
        extra = os.pathsep.join(str(Path(p)) for p in pythonpath)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = extra if not existing else extra + os.pathsep + existing

    try:
        proc = subprocess.run(
            [str(py), str(script_path)],
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT_S,
            cwd=str(work_dir),
            env=env,
            check=False,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        return (e.stdout or ""), f"[arena] code timed out after {CODE_TIMEOUT_S}s", 124

"""
Append-only JSONL database for the adversarial-MCQ arena.

Two files live under `db/`:

  - `questions.jsonl`  -- one JSON record per *unique* question (deduplicated
    by SHA-256 of the canonicalized question text + the four option strings).
    Each record carries a `well_defined` boolean: True iff the generator's
    verification code ran cleanly (returncode 0, no timeout) AND printed
    exactly one parseable `ANSWER: <A|B|C|D>` line.

  - `attempts.jsonl` -- one JSON record per solver attempt, linked to a
    question by `question_id`. Always appended (no dedupe), so we can study
    how often a given question fools the solver across matches.

Both files are line-delimited JSON ("JSON Lines"), so they're trivially
greppable, easy to back up, and crash-safe to append to.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "db"
QUESTIONS_PATH = DB_DIR / "questions.jsonl"
ATTEMPTS_PATH = DB_DIR / "attempts.jsonl"

_ANSWER_LINE_RE = re.compile(r"^\s*ANSWER:\s*([ABCD])\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Hashing / canonicalization
# ---------------------------------------------------------------------------


def _normalize_ws(s: str) -> str:
    """Collapse runs of whitespace; strip leading/trailing whitespace."""
    return re.sub(r"\s+", " ", s).strip()


def question_hash(question: str, options: dict[str, str]) -> str:
    """Stable SHA-256 hex digest over (question + four options).

    Whitespace is normalized so that trivial reformatting of the same question
    does not produce a different hash.
    """
    parts = [_normalize_ws(question)]
    for letter in ("A", "B", "C", "D"):
        parts.append(f"{letter}:{_normalize_ws(options.get(letter, ''))}")
    canonical = "\n".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Well-definedness check
# ---------------------------------------------------------------------------


def evaluate_well_defined(
    *,
    question: Optional[str],
    options: dict[str, str],
    code: Optional[str],
    code_stdout: str,
    code_stderr: str,
    code_returncode: int,
    code_timed_out: bool,
) -> tuple[bool, list[str]]:
    """Return (is_well_defined, list_of_failure_reasons).

    Per the project spec, well-defined means: the generator produced a
    question + four options + verification code, and the code ran cleanly
    and printed exactly one parseable `ANSWER: <A|B|C|D>` line.
    """
    reasons: list[str] = []
    if not question or not question.strip():
        reasons.append("missing question text")
    for letter in ("A", "B", "C", "D"):
        if not options.get(letter, "").strip():
            reasons.append(f"missing option {letter}")
    if not code or not code.strip():
        reasons.append("missing verification code")
    if code_timed_out:
        reasons.append("verification code timed out")
    elif code_returncode != 0:
        reasons.append(
            f"verification code exited with non-zero status {code_returncode}"
        )

    answer_lines = _ANSWER_LINE_RE.findall(code_stdout or "")
    if len(answer_lines) == 0:
        reasons.append(
            "verification code did not print any 'ANSWER: <A|B|C|D>' line"
        )
    elif len(answer_lines) > 1:
        # If the code prints multiple answer lines, the ground truth is
        # ambiguous unless they all agree. Even then we treat it as a
        # smell -- the generator should print exactly one.
        unique = sorted(set(answer_lines))
        if len(unique) == 1:
            reasons.append(
                f"verification code printed multiple ANSWER lines "
                f"(all '{unique[0]}', but still ambiguous form)"
            )
        else:
            reasons.append(
                f"verification code printed conflicting ANSWER lines: "
                f"{unique}"
            )

    # Suppress noisy stderr from the well-defined gate (warnings are OK as
    # long as returncode is 0 and the answer line is unique).
    _ = code_stderr

    return (len(reasons) == 0), reasons


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _ensure_dir() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_question_index() -> dict[str, dict[str, Any]]:
    """Return {question_id: latest_record} for fast dedup checks."""
    index: dict[str, dict[str, Any]] = {}
    for rec in _iter_jsonl(QUESTIONS_PATH):
        qid = rec.get("id")
        if qid:
            index[qid] = rec
    return index


def iter_questions() -> Iterator[dict[str, Any]]:
    yield from _iter_jsonl(QUESTIONS_PATH)


def iter_attempts() -> Iterator[dict[str, Any]]:
    yield from _iter_jsonl(ATTEMPTS_PATH)


def attempts_for(question_id: str) -> list[dict[str, Any]]:
    return [a for a in iter_attempts() if a.get("question_id") == question_id]


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def record_question(
    *,
    question: str,
    options: dict[str, str],
    code: str,
    declared_answer: Optional[str],
    ground_truth: Optional[str],
    code_stdout: str,
    code_stderr: str,
    code_returncode: int,
    code_timed_out: bool,
    well_defined: bool,
    well_defined_reasons: list[str],
    match_log: str,
    round_num: int,
    generator_model: str,
    topic_hint: Optional[str],
    existing_index: Optional[dict[str, dict[str, Any]]] = None,
) -> tuple[str, bool]:
    """Append the question to `questions.jsonl` if it isn't already present.

    Returns (question_id, was_new_insert).
    """
    _ensure_dir()
    qid = question_hash(question, options)
    index = existing_index if existing_index is not None else load_question_index()
    if qid in index:
        return qid, False

    record: dict[str, Any] = {
        "type": "question",
        "id": qid,
        "first_seen": _now_iso(),
        "first_seen_match": match_log,
        "first_seen_round": round_num,
        "generator_model": generator_model,
        "topic_hint": topic_hint,
        "question": question,
        "options": {k: options.get(k, "") for k in ("A", "B", "C", "D")},
        "code": code,
        "declared_answer": declared_answer,
        "ground_truth": ground_truth,
        "code_returncode": code_returncode,
        "code_timed_out": code_timed_out,
        "code_stdout": code_stdout,
        "code_stderr": code_stderr,
        "well_defined": well_defined,
        "well_defined_reason": (
            "code ran cleanly and printed a single ANSWER line"
            if well_defined else None
        ),
        "ill_defined_reasons": [] if well_defined else well_defined_reasons,
    }
    with QUESTIONS_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if existing_index is not None:
        existing_index[qid] = record
    return qid, True


def record_attempt(
    *,
    question_id: str,
    solver_model: str,
    solver_answer: Optional[str],
    ground_truth: Optional[str],
    correct: Optional[bool],
    generator_forfeit: bool,
    match_log: str,
    round_num: int,
    solver_raw_excerpt: str = "",
    solver_usage: Optional[dict[str, Any]] = None,
    solver_parse_method: str = "none",
) -> None:
    """Append a single solver-attempt record to `attempts.jsonl`.

    ``solver_usage`` is the normalized dict produced by
    ``agent_backends.extract_usage`` (tokens, cache hits, cost, model).
    It is recorded verbatim when present so per-attempt cost analytics
    are possible without re-running agents.

    ``solver_parse_method`` mirrors the field on ``RoundResult`` and is
    one of ``"strict"`` / ``"heuristic"`` / ``"none"`` -- it tells you,
    after the fact, whether the solver actually emitted the canonical
    ``<<< X >>>`` wrapper or whether the letter was recovered loosely.
    """
    _ensure_dir()
    record = {
        "type": "attempt",
        "timestamp": _now_iso(),
        "question_id": question_id,
        "match_log": match_log,
        "round": round_num,
        "solver_model": solver_model,
        "solver_answer": solver_answer,
        "solver_parse_method": solver_parse_method,
        "ground_truth": ground_truth,
        "correct": correct,
        "generator_forfeit": generator_forfeit,
        "solver_raw_excerpt": solver_raw_excerpt[:500],
    }
    if solver_usage:
        record["solver_usage"] = solver_usage
    with ATTEMPTS_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Rollups for `list` / `export` commands
# ---------------------------------------------------------------------------


def question_stats(question_id: str) -> dict[str, int]:
    """Aggregate solver attempts for a given question."""
    n_total = 0
    n_correct = 0
    for a in iter_attempts():
        if a.get("question_id") != question_id:
            continue
        if a.get("generator_forfeit"):
            continue
        n_total += 1
        if a.get("correct"):
            n_correct += 1
    return {"attempts": n_total, "solver_correct": n_correct}


def filter_questions(
    *,
    well_defined_only: bool = False,
    ill_defined_only: bool = False,
) -> Iterable[dict[str, Any]]:
    """Iterate questions, optionally filtered by well-definedness."""
    if well_defined_only and ill_defined_only:
        return iter([])  # mutually exclusive
    out: list[dict[str, Any]] = []
    for q in iter_questions():
        if well_defined_only and not q.get("well_defined"):
            continue
        if ill_defined_only and q.get("well_defined"):
            continue
        out.append(q)
    return out

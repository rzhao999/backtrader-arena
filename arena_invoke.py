"""
Agent invocation glue: resolves agent names to ``AgentDef`` objects,
invokes them with a prompt + role-specific extras, and packages the
result (stdout / stderr / returncode + extracted token usage) into an
``AgentInvocation``.

Also hosts the small interactive ``-I`` config editor for the
arena-level knobs (rounds + per-role agent / model / memory).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import arena_core as _core  # arena_core first: tweaks sys.path
from arena_core import DEFAULT_MODEL, MEMORY_MODES, _RED, _c

from agent_backends import AgentDef, extract_usage, get_agent, list_agents  # noqa: E402
import backtrader_source as bt_source  # noqa: E402


@dataclass
class AgentInvocation:
    """Result of invoking an agent for one round-role."""
    stdout: str
    stderr: str
    returncode: int
    usage: Optional[dict]
    cmd_ok: bool = True


def _extract_copilot_usage_from_otel(otel_path: Path) -> Optional[dict]:
    """Best-effort: extract token usage from Copilot OTel JSONL file.

    Copilot can export GenAI-semconv-compatible telemetry; the exact JSON
    structure can vary by CLI version and exporter implementation, so we
    use a tolerant heuristic:
    - parse JSONL
    - walk objects recursively
    - sum any numeric fields whose keys look like input/output token counters

    Returns a normalized agent_backends-style usage dict when any tokens
    are found, else None.
    """
    if not otel_path.exists():
        return None
    try:
        raw = otel_path.read_text(encoding="utf-8")
    except Exception:
        return None
    if not raw.strip():
        return None

    key_in = re.compile(r"(?:^|[_\\.])(input|prompt)[_\\.]?tokens?$", re.IGNORECASE)
    key_out = re.compile(r"(?:^|[_\\.])(output|completion)[_\\.]?tokens?$", re.IGNORECASE)
    key_cache_read = re.compile(r"(?:^|[_\\.])cache(?:d)?[_\\.]?(?:read|input)?[_\\.]?tokens?$", re.IGNORECASE)

    totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    model_seen: Optional[str] = None

    def walk(x):
        nonlocal model_seen
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str):
                    lk = k.lower()
                    if model_seen is None and lk in {"model", "gen_ai.request.model", "gen_ai.model"} and isinstance(v, str):
                        model_seen = v
                    if isinstance(v, (int, float)):
                        if key_in.search(lk):
                            totals["input_tokens"] += int(v)
                        elif key_out.search(lk):
                            totals["output_tokens"] += int(v)
                        elif key_cache_read.search(lk):
                            totals["cache_read_tokens"] += int(v)
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        walk(obj)

    if totals["input_tokens"] == 0 and totals["output_tokens"] == 0 and totals["cache_read_tokens"] == 0:
        return None
    return {
        "input_tokens": totals["input_tokens"] or None,
        "output_tokens": totals["output_tokens"] or None,
        "cache_read_tokens": totals["cache_read_tokens"] or None,
        "cache_creation_tokens": None,
        "total_cost_usd": None,
        "num_turns": None,
        "model": model_seen,
    }


def _resolve_agent(name: str) -> AgentDef:
    agent = get_agent(name)
    if agent is None:
        available = ", ".join(list_agents())
        raise SystemExit(
            f"Unknown agent {name!r}. Available: {available}"
        )
    return agent


def _agent_binary(agent: AgentDef) -> Optional[str]:
    path = shutil.which(agent.bin_default)
    return path or None


def _print_available_models_for_agent(agent_name: str) -> None:
    """Best-effort: ask the agent's CLI to list available models.

    This is intentionally tolerant: some CLIs print to stderr or return
    non-zero while still producing useful output.
    """
    ag = get_agent(agent_name)
    if ag is None:
        print(_c(f"  ! agent {agent_name!r} is not registered", _RED))
        return
    bin_path = _agent_binary(ag)
    if not bin_path:
        print(_c(f"  ! agent binary not found on PATH: {ag.bin_default!r}", _RED))
        return

    label = (
        "Claude" if agent_name == "claude"
        else "Cursor Agent" if agent_name == "cursor"
        else "Copilot" if agent_name == "copilot"
        else "Codex" if agent_name == "codex"
        else agent_name
    )
    print(f"\n[{label}] available models (best-effort via CLI):")

    candidates: list[list[str]] = [
        [bin_path, "models"],
        [bin_path, "models", "--json"],
        [bin_path, "--list-models"],
        [bin_path, "list-models"],
        [bin_path, "model", "list"],
    ]
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
            print(combined)
            if cp.returncode != 0:
                print("  (note: CLI returned non-zero, but printed output above)")
            print()
            return
    print("  (could not retrieve a model list from the CLI)\n")


def _ask_model(label: str, default: str, *, agent_name: str) -> str:
    """Prompt for a model string; allow typing '?' to list models."""
    hint = "  (type '?' to list models)"
    for _attempt in range(50):
        try:
            raw = input(f"  {label} [{default}]:{hint} ").strip()
        except EOFError:
            print("  (EOF on stdin -- keeping default)")
            return default
        if raw == "":
            return default
        if raw in {"?", "models", "list", "list-models"}:
            _print_available_models_for_agent(agent_name)
            continue
        return raw
    print("  (too many attempts -- keeping default)")
    return default


def _resolve_and_check_agents(
    *,
    generator_source: str,
    generator_agent_name: str,
    solver_agent_name: str,
) -> tuple[Optional[AgentDef], AgentDef, list[tuple[str, AgentDef]]]:
    """Resolve agent names to ``AgentDef``s and verify their binaries are
    on PATH. Exits on error. Generator is skipped for non-agent sources
    (e.g. backtrader) where no LLM generator is used.

    Returns ``(generator_agent, solver_agent, agent_roles)``.
    """
    solver_agent = _resolve_agent(solver_agent_name)
    generator_agent: Optional[AgentDef]
    if generator_source == "backtrader_agent":
        # The LLM-authored Backtrader source needs a real generator agent;
        # the pure "backtrader" source runs without one.
        generator_agent = _resolve_agent(generator_agent_name)
        agent_roles = [("generator", generator_agent), ("solver", solver_agent)]
    else:
        generator_agent = None
        agent_roles = [("solver", solver_agent)]

    for role, ag in agent_roles:
        if _agent_binary(ag) is None:
            print(_c(
                f"ERROR: {role} agent {ag.name!r} binary {ag.bin_default!r} "
                f"not found on PATH.",
                _RED,
            ))
            sys.exit(1)
    return generator_agent, solver_agent, agent_roles


def _prompt_arena_config(
    *,
    rounds: int,
    generator_agent_name: str,
    solver_agent_name: str,
    generator_model: Optional[str],
    solver_model: Optional[str],
    generator_memory: str,
    solver_memory: str,
    generator_source: str,
    generator_os_sandbox: bool,
    solver_os_sandbox: bool,
) -> tuple[int, str, str, Optional[str], Optional[str], str, str, bool, bool]:
    """Interactively tweak arena-level knobs (round count + per-role
    agent / model / memory) in role-grouped order::

        Rounds
        Generator agent  -> Generator model  -> Generator memory
        Solver agent     -> Solver model     -> Solver memory

    Pressing Enter keeps the current default shown in brackets. Non-TTY:
    returns inputs unchanged.

    Generator-side prompts are skipped when ``generator_source`` is not
    an LLM-driven source (e.g. the pure ``backtrader`` source).

    Returns possibly-updated ``(rounds, generator_agent_name,
    solver_agent_name, generator_model, solver_model, generator_memory,
    solver_memory, generator_os_sandbox, solver_os_sandbox)``.
    """
    if not sys.stdin.isatty():
        return (rounds, generator_agent_name, solver_agent_name,
                generator_model, solver_model,
                generator_memory, solver_memory,
                generator_os_sandbox, solver_os_sandbox)

    avail = list_agents() or [solver_agent_name]
    memory_choices = list(MEMORY_MODES)

    print("\n[arena] interactive match config")
    print("  Press Enter on any prompt to keep the default shown in [brackets].")

    new_rounds = bt_source._ask_field("Rounds", rounds, cast=int)

    new_gen_name = generator_agent_name
    new_gen_model = generator_model
    new_gen_mem = generator_memory
    new_gen_sb = generator_os_sandbox
    if generator_source == "backtrader_agent":
        new_gen_name = bt_source._ask_field(
            "Generator agent", generator_agent_name, choices=avail,
        )
        new_gen_model = _ask_model(
            "Generator model",
            generator_model if generator_model is not None else DEFAULT_MODEL,
            agent_name=new_gen_name,
        )
        new_gen_mem = bt_source._ask_field(
            "Generator memory", generator_memory, choices=memory_choices,
        )
        # Interactive memory mode "sandbox" implies OS sandboxing.
        if new_gen_mem == "sandbox":
            new_gen_sb = True
    else:
        print(
            f"  (Generator agent/model/memory skipped -- "
            f"generator_source={generator_source!r} has no LLM generator)"
        )

    new_sol_name = bt_source._ask_field(
        "Solver agent", solver_agent_name, choices=avail,
    )
    new_sol_model = _ask_model(
        "Solver model",
        solver_model if solver_model is not None else DEFAULT_MODEL,
        agent_name=new_sol_name,
    )
    new_sol_mem = bt_source._ask_field(
        "Solver memory", solver_memory, choices=memory_choices,
    )
    new_sol_sb = solver_os_sandbox
    # Interactive memory mode "sandbox" implies OS sandboxing.
    if new_sol_mem == "sandbox":
        new_sol_sb = True

    return (new_rounds, new_gen_name, new_sol_name,
            new_gen_model, new_sol_model,
            new_gen_mem, new_sol_mem,
            new_gen_sb, new_sol_sb)


def _extras_for_role(agent: AgentDef,
                     role: str,
                     model: Optional[str],
                     user_extra: list[str]) -> list[str]:
    """Build the extra-args list for ``agent`` playing ``role``.

    - For the ``cursor`` agent, the legacy ``--model X`` flag is
      auto-forwarded so the existing arena CLI (``--model composer-2``)
      keeps working. Role-specific tweaks are added too
      (generator gets ``--mode ask`` for a cheap read-only run).
    - For any other agent, we pass through ``user_extra`` verbatim; callers
      can select a model by adding e.g. ``--solver-extra --model
      --solver-extra claude-4.6-sonnet``.
    """
    extras = list(user_extra)
    # Best-effort model forwarding: support the arena's --model / --*-model
    # flags without forcing users to add --*-extra --model themselves.
    # Cursor / Claude accept --model; for Copilot this is best-effort (its
    # CLI may ignore or reject it depending on version).
    if agent.name in ("cursor", "claude", "copilot", "codex"):
        if model and not any(x == "--model" for x in extras):
            extras = ["--model", model, *extras]
        if agent.name == "cursor" and role == "generator" and "--mode" not in extras:
            # Generator just emits text; a read-only ask-mode run is
            # cheaper and prevents accidental worktree side-effects.
            extras = ["--mode", "ask", *extras]
    return extras


def _invoke_agent(
    agent: AgentDef,
    prompt: str,
    *,
    cwd: Path,
    timeout: int,
    role: str,
    model: Optional[str],
    user_extra: list[str],
) -> AgentInvocation:
    """Run ``agent`` with ``prompt`` and auto-extract token usage."""
    cwd.mkdir(parents=True, exist_ok=True)
    bin_path = _agent_binary(agent)
    if not bin_path:
        return AgentInvocation(
            stdout="",
            stderr=f"[arena] binary {agent.bin_default!r} not found on PATH",
            returncode=127,
            usage=None,
            cmd_ok=False,
        )

    extras = _extras_for_role(agent, role, model, user_extra)
    env_backup: dict[str, str] | None = None
    # Best-effort: tell local agent wrappers to apply an OS sandbox for this role.
    # The restriction is implemented in agent_backends.maybe_wrap_macos_sandbox_exec.
    want_sandbox = (
        (role == "solver" and getattr(_core, "SOLVER_OS_SANDBOX", False))
        or (role == "generator" and getattr(_core, "GENERATOR_OS_SANDBOX", False))
    )
    if want_sandbox:
        env_backup = dict(os.environ)
        os.environ["ARENA_OS_SANDBOX"] = "1"
        os.environ["ARENA_OS_SANDBOX_DIR"] = str(cwd)
        # Protect this match's artifact tree so the role cannot inspect
        # sibling roles or prior/future round files. The allowed sandbox
        # dir is its own per-round role cwd.
        protect_dir = cwd.parents[1] if len(cwd.parents) > 1 else _core.ROOT
        os.environ["ARENA_OS_SANDBOX_PROTECT_DIR"] = str(protect_dir)
    # Copilot can optionally export token usage via OpenTelemetry file exporter.
    # Enable per-round export into the role's cwd, then parse it after the run.
    copilot_otel_path: Optional[Path] = None
    if agent.name == "copilot":
        if env_backup is None:
            env_backup = dict(os.environ)
        copilot_otel_path = cwd / "copilot_otel.jsonl"
        os.environ["COPILOT_OTEL_ENABLED"] = "true"
        os.environ["COPILOT_OTEL_EXPORTER_TYPE"] = "file"
        os.environ["COPILOT_OTEL_FILE_EXPORTER_PATH"] = str(copilot_otel_path)
    try:
        proc = agent.run_fn(bin_path, prompt, timeout, cwd, extras)
        stdout, stderr, rc = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr_body = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        stderr = f"{stderr_body}\n[arena] agent timed out after {timeout}s".strip()
        rc = 124
    except Exception as exc:  # pragma: no cover -- defensive
        stdout = ""
        stderr = f"[arena] agent invocation raised {type(exc).__name__}: {exc}"
        rc = 1
    finally:
        if env_backup is not None:
            os.environ.clear()
            os.environ.update(env_backup)

    usage = extract_usage(stdout, stderr)
    if usage is None and copilot_otel_path is not None:
        usage = _extract_copilot_usage_from_otel(copilot_otel_path)
    if usage is not None and model and not usage.get("model"):
        usage = dict(usage)
        usage["model"] = model
    return AgentInvocation(stdout=stdout, stderr=stderr, returncode=rc, usage=usage)

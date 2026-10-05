"""Path grammar for banked eval results — the single source of truth.

One run is keyed by `(model_name, version_id, signal_type)` and occupies one
directory, laid out identically on disk and inside the HF dataset repo:

    results/batched/{model_name}/{version_id}/{signal_type}/
    ├── result.json   # {"summary": ..., "records": ...} — what rb_*.json holds today
    ├── run.json      # provenance: arm, config, git sha, timestamp, raw sha256s
    └── raw/*.jsonl   # every raw completion, verbatim

`model_name` is the model *slot*, not the served checkpoint name: `tuned` is
banked as `v5` because that is what the arms, tables and paper call it.

Prompt style is a presentation lever, not a repair signal, so it does not get
its own path component — but two arms can differ *only* by it
(`verireason_default_diag` vs `verireason_diag`, the `verireason_native_lift`
contrast). Those are disambiguated in the model slot via ARM_MODEL_OVERRIDES,
and `check_no_collisions()` fails loudly if any pair still maps to one key.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
from dataclasses import dataclass

# Where the banked matrix lives, relative to an output root.
BATCHED_PREFIX = ("results", "batched")

# Model slot per config `model` / `claude_model` value. Anything absent falls
# through to `slug()`, so a new arm works without editing this table; entries
# here exist only where the banked name differs from the served name.
MODEL_ALIASES = {
    "tuned": "v5",
    "OriGen_Fix": "origen-fix",
}

# Arms whose (model, signal) pair is not unique on its own. Keyed by the arm
# name in configs/experiments.json.
ARM_MODEL_OVERRIDES = {
    # generic-prompt strict-parity floor, vs the native-template arm below it
    "verireason_default_diag": "verireason-default",
}

# Canonical order of the extra levers stacked on the tool diagnostic.
SIGNAL_ORDER = ("spec", "locate")

# Underscore survives: it is the separator *inside* a signal_type ('diag_spec'),
# so collapsing it would rewrite the documented grammar.
_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def slug(value: str) -> str:
    """Lowercase, filesystem- and URL-safe component."""
    return _SAFE.sub("-", str(value)).strip("-_").lower()


def signal_type(signal) -> str:
    """`[]` -> 'diag'; `['spec']` -> 'diag_spec'; `['spec','locate']` -> 'diag_spec_locate'.

    Every arm carries the tool diagnostic, so 'diag' is the floor rather than a
    lever; extras are appended in SIGNAL_ORDER so ['locate','spec'] and
    ['spec','locate'] cannot bank to two different directories.
    """
    extras = set(signal or ())
    unknown = extras - set(SIGNAL_ORDER)
    if unknown:
        raise ValueError(f"unknown repair signal(s): {sorted(unknown)}")
    return "_".join(["diag"] + [s for s in SIGNAL_ORDER if s in extras])


def slot_for(served: str) -> str:
    """Banked model slot for a served model id.

    The single consumer of MODEL_ALIASES. Build a RunKey through this rather
    than passing a served name straight in: `RunKey` only slugs, so
    `OriGen_Fix` would bank as `origen_fix` while the rest of the tree — which
    goes through the alias table — calls that slot `origen-fix`.
    """
    return slug(MODEL_ALIASES.get(served, served))


def model_name(exp: dict) -> str:
    """Model slot for one experiment dict from configs/experiments.json."""
    if exp.get("name") in ARM_MODEL_OVERRIDES:
        return ARM_MODEL_OVERRIDES[exp["name"]]
    raw = (exp.get("claude_model") if exp.get("backend") == "claude"
           else exp.get("model")) or "base"
    return slot_for(raw)


def default_version_id(repo_dir: str | None = None) -> str:
    """`YYYYMMDD-<git short sha>` — sortable, and traceable to the harness state.

    Override with $RTLREPAIR_RUN_VERSION when banking outputs produced elsewhere
    (a collaborator's run, a hosted API) whose provenance is not this checkout.
    """
    env = os.environ.get("RTLREPAIR_RUN_VERSION")
    if env:
        return slug(env)
    day = _dt.date.today().strftime("%Y%m%d")
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir or os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        sha = ""
    return f"{day}-{sha}" if sha else f"{day}-nogit"


@dataclass(frozen=True)
class RunKey:
    """Identity of one banked run. Immutable, and the only way to build a path."""

    model: str
    version: str
    signal: str

    def __post_init__(self):
        for field in ("model", "version", "signal"):
            value = getattr(self, field)
            if not value:
                raise ValueError(f"RunKey.{field} must be non-empty")
            if value != slug(value):
                object.__setattr__(self, field, slug(value))

    @classmethod
    def from_experiment(cls, exp: dict, version: str | None = None) -> "RunKey":
        """Build the key for one configs/experiments.json entry."""
        return cls(model=model_name(exp),
                   version=version or default_version_id(),
                   signal=signal_type(exp.get("signal", [])))

    @property
    def parts(self) -> tuple[str, ...]:
        return BATCHED_PREFIX + (self.model, self.version, self.signal)

    def path(self, root: str = "") -> str:
        """Directory for this run, under `root` (or repo-relative if root='')."""
        return os.path.join(root, *self.parts) if root else os.path.join(*self.parts)

    def as_posix(self) -> str:
        """Repo-relative path with forward slashes — the HF path_in_repo."""
        return "/".join(self.parts)

    def __str__(self) -> str:
        return f"{self.model}/{self.version}/{self.signal}"


def check_no_collisions(experiments, version: str = "v") -> None:
    """Raise if two arms bank to one directory.

    The path grammar drops prompt style, so this is the guard that keeps a new
    arm from silently overwriting an existing one. Called by run_experiments and
    covered by tests/test_storage_layout.py.
    """
    seen: dict[str, str] = {}
    for exp in experiments:
        if exp.get("name", "").startswith("_"):
            continue
        key = str(RunKey.from_experiment(exp, version=version))
        if key in seen:
            raise ValueError(
                f"arms {seen[key]!r} and {exp['name']!r} both bank to {key!r}. "
                f"They differ by a lever the path grammar drops (prompt style). "
                f"Give one of them an entry in storage.layout.ARM_MODEL_OVERRIDES."
            )
        seen[key] = exp.get("name", "<unnamed>")

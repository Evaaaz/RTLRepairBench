#!/usr/bin/env python3
"""How much the semantic split separates systems that the lint split cannot.

The claim in Section 2 is that a syntax-scored benchmark has neither the headroom nor the
resolution to rank repair systems, and that the lint ranking is not a proxy for the semantic
one. Both halves are computed here from the same recovery table the paper prints, so the
numbers in the prose and the numbers in Table 1 cannot drift apart.

Two things are worth saying about the rank correlation. It is Spearman on seven points, so it
is descriptive and we report it as such. And it is dominated by the two frontier models, which
sit at 100% on lint and therefore carry no rank information at all: excluding them is not a
robustness check but the honest version of the statistic.

Writes artifacts/public/capability/split_discrimination.json.
"""
import json
import re
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from project_paths import project_root  # noqa: E402

ROOT = project_root()
TEX = ROOT / "paper/paper.tex"
OUT = ROOT / "artifacts/public/capability/split_discrimination.json"
SATURATED = 99.0  # a lint cell at or above this carries no rank information


def spearman(xs, ys):
    n = len(xs)
    rank_x = {v: i for i, v in enumerate(sorted(xs))}
    rank_y = {v: i for i, v in enumerate(sorted(ys))}
    d2 = sum((rank_x[a] - rank_y[b]) ** 2 for a, b in zip(xs, ys))
    return round(1 - 6 * d2 / (n * (n * n - 1)), 2)


def no_signal_rows(tex: str) -> dict[str, tuple[float, float]]:
    """The tool-diagnostic row of every model in the recovery table.

    Each model's first row carries its name; continuation rows start with '&'. Only the
    no-signal arm is comparable across the two splits, so the spec rows are skipped.
    """
    block = re.search(r"\\label\{tab:recovery\}.*?\\end\{tabular\}", tex, re.S)
    if not block:
        raise SystemExit("recovery table not found in paper.tex")
    out = {}
    for line in block.group(0).split("\\\\"):
        line = re.sub(r"\\c?midrule(\(l\)\{[^}]*\})?", "", line).strip()
        if not line or line.startswith("&") or "tool diagnostic" not in line:
            continue
        if "$+$ spec" in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 4:
            continue
        name = re.sub(r"\\[a-zA-Z]+|[{}$^\\]|\(.*?\)", "", cells[0]).strip()
        pcts = []
        for cell in cells[2:4]:
            m = re.search(r"(\d+(?:\.\d+)?)\\%", cell)
            pcts.append(float(m.group(1)) if m else None)
        if name and all(p is not None for p in pcts):
            out[name] = (pcts[0], pcts[1])
    return out


def main() -> int:
    rows = no_signal_rows(TEX.read_text(encoding="utf-8"))
    if len(rows) < 5:
        raise SystemExit(f"parsed only {len(rows)} models from the recovery table")
    lint = {k: v[0] for k, v in rows.items()}
    sem = {k: v[1] for k, v in rows.items()}

    clustered = [k for k in lint if lint[k] >= 85.0]
    unsaturated = [k for k in lint if lint[k] < SATURATED]

    report = {
        "generated_by": "code/scripts/ws_split_discrimination.py",
        "source": "paper/paper.tex, the tool-diagnostic rows of the recovery table",
        "n_models": len(rows),
        "per_model": {k: {"lint_pct": lint[k], "semantic_pct": sem[k]} for k in sorted(rows)},
        "clustered_on_lint": {
            "models": sorted(clustered),
            "n": len(clustered),
            "lint_span_pp": round(max(lint[k] for k in clustered) - min(lint[k] for k in clustered), 1),
            "lint_range": [min(lint[k] for k in clustered), max(lint[k] for k in clustered)],
            "semantic_span_pp": round(max(sem[k] for k in clustered) - min(sem[k] for k in clustered), 1),
            "semantic_range": [min(sem[k] for k in clustered), max(sem[k] for k in clustered)],
        },
        "rank_correlation": {
            "all_models": spearman([lint[k] for k in rows], [sem[k] for k in rows]),
            "excluding_saturated_lint": spearman(
                [lint[k] for k in unsaturated], [sem[k] for k in unsaturated]),
            "n_excluding_saturated": len(unsaturated),
            "saturated_lint_models": sorted(k for k in lint if lint[k] >= SATURATED),
            "note": "descriptive Spearman on few points; not a test",
        },
        "rank_by_lint": sorted(rows, key=lambda k: -lint[k]),
        "rank_by_semantic": sorted(rows, key=lambda k: -sem[k]),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    c = report["clustered_on_lint"]
    r = report["rank_correlation"]
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {c['n']} of {report['n_models']} models within {c['lint_span_pp']}pp on lint "
          f"({c['lint_range'][0]}-{c['lint_range'][1]}%)")
    print(f"  the same {c['n']} span {c['semantic_span_pp']}pp on semantic "
          f"({c['semantic_range'][0]}-{c['semantic_range'][1]}%)")
    print(f"  Spearman(lint, semantic) = {r['all_models']:+.2f} over all, "
          f"{r['excluding_saturated_lint']:+.2f} over the {r['n_excluding_saturated']} "
          f"with an unsaturated lint score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

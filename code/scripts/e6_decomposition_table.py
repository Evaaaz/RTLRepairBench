#!/usr/bin/env python3
"""Generate the one-row-per-model decomposition table.

the second author's Table 4 format: model, no-signal baseline, then What / Where / Counterexample
as a point estimate with a 95% cluster-bootstrap CI.

Every cell is now filled. The Holm claim the caption used to carry is gone: nothing
here computed it -- this script formats JSON, and no correction is applied anywhere on
the path -- so rather than implement one that would widen the internally prespecified row's
published intervals, the caption states what is true.

Sources, all committed:
  artifacts/public/capability/capability_curve.json   What/Where for the curve models
  artifacts/public/capability/openweight_tier_f.json  Tier F rates for the open weights
  artifacts/public/capability/openweight_what_effects.json  their baseline, What and Where
                                                      (Where and baseline merged in by
                                                      e7_openweight_decomposition.py)
  artifacts/public/capability/counterexample_contrast.json  the counterexample contrast

Every input is a public artifact inside the bundle, so a fresh export rebuilds this
table without the run tree.

Writes workshop/lint-vs-semantic/paper/tab_decomposition.tex.
"""
import json, os, sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from project_paths import output_root, project_root  # noqa: E402

SOURCE_ROOT = str(project_root())
ROOT = str(output_root().parent)
OUT = f"{SOURCE_ROOT}/paper/tab_decomposition.tex"


def cell(pt, ci):
    if pt is None:
        return "---"
    lo, hi = ci
    return f"${pt:+.1f}$ {{\\tiny $[{lo:+.1f}, {hi:+.1f}]$}}"


def main():
    cap = {r["model"]: r for r in json.load(
        open(f"{SOURCE_ROOT}/artifacts/public/capability/capability_curve.json"))}
    ow = json.load(open(f"{SOURCE_ROOT}/artifacts/public/capability/openweight_what_effects.json"))
    owf = json.load(open(f"{SOURCE_ROOT}/artifacts/public/capability/openweight_tier_f.json"))
    ce = json.load(open(
        f"{SOURCE_ROOT}/artifacts/public/capability/counterexample_contrast.json"))

    def ce_cell(tag):
        d = ce.get(tag)
        if not d:
            return "---"
        # the counterexample contrast has no cluster bootstrap; report the paired
        # risk difference and its exact McNemar p, and say so in the caption
        return f"${d['risk_diff_pp']:+.1f}$ {{\\tiny $n{{=}}{d['n_paired']}$}}"

    # Claude Opus 4.8 was dropped when the table standardized on the formal tier; it is back
    # because its arms were re-run and formally scored, on the same estimand as every other row.
    c48 = json.load(open(
        f"{SOURCE_ROOT}/artifacts/public/capability/second_anchor_claude_opus_4_8_tierf.json"))
    sm = c48["seed_macro"]

    rows = [
        # label, dagger, baseline, What, Where, Counterexample
        ("gpt-5.5 (frozen primary)", "$^\\dagger$", 84.7,
         cell(5.1, (-4.6, 15.4)), cell(-0.3, (-5.9, 4.8)), "$-12.5$ {\\tiny $n{=}5$}"),
        ("Claude Opus 4.8", "$^\\ddagger$", c48["where"]["baseline_pct"],
         cell(sm["what"]["pp"], sm["what"]["ci"]),
         cell(sm["where"]["pp"], sm["where"]["ci"]), ce_cell("claude48")),
        ("gpt-4.1", "$^\\ddagger$", cap["azure/openai/gpt-4.1"]["baseline_pct"],
         cell(cap["azure/openai/gpt-4.1"]["what_pp"], cap["azure/openai/gpt-4.1"]["what_ci"]),
         cell(cap["azure/openai/gpt-4.1"]["where_pp"], cap["azure/openai/gpt-4.1"]["where_ci"]),
         ce_cell("gpt41")),
        ("gpt-4o-mini", "$^\\ddagger$", cap["openai/openai/gpt-4o-mini"]["baseline_pct"],
         cell(cap["openai/openai/gpt-4o-mini"]["what_pp"], cap["openai/openai/gpt-4o-mini"]["what_ci"]),
         cell(cap["openai/openai/gpt-4o-mini"]["where_pp"], cap["openai/openai/gpt-4o-mini"]["where_ci"]),
         ce_cell("gpt4omini")),
        ("llama-3.3-70b", "$^\\ddagger$", cap["nvcf/meta/llama-3.3-70b-instruct"]["baseline_pct"],
         cell(cap["nvcf/meta/llama-3.3-70b-instruct"]["what_pp"],
              cap["nvcf/meta/llama-3.3-70b-instruct"]["what_ci"]),
         cell(cap["nvcf/meta/llama-3.3-70b-instruct"]["where_pp"],
              cap["nvcf/meta/llama-3.3-70b-instruct"]["where_ci"]),
         ce_cell("llama")),
        ("OriGen\\_Fix", "$^\\ddagger$", cap["OriGen_Fix (RTL-specific 7B)"]["baseline_pct"],
         cell(cap["OriGen_Fix (RTL-specific 7B)"]["what_pp"],
              cap["OriGen_Fix (RTL-specific 7B)"]["what_ci"]),
         cell(cap["OriGen_Fix (RTL-specific 7B)"]["where_pp"],
              cap["OriGen_Fix (RTL-specific 7B)"]["where_ci"]), ce_cell("origen")),
    ]
    for key, label in (("v5", "qwen2.5-coder-7b-repair-sft"), ("base", "Qwen2.5-Coder-7B (base)"),
                       ("vrqwen", "VeriReason-Qwen2.5-7B (GRPO)"),
                       ("verireason", "VeriReason-CodeLlama-7B (GRPO)")):
        # The baseline is carried in two public artifacts now; they must agree, and a
        # table is the wrong place to discover that they do not.
        base_pct = owf["arms"][f"{key}__diag"]["tier_f_pct"]
        derived = ow[key].get("baseline_pct")
        if derived is not None and abs(derived - base_pct) > 0.05:
            raise SystemExit(f"{key}: baseline {derived} in openweight_what_effects.json "
                             f"disagrees with {base_pct} in openweight_tier_f.json")
        where = ("---" if "where_pp" not in ow[key]
                 else cell(ow[key]["where_pp"], ow[key]["where_ci"]))
        rows.append((label, "$^\\ddagger$", base_pct,
                     cell(ow[key]["what_pp"], ow[key]["ci"]), where, ce_cell(key)))

    # the caption promises rows ordered by baseline, so sort rather than trusting the
    # order they were appended in
    rows.sort(key=lambda r: -r[2])
    body = "\n".join(
        f"{lab}{dag} & {b:.1f} & {w} & {wh} & {c} \\\\" for lab, dag, b, w, wh, c in rows)
    tex = f"""% GENERATED by scripts/e6_decomposition_table.py -- do not edit by hand.
\\begin{{table}}[!h]\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{3.5pt}}
\\caption{{\\textbf{{Exploratory decomposition by no-signal recovery.}} Cells are seed-macro paired risk
differences in percentage points with 95\\% cluster-bootstrap CIs; \\CE{{}} reports its paired $n$
instead. $^\\dagger$~is the prespecified primary, broken out in Table~\\ref{{tab:primary}}.
$^\\ddagger$~rows are exploratory and uncorrected, and their prompts, cohorts and scoring tiers differ.}}
\\label{{tab:decomposition}}
\\begin{{tabular}}{{lrccc}}\\toprule
 & no-signal & \\multicolumn{{3}}{{c}}{{effect vs.\\ the no-signal arm (pp, [95\\% CI])}} \\\\
\\cmidrule(l){{3-5}}
model & baseline & \\What{{}} (spec) & \\Where{{}} (oracle line) & \\CE{{}} \\\\\\midrule
{body}\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    open(OUT, "w").write(tex)
    print(f"wrote {os.path.relpath(OUT, SOURCE_ROOT)} with {len(rows)} model rows")
    for lab, _, b, w, wh, c in rows:
        print(f"  {lab:<32}{b:>7.1f}  What {w[:14]:<16} Where {wh[:14]:<16} CE {c[:12]}")


if __name__ == "__main__":
    main()

# Third-party notices and provenance boundary

The repository-level Apache-2.0 license covers only material authored for this
project. It does not override upstream licenses, dataset provenance, or hosted
model-provider terms.

## MG-Verilog

- Upstream: <https://huggingface.co/datasets/GaTech-EIC/MG-Verilog>
- Code repository: <https://github.com/GATECH-EIC/mg-verilog>
- Located local dataset snapshot identifier:
  `82944845640b41d190e33e1c3b33f1ef5a868359`
- Declared dataset-card license: MIT
- Use here: source seeds for the released 5,775-row training file
  (`data/rtlrepairdataset_train.jsonl`) and the historical 5,784-row output

The dataset card says that source code was gathered from multiple online
sources. A package-level MIT label is therefore not treated as proof of
per-sample provenance or redistribution permission. The training file ships
with the MG-Verilog MIT notice reproduced in `NOTICE`; per the SFT mixture's
dataset card, its rows derive only from MG-Verilog seeds.

## VerilogEval

- Upstream: <https://github.com/NVlabs/verilog-eval>
- License file: <https://github.com/NVlabs/verilog-eval/blob/main/LICENSE>
- Historical project record: abbreviated revision `c498220`
- Declared repository license: MIT
- Use here: held-out specifications, reference RTL, derived mutations, and
  mined-failure tasks

Only an abbreviated historical revision was recorded in the surviving project
notes. The release does not claim that this abbreviation is a complete
immutable source manifest.

## RTLLM

- Upstream: <https://github.com/hkust-zhiyao/RTLLM>
- License file: <https://github.com/hkust-zhiyao/RTLLM/blob/main/LICENSE>
- Historical project record: abbreviated revision `41b2689`
- Declared repository license: MIT
- Use here: historical held-out/decontamination inputs and larger-design probes

As with VerilogEval, the surviving record contains an abbreviated rather than
full commit identifier. The current poster does not report the legacy RTLLM
probe as a main result.

## Hosted-model outputs

The mined-failure records and some aggregate evaluations originate from
outputs produced by hosted or third-party models. Provider terms and any
applicable output restrictions remain in force. This repository does not grant
additional rights in those outputs. Raw provider responses and reasoning
traces are excluded from this package.

## Release decision

This public camera-ready release, made by the authors in October 2026,
contains project-authored code, documentation, sanitized aggregate evidence, the
training file, and the allowlisted benchmark snapshot. It does not represent a
blanket relicensing of upstream HDL or model outputs. Required upstream notices
are in `NOTICE`; items still open are listed in `docs/RELEASE_CHECKLIST.md`.

# Camera-ready paper source

This directory contains the source and PDF of the camera-ready paper:

> Yeyin (Eva) Zhu and Rishabh Ranawat. *What Do RTL Repair Benchmarks Actually
> Measure?* NeurIPS 2026 Workshop on AI for Chip Design.

`paper.tex` is the source of truth and `paper.pdf` is its camera-ready build.
The anonymous submission, with its NeurIPS checklist, is the version reviewed on
OpenReview. The two poster figures and the official style file are colocated
here. Other small generated TeX fragments are retained only as Level-A
reproduction targets; they are not alternate manuscripts.

## Build

From the repository root, use the release build target:

```bash
make paper
```

The target rebuilds `fig_repairbench_pipeline.pdf` and
`fig_oracle_tiers_poster.pdf`, creates previews under `build/paper/assets/`, and
writes the paper to `build/paper/paper.pdf`. It then enforces a 3--4-page body
and rejects unresolved references, review
comments, and TODO markers. Ordinary builds modify only `build/`; release
maintainers promote the single PDF snapshot explicitly and refresh the artifact
lock.


## Results provenance

The paper source is not itself evidence that every reported number is
reproducible. The machine-readable mapping from claims to scripts, inputs, and
frozen outputs is [`../claims.json`](../claims.json); the interpretation and
known gaps are documented in
[`../docs/RESULTS_AND_PROVENANCE.md`](../docs/RESULTS_AND_PROVENANCE.md).

Long-paper and supplementary drafts are not part of this export or the current
reproduction path.

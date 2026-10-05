# VeriReason on Modal — serving + RepairBench eval runbook

Self-contained Modal deployment of
[`Nellyw888/VeriReason-codeLlama-7b-RTLCoder-Verilog-GRPO-reasoning-tb`](https://huggingface.co/Nellyw888/VeriReason-codeLlama-7b-RTLCoder-Verilog-GRPO-reasoning-tb)
(CodeLlama-7B, GRPO-trained Verilog reasoning model) as an OpenAI-compatible
vLLM endpoint, evaluated with the released `code/benchmarks/` RepairBench
harness exactly as the base/v5 arms were.

This package imports **nothing** from the rest of `backend/` — only `modal`.

```
code/benchmarks/backend/modal_inference/
├── modal_verireason.py             # the Modal app (deploy this file)
├── chat_template_verireason.jinja  # raw-completion chat template (see notes in the .py)
└── README.md                       # this runbook
```

## One-time setup (USER)

```bash
pip install modal
modal setup                                   # browser auth
modal secret create verireason-api-key VLLM_API_KEY=$(openssl rand -hex 24)
modal secret create qwen-v5-adapter QWEN_V5_ADAPTER_REPO=<authorized-namespace/adapter>
```

Also recommended: set a workspace **spend limit** (~$20) in the Modal dashboard.
No HF token is needed — the model is public and ungated. (License note: the
card declares none; base CodeLlama carries the Llama 2 Community License.)

## Deploy + health check

```bash
modal deploy code/benchmarks/backend/modal_inference/modal_verireason.py
# prints: https://<workspace>--verireason-vllm-serve.modal.run

export VERIREASON_LLM_URL="https://<that-url>/v1"
export RTLREPAIR_LLM_API_KEY="<the VLLM_API_KEY value>"

# first request cold-starts: ~10–20 min on first ever boot (27 GB fp32 download
# into the verireason-hf-cache volume), ~2–4 min on later cold starts
curl -sS -H "Authorization: Bearer $RTLREPAIR_LLM_API_KEY" \
     "$VERIREASON_LLM_URL/models"
# expect: {"data":[{"id":"verireason",...}]}
```

Then one manual completion to verify the template took effect (vLLM's logged
prompt must show `<s>` and NO `[INST]`) and that fp16-cast output is sane
(the checkpoint is fp32 on disk):

```bash
curl -sS "$VERIREASON_LLM_URL/chat/completions" \
  -H "Authorization: Bearer $RTLREPAIR_LLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"verireason","messages":[{"role":"user","content":"Write a verilog module for a 2-to-1 mux."}],"max_tokens":512}'
# expect a reasoning section followed by an answer with a ```verilog fence
```

## Run the eval

Run these commands from the repository root. Prerequisites inside the worktree
(once):

```bash
python3 code/benchmarks/make_seeds.py --out generated/repairbench/seeds
# restore prior base/v5 result JSONs into results/ (untracked in the main
# checkout) so analyze_repairbench.py has records to pair the McNemar contrasts
# against — otherwise budget a second (Qwen+LoRA) deployment to regenerate them.
```

Smoke (items 1–50 are all lint; add a semantic slice for real signal):

```bash
RTLREPAIR_LLM_URL=$VERIREASON_LLM_URL \
RTLREPAIR_SEEDS=generated/repairbench/seeds \
python3 code/benchmarks/repairbench_eval.py --model verireason --prompt-style verireason \
  --anon --temp 0.0 --limit 50

# 20-item semantic smoke slice:
RTLREPAIR_SEEDS=generated/repairbench/seeds python - <<'EOF'
import json, itertools
items = [json.loads(l) for l in open("data/repairbench_heldout.jsonl")]
sem = [it for it in items if it["bucket"] == "semantic"][:20]
with open("/tmp/rb_semantic_smoke.jsonl", "w") as f:
    f.writelines(json.dumps(it) + "\n" for it in sem)
EOF
RTLREPAIR_LLM_URL=$VERIREASON_LLM_URL \
python3 code/benchmarks/repairbench_eval.py --model verireason --prompt-style verireason \
  --anon --temp 0.0 --bench /tmp/rb_semantic_smoke.jsonl
```

**Validity gate:** the evaluator requires an exact response roster and
`by_bucket.*.used_llm_rate == 1.0` by default. Endpoint or authentication
failures produce a nonzero exit and only an `.incomplete` diagnostic artifact.
`--allow-incomplete-run` relaxes this gate for debugging only; it never replaces
the canonical output.

Full arms (three, via the experiment matrix):

```bash
VERIREASON_LLM_URL=... RTLREPAIR_LLM_API_KEY=... \
python3 code/benchmarks/run_experiments.py --only verireason_default_diag verireason_diag verireason_diag_spec
RTLREPAIR_OUT=generated/repairbench python3 code/benchmarks/analyze_repairbench.py
```

| arm | prompt | max_tokens | role |
|---|---|---|---|
| `verireason_default_diag` | generic (`default`) | 1024 | strict-parity floor vs `base_diag`; expect truncation — label as floor |
| `verireason_diag` | native think/answer | 2048 (`VERIREASON_MAX_TOKENS`) | headline diagnostic arm |
| `verireason_diag_spec` | native + spec | 2048 | natural comparison (training always had a task description) |

Truncation audit for the native arms: set `VERIREASON_DUMP=/tmp/vr_raw.jsonl`
to append every raw completion; count suspected truncations with
`grep -c '"truncated_guess": true' /tmp/vr_raw.jsonl`.

## Caveats to report with the numbers

* VeriReason's native contract is **spec→RTL generation, not repair** — repair
  is off-distribution under *any* prompt (a bigger caveat than OriGen's, whose
  contract *is* repair). Both buckets are lower bounds; compare the native arms
  against `base_diag`/`base_diag_spec` per the OriGen precedent.
* Served as raw-completion via a custom chat template (the stock CodeLlama
  `[INST]` template was never seen in GRPO training).
* Greedy `--temp 0.0` for paired parity with the other arms; the card
  recommends 0.2/0.95 — if truncation/repetition rates are high, add a temp-0.2
  sensitivity run alongside (not instead).

## Cost (Modal, July 2026 rates; L40S $1.95/hr)

Scale-to-zero + `scaledown_window=15min`; **never** set `min_containers` —
an always-on L40S is ~$1,400/mo. Expected all-in for 3 arms + smoke + dev
iteration: **~$15–25**, inside Modal Starter's $30/mo free credits (tighter if
arms are re-run or the baseline must be re-served).

## Teardown

```bash
modal app stop verireason-vllm        # containers stop; volume persists (~27 GB, free tier)
modal volume delete verireason-hf-cache   # optional full cleanup
```

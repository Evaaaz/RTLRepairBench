"""Self-contained Modal app serving DeepSeek-Coder-7B-Instruct-v1.5 + the OriGen_Fix LoRA.

Deploy (requires `modal setup` and the shared secret; see README.md):

    modal deploy workshop/lint-vs-semantic/code/benchmarks/backend/modal_inference/modal_origen.py

Replaces the local Apple-MPS transformers+peft runner (scripts/origen_run.py, venv
/tmp/origen_env, now gone) that produced the banked 2x2 exploration. The eval
connects with zero harness changes:

    RTLREPAIR_LLM_URL=https://<workspace>--origen-vllm-serve.modal.run/v1
    RTLREPAIR_LLM_API_KEY=<the VLLM_API_KEY value>
    python repairbench_eval.py --model OriGen_Fix --prompt-style origen \\
      --anon --spec --temp 0.0 --workers 8

(configs/experiments.json calls the same URL ORIGEN_LLM_URL for the
`origen_diag_spec` arm; run_experiments.py maps endpoint_env -> RTLREPAIR_LLM_URL.)

The adapter -- not the base -- is the eval arm, so it is the LoRA module that
carries the served id: `llm_client._served_model()` passes `--model OriGen_Fix`
through unchanged, and vLLM routes an `OriGen_Fix` request to the adapter.

PARITY WITH scripts/origen_run.py -- this file reproduces the banked local run's
inference contract deliberately; do not "improve" these without regenerating the
banked 2x2 you would then be comparing against:
* --dtype float16. The base checkpoint is bfloat16 on disk (config.json
  torch_dtype), but the OriGen card is explicit that the LoRA was trained in
  fp16 and that fp16 outperforms bf16 for it -- and origen_run.py loaded the
  base with torch_dtype=torch.float16. vLLM applies LoRA deltas in the base
  compute dtype, so fp16 here is what puts the adapter in its trained numerics.
  This is the same reasoning as modal_verireason.py's --dtype float16, for a
  different reason (there: an fp32 checkpoint; here: fp16-trained LoRA).
* --max-lora-rank 32 (adapter_config.json declares r=32, lora_alpha=32; note
  this is NOT the 16 that modal_qwen_v5.py's v5 adapter uses).
* NO --chat-template. Unlike VeriReason-CodeLlama, no override is needed: the
  stock deepseek-coder-instruct-v1.5 tokenizer template already wraps a user
  turn in `### Instruction:` and primes `### Response:` -- which IS
  OriGen's native template framing (prompts.origen_prompt builds exactly that
  instruction body and drops the `### Response:{header}` priming precisely
  because the template supplies it). The template also emits a supplied system
  message verbatim ahead of the instruction, so prompts.py's "You are a
  professional Verilog designer." lands where it should. origen_run.py likewise
  used AutoTokenizer.from_pretrained(BASE_MODEL).apply_chat_template, i.e. the
  BASE tokenizer's template -- the adapter repo ships its own tokenizer files
  but they were not used then and are not used now (vLLM's `name=path`
  --lora-modules form does not swap the tokenizer).
* NO --max-model-len: vLLM defaults to the config's max_position_embeddings,
  which for this checkpoint is 4096 (rope_scaling null -- it cannot honestly be
  raised). That is the real ceiling for prompt + 1024 completion tokens; see the
  overflow note in README.md.
* --enforce-eager: without it vLLM startup can hang in CUDA-graph capture
  (see the setup notes in serving/serve_vllm.sh); also matches every other arm.

Other choices carried over from modal_qwen_v5.py / modal_verireason.py:
* @modal.web_server (NOT @app.server): web_server QUEUES requests through cold
  starts; Servers 503 immediately. A 503 mid-run is scored as no-fix by the
  harness (repairbench_eval.eval_one catches the exception from
  generate_strict and records used_llm=False), silently deflating the arm.
* Auth is vLLM's own --api-key bearer check (NOT Modal proxy auth, which
  requires Modal-Key/Modal-Secret headers the harness's OpenAI client cannot
  send). Shared `verireason-api-key` secret, same as the other three apps.
* VLLM_USE_FLASHINFER_SAMPLER=0: debian_slim has no CUDA toolkit, so
  FlashInfer's sampler JIT-builds with nvcc during KV-cache profiling and
  hard-fails engine startup.
"""
import os
import subprocess

import modal

BASE_MODEL = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
ADAPTER_REPO = "henryen/OriGen_Fix"
# The eval driver is invoked with `--model OriGen_Fix`, so the LoRA module name
# must match that string exactly.
SERVED_NAME = "OriGen_Fix"
# The base is reachable too (handy for an adapter-vs-base sanity poke), but NOT
# under the id `base` -- that id means Qwen2.5-Coder-7B everywhere else in this
# tree (modal_qwen_v5.py, results/rb_base*.json) and reusing it would silently
# mislabel results.
BASE_SERVED_NAME = "deepseek-base"
MAX_LORA_RANK = 32  # adapter_config.json: r=32
PORT = 8000
MINUTES = 60  # seconds

# Override at deploy time with e.g. ORIGEN_GPU=A100-40GB when L40S has no
# capacity. L40S (48 GB) holds the fp16 7B (~13.8 GB) plus the rank-32 adapter
# with generous KV headroom at this model's 4096 context.
GPU = os.environ.get("ORIGEN_GPU", "L40S")

hf_cache = modal.Volume.from_name("origen-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("vllm>=0.23,<0.24", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache/hf",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)

app = modal.App("origen-vllm")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache/hf": hf_cache},
    secrets=[modal.Secret.from_name("verireason-api-key")],
    timeout=24 * 60 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=8)  # the eval runs --workers 8
@modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
def serve():
    # Resolve the adapter to a local path before launching. Passing the bare repo
    # id would make vLLM fetch it mid-startup, where a hub failure surfaces as an
    # opaque engine crash instead of a clear error here.
    from huggingface_hub import snapshot_download

    # The repo ships its raw training checkpoint alongside the adapter
    # (optimizer.pt is 600 MB, more than the 300 MB adapter itself). vLLM reads
    # none of it; skipping it halves the first cold start. Everything vLLM or
    # PEFT could plausibly open -- adapter_config.json, adapter_model.safetensors,
    # the tokenizer files -- is still fetched.
    adapter_path = snapshot_download(
        ADAPTER_REPO,
        ignore_patterns=[
            "optimizer.pt",
            "rng_state.pth",
            "scheduler.pt",
            "training_args.bin",
            "trainer_state.json",
        ],
    )
    print(f"[serve] adapter resolved to {adapter_path}", flush=True)

    subprocess.Popen(
        [
            "vllm", "serve", BASE_MODEL,
            "--served-model-name", BASE_SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--dtype", "float16",
            "--gpu-memory-utilization", "0.90",
            "--enforce-eager",
            "--enable-lora",
            "--lora-modules", f"{SERVED_NAME}={adapter_path}",
            "--max-lora-rank", str(MAX_LORA_RANK),
            "--api-key", os.environ["VLLM_API_KEY"],
        ]
    )

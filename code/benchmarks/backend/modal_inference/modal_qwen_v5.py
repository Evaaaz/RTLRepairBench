"""Self-contained Modal app serving Qwen2.5-Coder-7B (`base`) + the v5 LoRA (`tuned`).

Deploy (requires `modal setup` and the shared secret; see README.md):

    modal deploy code/benchmarks/backend/modal_inference/modal_qwen_v5.py

Replaces the retired private pod that originally served these two arms. The
eval connects with zero harness changes:

    RTLREPAIR_LLM_URL=https://<workspace>--qwen-v5-vllm-serve.modal.run/v1
    RTLREPAIR_LLM_API_KEY=<the VLLM_API_KEY value>
    python run_experiments.py --only base_diag base_diag_spec v5_diag v5_diag_spec

`llm_client._served_model()` passes `base` and `tuned` through unchanged, so the
OpenAI `model=` field selects the arm exactly as it did on the pod.

PARITY WITH serving/serve_vllm.sh -- this file reproduces that script's launch
args deliberately; do not "improve" them without regenerating every arm:
* --served-model-name base, --lora-modules tuned=<adapter>, --max-lora-rank 16
  (adapter_config.json declares r=16), --gpu-memory-utilization 0.90.
* NO --chat-template. The pod passed none, so both arms rendered with the BASE
  Qwen tokenizer template -- including `tuned`, even though the adapter repo
  ships its own chat_template.jinja. Mounting that template here would change
  the v5 prompt relative to every previously collected v5 number.
* NO --dtype and NO --max-model-len: the pod let vLLM default both (bf16,
  32768 for this checkpoint).
* --enforce-eager: without it vLLM startup can hang in CUDA-graph capture
  (see the setup notes in serve_vllm.sh).

Other choices carried over from modal_verireason.py:
* @modal.web_server (NOT @app.server): web_server QUEUES requests through cold
  starts; Servers 503 immediately. A 503 mid-run is scored as no-fix by the
  harness (used_llm=False), silently deflating the arm.
* VLLM_USE_FLASHINFER_SAMPLER=0: debian_slim has no CUDA toolkit, so
  FlashInfer's sampler JIT-builds with nvcc during KV-cache profiling and
  hard-fails engine startup.
"""
import os
import subprocess

import modal

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_REPO_ENV = "QWEN_V5_ADAPTER_REPO"
MAX_LORA_RANK = 16  # adapter_config.json: r=16
PORT = 8000
MINUTES = 60  # seconds

# Override at deploy time with e.g. QWEN_GPU=A100-40GB when L40S has no capacity.
GPU = os.environ.get("QWEN_GPU", "L40S")

hf_cache = modal.Volume.from_name("qwen-v5-hf-cache", create_if_missing=True)

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

app = modal.App("qwen-v5-vllm")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache/hf": hf_cache},
    secrets=[
        modal.Secret.from_name("verireason-api-key"),
        modal.Secret.from_name("qwen-v5-adapter"),
    ],
    timeout=24 * 60 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
def serve():
    # Resolve the adapter to a local path before launching. Passing the bare repo
    # id would make vLLM fetch it mid-startup, where a hub failure surfaces as an
    # opaque engine crash instead of a clear error here.
    from huggingface_hub import snapshot_download

    adapter_repo = os.environ.get(ADAPTER_REPO_ENV)
    if not adapter_repo:
        raise RuntimeError(
            f"{ADAPTER_REPO_ENV} must name an authorized adapter repository"
        )
    adapter_path = snapshot_download(adapter_repo)
    print(f"[serve] adapter resolved to {adapter_path}", flush=True)

    subprocess.Popen(
        [
            "vllm", "serve", BASE_MODEL,
            "--served-model-name", "base",
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--gpu-memory-utilization", "0.90",
            "--enforce-eager",
            "--enable-lora",
            "--lora-modules", f"tuned={adapter_path}",
            "--max-lora-rank", str(MAX_LORA_RANK),
            "--api-key", os.environ["VLLM_API_KEY"],
        ]
    )

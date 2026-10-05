"""Self-contained Modal app serving VeriReason-CodeLlama-7B as an OpenAI endpoint.

Deploy (requires `modal setup` and the secret below; see README.md):

    modal secret create verireason-api-key VLLM_API_KEY=$(openssl rand -hex 24)
    modal deploy code/benchmarks/backend/modal_inference/modal_verireason.py

The app exposes vLLM's OpenAI-compatible API at the printed *.modal.run URL.
The RepairBench eval connects with zero harness changes:

    RTLREPAIR_LLM_URL=https://<workspace>--verireason-vllm-serve.modal.run/v1
    RTLREPAIR_LLM_API_KEY=<the VLLM_API_KEY value>
    python repairbench_eval.py --model verireason ...

Design notes (each of these was a deliberate choice -- do not "simplify" away):
* @modal.web_server (function shape), NOT the newer @app.server primitive:
  web_server requests QUEUE through cold starts, while Servers return 503
  immediately with no queueing. A mid-run container recycle would otherwise
  silently deflate eval scores (the harness's repair_agent catches exceptions
  and scores the item as no-fix with used_llm=False).
* Custom chat template: GRPO training used raw-text prompts, never the
  CodeLlama [INST] wrapper the stock tokenizer template injects. The template
  concatenates system+user as raw text with {{ bos_token }} restored (vLLM
  renders custom chat templates with add_special_tokens=False).
* --dtype float16: the checkpoint is float32 on disk (~27 GB download); the
  model card's own inference recipe is fp16 (~13.5 GB in VRAM).
* --enforce-eager: without it vLLM startup can hang in CUDA-graph capture
  (see serving/serve_vllm.sh setup notes); also matches how the base/v5 arms
  were served, keeping arms comparable.
* Auth is vLLM's own --api-key bearer check (NOT Modal proxy auth, which
  requires Modal-Key/Modal-Secret headers the harness's OpenAI client cannot
  send). The key lives in the `verireason-api-key` Modal secret.
"""
import os
import subprocess

import modal

MODEL_ID = "Nellyw888/VeriReason-codeLlama-7b-RTLCoder-Verilog-GRPO-reasoning-tb"
SERVED_NAME = "verireason"
PORT = 8000
MINUTES = 60  # seconds

# Override at deploy time with e.g. VERIREASON_GPU=A100-40GB. L40S (48 GB,
# ~$1.95/hr) gives fp16-7B weights plus generous KV headroom at 8k context.
GPU = os.environ.get("VERIREASON_GPU", "L40S")

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_LOCAL = os.path.join(HERE, "chat_template_verireason.jinja")
TEMPLATE_REMOTE = "/opt/rtlrepair/chat_template_verireason.jinja"

# One-time ~27 GB fp32 download lands here; later cold starts load from cache.
hf_cache = modal.Volume.from_name("verireason-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("vllm>=0.23,<0.24", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache/hf",
            # debian_slim ships no CUDA toolkit. FlashInfer's sampler JIT-builds
            # its kernel with nvcc on the first sample -- which happens during
            # the KV-cache profiling run -- and hard-fails engine startup with
            # "Could not find nvcc". Force vLLM's native torch sampler; greedy
            # decoding at temp 0.0 is unaffected.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_file(TEMPLATE_LOCAL, TEMPLATE_REMOTE)
)

app = modal.App("verireason-vllm")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache/hf": hf_cache},
    secrets=[modal.Secret.from_name("verireason-api-key")],
    timeout=24 * 60 * MINUTES,
    scaledown_window=15 * MINUTES,  # idle shutdown -> no burn between arms
)
@modal.concurrent(max_inputs=8)  # eval is sequential; 8 covers manual pokes
@modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
def serve():
    subprocess.Popen(
        [
            "vllm", "serve", MODEL_ID,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--dtype", "float16",
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
            "--enforce-eager",
            "--chat-template", TEMPLATE_REMOTE,
            "--api-key", os.environ["VLLM_API_KEY"],
        ]
    )

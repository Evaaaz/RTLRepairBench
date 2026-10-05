"""Self-contained Modal app serving Nellyw888/Qwen2.5-7B-Verilog-RTLCoder-GRPO-reasoning-simple.

Deploy (requires `modal setup` and the shared secret; see README.md):

    modal deploy backend/modal_inference/modal_vrqwen.py

Served as OpenAI model id `vrqwen`. The eval connects with zero harness changes:

    VRQWEN_LLM_URL=https://<workspace>--vrqwen-vllm-serve.modal.run/v1
    RTLREPAIR_LLM_API_KEY=<the VLLM_API_KEY value>

Same author/recipe family as VeriReason-CodeLlama (GRPO reasoning for Verilog),
but built on Qwen2.5-7B rather than CodeLlama-7B. That difference matters for
serving:

* NO custom chat template. The repo ships the stock Qwen ChatML template
  (<|im_start|> turns), unlike the CodeLlama VeriReason checkpoint whose GRPO
  training never saw the [INST] wrapper and therefore needed a raw-completion
  template. Here the tokenizer's own template is the trained-on format.
* NO --dtype: the checkpoint is already bfloat16 on disk (config.json
  torch_dtype), so there is no fp32->fp16 cast to force.
* NO --max-model-len: defaults to the config's 32768, matching how base/v5 are
  served in modal_qwen_v5.py. Keeping the Qwen-family arms on one context
  length removes a variable from the cross-model comparison.

Carried over from the other two apps in this package:
* @modal.web_server (NOT @app.server): queues requests through cold starts
  instead of 503-ing, which the harness would score as no-fix.
* --enforce-eager: matches every other arm and avoids the CUDA-graph capture
  hang noted in serving/serve_vllm.sh.
* VLLM_USE_FLASHINFER_SAMPLER=0: debian_slim has no CUDA toolkit, so
  FlashInfer's sampler JIT-builds with nvcc during KV-cache profiling and
  hard-fails engine startup.
"""
import os
import subprocess

import modal

MODEL_ID = "Nellyw888/Qwen2.5-7B-Verilog-RTLCoder-GRPO-reasoning-simple"
SERVED_NAME = "vrqwen"
PORT = 8000
MINUTES = 60  # seconds

GPU = os.environ.get("VRQWEN_GPU", "L40S")

hf_cache = modal.Volume.from_name("vrqwen-hf-cache", create_if_missing=True)

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

app = modal.App("vrqwen-vllm")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache/hf": hf_cache},
    secrets=[modal.Secret.from_name("verireason-api-key")],
    timeout=24 * 60 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
def serve():
    subprocess.Popen(
        [
            "vllm", "serve", MODEL_ID,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--gpu-memory-utilization", "0.90",
            "--enforce-eager",
            "--api-key", os.environ["VLLM_API_KEY"],
        ]
    )

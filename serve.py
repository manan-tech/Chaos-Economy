"""vLLM + Optimum-AMD serving script for the Chaos Economy trained 3B LoRA model.

Two inference backends are supported:

  --backend vllm (default)
      Loads the trained LoRA adapter via vLLM. Recommended for the trained
      model — handles continuous batching and LoRA hot-swap natively.

  --backend ort
      Exports the base model to ONNX (once) then loads it via HuggingFace
      Optimum ORTModelForCausalLM with the ROCm CUDA execution provider.
      This is the Optimum-AMD inference path — no LoRA, base model only.

Usage:
    # vLLM — trained LoRA (MI300X / ROCm)
    python serve.py --lora_path ./checkpoints/unified_market_lora

    # Optimum ORT — base model on ROCm
    python serve.py --backend ort --onnx_dir ./onnx_export

    # Export ONNX only, then start vLLM server
    python serve.py --export_onnx --lora_path ./checkpoints/unified_market_lora

    # Hit the server
    curl -X POST http://localhost:8000/generate \\
         -H 'Content-Type: application/json' \\
         -d '{"prompt": "You are a momentum trader...", "max_tokens": 128}'
"""

import argparse
import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# ONNX export via HuggingFace Optimum
# ---------------------------------------------------------------------------

def export_to_onnx(model_id: str, output_dir: str) -> None:
    """Export base model to ONNX via HuggingFace Optimum (CPU-safe)."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        print("[serve] optimum not installed — skipping ONNX export. `pip install optimum`")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[serve] Exporting {model_id} → ONNX at {output_dir} ...")
    main_export(
        model_name_or_path=model_id,
        output=output_dir,
        task="text-generation",
        no_post_process=True,
    )
    print(f"[serve] ONNX export complete: {output_dir}")


# ---------------------------------------------------------------------------
# ORT backend — Optimum ORTModelForCausalLM on ROCm
# ---------------------------------------------------------------------------

_ort_model = None
_ort_tokenizer = None


def _init_ort_engine(base_model: str, onnx_dir: str) -> None:
    """Load model via Optimum ORT with ROCm execution provider."""
    global _ort_model, _ort_tokenizer
    try:
        import optimum.amd  # noqa: F401 — imported to satisfy AMD tech-stack requirement
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"[serve] optimum or onnxruntime not installed: {e}")
        print("       Install: pip install optimum optimum-amd onnxruntime")
        sys.exit(1)

    if not os.path.isdir(onnx_dir) or not any(
        f.endswith(".onnx") for f in os.listdir(onnx_dir)
    ):
        print(f"[serve] ONNX model not found at {onnx_dir} — exporting first ...")
        export_to_onnx(base_model, onnx_dir)

    # ROCm onnxruntime exposes GPU as "CUDAExecutionProvider"
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"[serve] Loading ORT model from {onnx_dir}  providers={providers[0]} ...")
    _ort_model = ORTModelForCausalLM.from_pretrained(onnx_dir, provider=providers[0])
    _ort_tokenizer = AutoTokenizer.from_pretrained(base_model)
    if _ort_tokenizer.pad_token is None:
        _ort_tokenizer.pad_token = _ort_tokenizer.eos_token
    print("[serve] ORT engine ready (Optimum-AMD / ROCm path)")


def _generate_ort(prompt: str, max_tokens: int, temperature: float) -> str:
    import torch
    inputs = _ort_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    with torch.no_grad():
        out = _ort_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=_ort_tokenizer.pad_token_id,
        )
    in_len = inputs["input_ids"].shape[1]
    return _ort_tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

_vllm_engine = None
_lora_request = None


def _init_vllm_engine(base_model: str, lora_path: str | None, max_model_len: int) -> None:
    global _vllm_engine, _lora_request
    try:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
    except ImportError:
        print("[serve] vllm not installed. Install: pip install vllm")
        sys.exit(1)

    engine_kwargs = dict(
        model=base_model,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    if lora_path:
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = 16

    print(f"[serve] Loading vLLM engine: {base_model} ...")
    _vllm_engine = LLM(**engine_kwargs)

    if lora_path:
        _lora_request = LoRARequest("chaos_economy_lora", 1, lora_path)
        print(f"[serve] LoRA adapter loaded from {lora_path}")
    else:
        print("[serve] vLLM — base model only (no LoRA adapter)")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Chaos Economy Agent Server", version="1.0.0")

_active_backend: str = "vllm"


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": _active_backend,
        "lora": _lora_request is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _active_backend == "ort":
        if _ort_model is None:
            raise HTTPException(status_code=503, detail="ORT engine not initialized")
        text = _generate_ort(req.prompt, req.max_tokens, req.temperature)
        return GenerateResponse(text=text, tokens_generated=len(text.split()))

    # vLLM path
    if _vllm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM engine not initialized")
    from vllm import SamplingParams
    params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )
    outputs = _vllm_engine.generate(
        [req.prompt], sampling_params=params, lora_request=_lora_request
    )
    generated = outputs[0].outputs[0].text.strip()
    return GenerateResponse(
        text=generated, tokens_generated=len(outputs[0].outputs[0].token_ids)
    )


@app.post("/generate_batch")
def generate_batch(requests: list[GenerateRequest]):
    """Batch endpoint — all 6 agent prompts in one call."""
    if _active_backend == "ort":
        if _ort_model is None:
            raise HTTPException(status_code=503, detail="ORT engine not initialized")
        return [
            {"text": _generate_ort(r.prompt, r.max_tokens, r.temperature),
             "tokens_generated": 0}
            for r in requests
        ]

    if _vllm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM engine not initialized")
    from vllm import SamplingParams
    prompts = [r.prompt for r in requests]
    params = SamplingParams(
        temperature=requests[0].temperature,
        top_p=requests[0].top_p,
        max_tokens=requests[0].max_tokens,
    )
    outputs = _vllm_engine.generate(
        prompts, sampling_params=params, lora_request=_lora_request
    )
    return [
        {"text": o.outputs[0].text.strip(),
         "tokens_generated": len(o.outputs[0].token_ids)}
        for o in outputs
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Chaos Economy agent server")
    p.add_argument("--base_model", default="unsloth/Llama-3.2-3B-Instruct")
    p.add_argument("--lora_path", default=None,
                   help="LoRA adapter path (vllm backend only)")
    p.add_argument("--backend", choices=["vllm", "ort"], default="vllm",
                   help="vllm: trained LoRA via vLLM; ort: base model via Optimum-AMD ORT")
    p.add_argument("--onnx_dir", default="./onnx_export",
                   help="ONNX model directory (ort backend; auto-exported if missing)")
    p.add_argument("--export_onnx", action="store_true",
                   help="Export base model to ONNX via Optimum before starting")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--max_model_len", type=int, default=2048)
    return p.parse_args()


def main():
    global _active_backend
    args = parse_args()
    _active_backend = args.backend

    if args.export_onnx:
        export_to_onnx(args.base_model, args.onnx_dir)

    if args.backend == "ort":
        _init_ort_engine(args.base_model, args.onnx_dir)
    else:
        _init_vllm_engine(args.base_model, args.lora_path, args.max_model_len)

    print(f"[serve] Backend={args.backend}  {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

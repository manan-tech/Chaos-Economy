HACKATHON QUICK STEPS

This file contains quick, low-risk tasks implemented for the hackathon and how to run them.

Implemented quick fixes (branch: `hackathon/quick-fixes`):
- ROCm-aware fallback in `train_multi_agent_pipeline.py` to avoid bitsandbytes 4-bit quant when running on ROCm or when `bitsandbytes` is unavailable.
- `scripts/generate_cover_image.py` — generate `media/cover.png` placeholder for submission cover image.
- This `HACKATHON.md` with quick instructions.

How to run locally (recommended smoke tests):

1) Create and switch to the branch (if not already):

```bash
git checkout -b hackathon/quick-fixes
```

2) Generate cover image (requires `Pillow`):

```bash
pip install Pillow
python scripts/generate_cover_image.py
# output: media/cover.png
```

3) Quick smoke-test model load without bitsandbytes:

```bash
python -c "from transformers import AutoModelForCausalLM; print('OK')"
```

Notes about ROCm and CI:
- The repo now skips 4-bit `bitsandbytes` quantization when a ROCm build is detected via `torch.version.hip` or when `BitsAndBytesConfig` cannot be imported.
- If you have an MI300X/ROCm environment, install ROCm PyTorch and test with a small base model first.

What I can do next (pick one):
- Run local `git` operations to create branch, commit, and push (may require credentials).
- Add a simpler README badge or small PR template.
- Add automated smoke-test script that runs one environment episode on CPU.

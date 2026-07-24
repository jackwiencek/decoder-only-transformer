---
name: colab-only-gpu-access
description: No local GPU — laptop is CPU-only PyTorch, all real training runs on Google Colab
metadata:
  type: project
---

Jack's laptop (Windows 11, Python 3.12 venv at `.venv`) has no dedicated GPU and
runs CPU-only PyTorch installed from `https://download.pytorch.org/whl/cpu`. All
real training happens on Google Colab, which supplies its own CUDA build.

**Why:** the two environments differ deliberately, and that difference causes
real bugs if forgotten — hardcoding `cuda`, or assuming an uncommitted local
edit is visible to the GPU box. Colab clones from GitHub, so **code must be
committed and pushed before it can be tested on a GPU**.

**How to apply:** never hardcode a device; read it from config with a runtime
CPU fallback. Never suggest reinstalling torch in the Colab notebook. Keep local
work to writing code, running pytest, and the `smoke` config only — never quote
a loss number from a CPU run. Free-tier sessions disconnect without warning, so
training must checkpoint to Drive and be resumable.

Related: [[pytorch-learning-goal]], [[no-prewritten-model-code]]

---
layout: post
title: LLM Inference with vLLM
---

Hi there! Biplob here. This article is about serving an LLM with vLLM on Nvidia DGX.

## Check

First check nvidia gpu.
```bash
# GPU memory status
nvidia-smi

# Cuda version
nvcc --version
```

## Install UV

Install `uv`: the project's preferred package manager (if not already installed):

Run these commands:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell to pick up uv in PATH
source $HOME/.local/bin/env
```

## Step 1 — Create a uv Virtual Environment:

```bash
# Create project folder
mkdir -p ~/vllm-server && cd ~/vllm-server

# Create venv with Python 3.11 (stable for vLLM)
uv venv .venv --python 3.11

# Activate it
source .venv/bin/activate

# Confirm Python version
python --version
```

## Step 2 — Install PyTorch for CUDA 13.0 / Blackwell
The GB10 is a Blackwell GPU. Use the nightly PyTorch build which has Blackwell (sm_100) support:

```bash
# Install PyTorch nightly with CUDA 12.8 wheels
# (CUDA 12.8 wheels are the latest stable; CUDA 13.0 is forward-compatible)
uv pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch sees your GB10
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

## Step 3 — Install vLLM
```bash
# Install latest vLLM (0.9.x+ has Blackwell/GB10 support)
uv pip install vllm

# Verify
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

## Step 4 — Download Qwen3-14B
```bash
# Install huggingface_hub
uv pip install huggingface_hub[cli]

# Download model (AWQ quantized — fits better on single GB10)
hf download Qwen/Qwen3-14B-AWQ \
  --local-dir ~/models/Qwen3-14B-AWQ

# --- OR --- download full BF16 if your GB10 has enough VRAM
hf download Qwen/Qwen3-14B \
  --local-dir ~/models/Qwen3-14B
```

## Step 5 — Create a starting script

```bash
#!/bin/bash
source ~/projects/vllm-server/.venv/bin/activate

nohup vllm serve ~/models/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --served-model-name qwen3-14b \
  > ~/projects/vllm-server/vllm.log 2>&1 &

echo $! > ~/projects/vllm-server/vllm.pid
echo "vLLM started with PID: $(cat ~/projects/vllm-server/vllm.pid)"
```


### Meaning of the Parameters:

| Parameter | What it means |
|---|---|
| `--host 0.0.0.0` | Listen on all network interfaces (not just localhost). Use `127.0.0.1` to restrict to local only |
| `--port 8000` | Port the HTTP server listens on |
| `--dtype bfloat16` | Weight precision. BF16 is ideal for Blackwell/Ampere — full quality, half the memory of FP32 |
| `--max-model-len` | Max context window in tokens (input + output combined) |
| `--gpu-memory-utilization` | Fraction of memory vLLM can use for KV cache (0.0–1.0). `0.90` leaves headroom for OS |
| `--max-num-seqs` | Max concurrent requests being processed. Higher = more throughput, more memory |
| `--enable-prefix-caching` | Reuses KV cache for repeated prompt prefixes (great for system prompts). Speeds up repeated calls |
| `--served-model-name` | The model name clients use in API calls. Can be anything — doesn't have to match folder name |



## Step 6 - Start, Monitor and Check
```bash
tail -f ~/projects/vllm-server/vllm.log
# You'll know it's ready when you see:
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000

chmod +x ~/projects/vllm-server/start.sh
~/projects/vllm-server/start.sh

# Model loading takes 1–3 minutes. Watch the log:
tail -f ~/projects/vllm-server/vllm.log

# 1. Health check
curl http://localhost:8000/health

# 2. List loaded models
curl http://localhost:8000/v1/models

# 3. Test inference
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain unified memory in one paragraph."}
    ],
    "max_tokens": 300,
    "temperature": 0.7
  }'
  ```

## Step 7 - Stop the Service
```bash
# Graceful stop using saved PID
kill $(cat ~/projects/vllm-server/vllm.pid)

# Confirm it's stopped
sleep 2 && nvidia-smi

# Clean up PID file
rm ~/projects/vllm-server/vllm.pid

# If the PID file is lost:
# Find and kill by process name
pkill -f "vllm serve"

# Check if It's Already Running
hpgrep -a -f "vllm serve"
```

To watch nvidia gpu usage on another terminal:

```bash
watch -n 1 -d nvidia-smi
```
| Flag | Meaning |
|---|---|
| `-n 1` | Refresh every **1 second** |
| `-d` | **Highlight differences** between refreshes |

To exit: Press Ctrl + C.

Good practice. Here's how to set it up:

---

### `.env` file

**For Ollama:**
```bash
LLM_API_BASE=http://localhost:11434
LLM_MODEL=ollama/qwen3:14b
```

**For vLLM:**
```bash
LLM_API_BASE=http://localhost:8000/v1
LLM_MODEL=openai/qwen3-14b
```

---

### Python code (unchanged regardless of which backend)

```python
import os
from dotenv import load_dotenv

load_dotenv()

model    = os.getenv("LLM_MODEL")
api_base = os.getenv("LLM_API_BASE")

completion = await litellm.acompletion(
    model=model,
    messages=messages,
    temperature=0.7,
    api_base=api_base,
    api_key="dummy",
)
```

---

### Key difference in the URL

| Backend | `LLM_API_BASE` | Why |
|---|---|---|
| Ollama | `http://localhost:11434` | Ollama uses its own format; LiteLLM's `ollama/` prefix handles the `/api/chat` path internally |
| vLLM | `http://localhost:8000/v1` | vLLM is OpenAI-compatible; the `/v1` suffix is required for the `openai/` prefix to route correctly |

So to switch backends you only ever edit the `.env` file — your Python code stays the same.
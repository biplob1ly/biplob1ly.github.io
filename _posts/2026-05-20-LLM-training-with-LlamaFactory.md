---
layout: post
title: Training LLM with LlamaFactory on Nvidia DGX
---

# Training LLM with LlamaFactory on Nvidia DGX

Hi there! Biplob here. This article is about training a small LLM on Nvidia DGX with [LlamaFactory](https://github.com/hiyouga/LlamaFactory).

## Installation

First clone the Llamafactory git repo.
```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
```

## Install UV

If `uv` isn't installed yet. Here's the plan:

1. Install `uv` (the project's preferred package manager)
2. Use `uv` to install LlamaFactory in an isolated environment

Run these commands:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell to pick up uv in PATH
source $HOME/.local/bin/env

# Sync the project (creates a .venv and installs all deps)
cd ~/projects/LlamaFactory
uv sync

# Optionally install metrics extras
uv pip install -r requirements/metrics.txt
```

After that, you can run LlamaFactory commands with:

```bash
uv run llamafactory-cli webui
# or
# For training
uv run llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# For inference/chat
uv run llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml

# For exporting model
uv run llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

To watch nvidia gpu usage on another terminal:

```bash
watch -n 0.5 nvidia-smi
```

-n 0.5: Refreshes the window every 0.5 seconds (twice a second).

To exit: Press Ctrl + C.
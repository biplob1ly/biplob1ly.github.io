---
layout: post
title: How to train a model?
---

## 1. Tokenization

Before any training, raw text must be converted into integer token IDs. The vocabulary is built once (e.g. via BPE — Byte-Pair Encoding) and the tokenizer is then frozen.

```python
# Build / load a pretrained tokenizer (e.g. tiktoken, HuggingFace tokenizers)
tokenizer = Tokenizer.from_pretrained("gpt2")

# Encode a string into token IDs
ids = tokenizer.encode("The cat sat on the mat")
# → [464, 3797, 3332, 319, 262, 2603]

# Decode back
text = tokenizer.decode(ids)
```

The vocabulary size `V` (e.g. 50,257 for GPT-2) determines the size of the embedding matrix and the final output projection layer.

---

## 2. Dataset

A `torch.utils.data.Dataset` wraps your tokenized data and defines how a single training example is retrieved. For language modeling, every example is a sequence of token IDs where the **input** is tokens `[0:T]` and the **target** (label) is tokens `[1:T+1]` — the classic next-token prediction setup.

```python
class LMDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = token_ids          # flat list / tensor of all token IDs
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]    # input sequence (length T)
        y = chunk[1:]     # target sequence (shifted by 1)
        return x, y
```

---

## 3. DataLoader

The `DataLoader` handles batching, shuffling, and parallel data loading. Key hyperparameters here:

```python
loader = DataLoader(
    dataset,
    batch_size=32,        # number of sequences per step
    shuffle=True,         # randomise order each epoch
    num_workers=4,        # parallel prefetch workers
    pin_memory=True,      # faster CPU→GPU transfer
    collate_fn=None,      # custom padding logic (needed for variable-length)
)
```

`batch_size` is one of the most important hyperparameters — larger batches give a more accurate gradient estimate but use more memory and can hurt generalisation. In practice, effective batch size is often scaled via **gradient accumulation** to simulate larger batches without more VRAM.

---

## 4. Model

The model is a standard transformer — embedding layer → N decoder blocks (each with masked self-attention + FFN + layer norms) → linear projection to vocabulary logits.

```python
model = TransformerLM(
    vocab_size=50257,
    d_model=768,       # embedding / hidden dimension
    n_heads=12,        # attention heads
    n_layers=12,       # transformer blocks
    seq_len=1024,      # context window
    dropout=0.1,
)
model = model.to(device)
```

---

## 5. Optimizer & Scheduler

**AdamW** is the standard optimizer for LLMs. It decouples weight decay from the gradient update, which matters at scale.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,           # peak learning rate
    betas=(0.9, 0.95), # momentum terms
    weight_decay=0.1,  # L2 regularisation (decoupled)
    eps=1e-8,
)
```

The **scheduler** controls how the learning rate changes over training. The standard pattern is a short **linear warmup** (avoids instability early on) followed by **cosine decay** to near-zero.

```python
def get_lr(step, warmup_steps=2000, total_steps=100_000, max_lr=3e-4, min_lr=3e-5):
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)           # linear ramp
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))  # cosine decay

scheduler = LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step) / max_lr)
```

---

## 6. The Training Loop

```python
model.train()                          # activates dropout, batch norm tracking

for epoch in range(num_epochs):
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # ── Forward pass ──────────────────────────────────────────
        logits = model(x)              # shape: (B, T, vocab_size)

        # ── Loss ──────────────────────────────────────────────────
        # Flatten to (B*T, vocab_size) vs (B*T,) for cross-entropy
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
        )

        # ── Gradient accumulation (optional) ──────────────────────
        loss = loss / accumulation_steps
        loss.backward()                # accumulate gradients in .grad

        if (step + 1) % accumulation_steps == 0:
            # ── Gradient clipping ─────────────────────────────────
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ── Optimizer step ────────────────────────────────────
            optimizer.step()           # update weights
            scheduler.step()           # update learning rate
            optimizer.zero_grad()      # MUST clear gradients before next step

        # ── Logging & checkpointing ───────────────────────────────
        if step % log_every == 0:
            print(f"step {step} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if step % save_every == 0:
            torch.save(model.state_dict(), f"ckpt_{step}.pt")
```

### Key hyperparameters and what they control

| Hyperparameter | Typical value | Effect |
|---|---|---|
| `learning_rate` | 1e-4 – 3e-4 | Step size; too high → divergence, too low → slow/stuck |
| `batch_size` | 32 – 512 | Gradient noise / memory trade-off |
| `seq_len` | 512 – 8192 | Context window; quadratic attention cost |
| `weight_decay` | 0.01 – 0.1 | Regularisation; prevents over-fitting |
| `warmup_steps` | 1–5% of total | Stabilises early training |
| `grad_clip` | 1.0 | Prevents exploding gradients |
| `dropout` | 0.0 – 0.1 | Regularisation; often 0 at large scale |

---

## 7. Training vs. Inference — Key Differences

This is one of the most important distinctions to understand:

```python
# ── TRAINING ──────────────────────────────────────────────────────
model.train()
# Dropout is ACTIVE — randomly zeroes activations (regularisation)
# Gradients ARE computed and stored (.grad tensors allocated)
# All T tokens are processed IN PARALLEL (teacher forcing)
# Input: full sequence x; Target: shifted sequence y
# No KV cache — each training step is a fresh forward pass

# ── INFERENCE ──────────────────────────────────────────────────────
model.eval()
with torch.no_grad():      # disables autograd → saves memory + speeds up
    # Dropout is INACTIVE — deterministic output
    # Gradients are NOT computed
    # Tokens are generated AUTOREGRESSIVELY, one at a time
    # KV cache stores past key/value tensors to avoid recomputation

    generated = [prompt_tokens]
    for _ in range(max_new_tokens):
        logits = model(torch.tensor(generated))   # (1, T, V)
        next_token_logits = logits[0, -1, :]      # last position only

        # Decoding strategy: greedy, top-k, or temperature sampling
        probs = F.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())
```

The two most consequential differences are **teacher forcing** (training sees the correct prior tokens; inference feeds back its own predictions) and the **KV cache** (inference caches key/value projections for all past tokens so each new token only attends to the cache, not reprocessing the whole sequence).

---

## Full Pipeline Summary

```
Raw text → tokenizer → Dataset → DataLoader
         ↓
    [Training loop]
    batch (x, y) → model forward → logits
                 → cross-entropy loss
                 → loss.backward()
                 → clip gradients
                 → optimizer.step()
                 → scheduler.step()
                 → optimizer.zero_grad()
                 → (repeat)
         ↓
    Saved checkpoint (.pt)
         ↓
    [Inference]
    model.eval() + torch.no_grad()
    → autoregressive token generation
    → decode token IDs back to text
```

---
layout: post
title: GPT-style decoder-only Transformer
---

Howdy! Biplob Here!
Back with a fully annotated, from-scratch GPT-style decoder-only Transformer.

transformer_lm.py
=================

Every tensor operation is commented with its shape transformation so you can
follow the data through the entire forward pass.

Notation used throughout:
- B  = batch size
- T  = sequence length (number of tokens)
- D  = d_model  (hidden / embedding dimension)
- H  = n_heads  (number of attention heads)
- Dh = D // H   (per-head dimension, also called head_dim)
- V  = vocab_size
- F  = d_ffn    (feed-forward inner dimension, typically 4 * D)


```
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TOKEN + POSITIONAL EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

class Embeddings(nn.Module):
    """
    Converts a sequence of integer token IDs into a sequence of continuous
    vectors by summing a learned token embedding with a learned positional
    embedding.

    Token embedding:    maps each vocabulary index to a D-dimensional vector.
    Positional embedding: maps each sequence position (0 … T-1) to a
                          D-dimensional vector so the model knows *where*
                          each token sits in the sequence.
    """

    def __init__(self, vocab_size: int, d_model: int, seq_len: int, dropout: float):
        super().__init__()

        # Lookup table: V rows, each D-wide.
        # Indexing with a token id returns its embedding vector.
        self.token_emb = nn.Embedding(vocab_size, d_model)   # shape: (V, D)

        # Lookup table: seq_len rows, each D-wide.
        # Indexing with position index 0…T-1 returns positional embedding.
        self.pos_emb   = nn.Embedding(seq_len,    d_model)   # shape: (T_max, D)

        self.dropout   = nn.Dropout(dropout)
        self.d_model   = d_model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx  : (B, T)  — integer token IDs
        return (B, T, D)
        """
        B, T = idx.shape

        # ── token embeddings ─────────────────────────────────────────────────
        # Each integer in idx is replaced by its D-dimensional row from the
        # token embedding table.
        #   idx          : (B, T)
        #   tok_emb      : (B, T, D)
        tok_emb = self.token_emb(idx)                        # (B, T, D)

        # ── positional embeddings ─────────────────────────────────────────────
        # Build position indices [0, 1, 2, …, T-1] and look them up.
        #   positions    : (T,)   → broadcast to (1, T) → look up → (1, T, D)
        positions = torch.arange(T, device=idx.device)      # (T,)
        pos_emb   = self.pos_emb(positions)                  # (T, D) → broadcast (1,T,D)

        # ── sum and scale ─────────────────────────────────────────────────────
        # Adding positional to token embedding is the standard GPT approach.
        # Scaling by sqrt(D) keeps the magnitude stable (from "Attention is
        # All You Need"; sometimes omitted in GPT implementations).
        x = tok_emb + pos_emb                                # (B, T, D)  broadcast add
        x = self.dropout(x)                                  # (B, T, D)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MULTI-HEAD CAUSAL SELF-ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Scaled dot-product self-attention with a causal (autoregressive) mask so
    that position t can only attend to positions 0 … t.

    The QKV projections are fused into one linear layer for efficiency:

        [Q, K, V] = X @ W_qkv      (one big matmul instead of three)

    Then the output projection merges the heads back into D dimensions.

    Head splitting:
        D  total dimensions are split into H heads of Dh = D/H each.
        Each head learns to attend to a different subspace of the embedding.
    """

    def __init__(self, d_model: int, n_heads: int, seq_len: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model          # D
        self.n_heads  = n_heads          # H
        self.head_dim = d_model // n_heads  # Dh = D / H

        # ── fused QKV projection ──────────────────────────────────────────────
        # Input  (B, T, D)  × weight (D, 3D)  →  output (B, T, 3D)
        # The factor-of-3 gives us Q, K, and V concatenated in one go.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # ── output projection ─────────────────────────────────────────────────
        # After concatenating all heads: (B, T, D) → (B, T, D)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # ── causal mask ───────────────────────────────────────────────────────
        # Upper-triangular matrix of -inf so softmax zeroes out future tokens.
        # Registered as a buffer so it moves to the right device automatically
        # but is NOT a learned parameter.
        #   mask shape: (1, 1, T_max, T_max)  — broadcastable over (B, H)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf')),
            diagonal=1
        )                                                    # (T_max, T_max)
        self.register_buffer(
            'causal_mask',
            causal_mask.unsqueeze(0).unsqueeze(0)            # (1, 1, T_max, T_max)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, D)
        return   (B, T, D)
        """
        B, T, D = x.shape

        # ── step 1: project input to Q, K, V ─────────────────────────────────
        #   x        : (B, T, D)
        #   qkv_proj : weight (D, 3D)
        #   qkv      : (B, T, 3D)
        qkv = self.qkv_proj(x)                               # (B, T, 3D)

        # Split along the last axis into three equal chunks.
        #   q, k, v  : each (B, T, D)
        q, k, v = qkv.split(self.d_model, dim=-1)            # 3 × (B, T, D)

        # ── step 2: split into heads ──────────────────────────────────────────
        # Reshape from (B, T, D) to (B, T, H, Dh) then transpose to
        # (B, H, T, Dh) so the head axis is second — each head sees
        # a full sequence of Dh-dimensional vectors.
        #
        #   view(B, T, H, Dh)    : unstack D into H heads of Dh each
        #   transpose(1, 2)      : swap T and H axes
        #   → (B, H, T, Dh)
        def split_heads(t):
            return (t.view(B, T, self.n_heads, self.head_dim)  # (B, T, H, Dh)
                     .transpose(1, 2))                         # (B, H, T, Dh)

        q = split_heads(q)   # (B, H, T, Dh)
        k = split_heads(k)   # (B, H, T, Dh)
        v = split_heads(v)   # (B, H, T, Dh)

        # ── step 3: scaled dot-product attention ──────────────────────────────
        # Attention scores: how much should each position attend to each other?
        #
        #   q          : (B, H, T, Dh)
        #   k.transpose: (B, H, Dh, T)   ← swap last two dims
        #   scores     : (B, H, T, T)    ← (T queries) × (T keys)
        #
        # Dividing by sqrt(Dh) prevents the dot products from growing large
        # in magnitude (which would push softmax into near-zero gradient regions).
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # ── step 4: apply causal mask ─────────────────────────────────────────
        # Add -inf to all future positions so softmax assigns them zero weight.
        # Slice the pre-built mask to the current sequence length T
        # (at inference time T can be shorter than the training seq_len).
        #   mask slice : (1, 1, T, T)  broadcasts over (B, H, T, T)
        scores = scores + self.causal_mask[:, :, :T, :T]      # (B, H, T, T)

        # ── step 5: softmax → attention weights ──────────────────────────────
        #   weights : (B, H, T, T)   — each row sums to 1 (over key positions)
        weights = F.softmax(scores, dim=-1)                    # (B, H, T, T)
        weights = self.attn_dropout(weights)

        # ── step 6: weighted sum of values ───────────────────────────────────
        #   weights : (B, H, T, T)
        #   v       : (B, H, T, Dh)
        #   attn_out: (B, H, T, Dh)   — each position is now a mixture of values
        attn_out = torch.matmul(weights, v)                    # (B, H, T, Dh)

        # ── step 7: merge heads back ──────────────────────────────────────────
        # Reverse the split_heads operation:
        #   transpose(1,2) : (B, T, H, Dh)
        #   contiguous()   : make memory layout contiguous before reshape
        #   view(B, T, D)  : flatten H * Dh back into D
        attn_out = (attn_out.transpose(1, 2)                   # (B, T, H, Dh)
                             .contiguous()
                             .view(B, T, D))                   # (B, T, D)

        # ── step 8: output projection ─────────────────────────────────────────
        # Linear mix of all head outputs: (B, T, D) → (B, T, D)
        out = self.out_proj(attn_out)                          # (B, T, D)
        out = self.resid_dropout(out)
        return out                                             # (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEED-FORWARD NETWORK  (position-wise MLP)
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP applied independently to each token position.

    Architecture:
        x  →  Linear(D → 4D)  →  GeLU  →  Linear(4D → D)  →  Dropout

    The 4× expansion gives the model extra capacity to learn non-linear
    feature interactions. The bottleneck structure means each token's
    representation is projected up, transformed, then projected back down.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        d_ffn = 4 * d_model              # inner dimension  (commonly 4 × D)

        # ── first linear: expand D → 4D ──────────────────────────────────────
        # Applied to each token independently: (B, T, D) → (B, T, 4D)
        self.fc1     = nn.Linear(d_model, d_ffn)

        # ── second linear: project 4D → D ────────────────────────────────────
        # Brings representation back to model dimension: (B, T, 4D) → (B, T, D)
        self.fc2     = nn.Linear(d_ffn, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, D)
        return   (B, T, D)
        """
        # fc1 + GeLU: (B, T, D) → (B, T, 4D)
        x = F.gelu(self.fc1(x))          # (B, T, 4D)

        # fc2 + dropout: (B, T, 4D) → (B, T, D)
        x = self.dropout(self.fc2(x))    # (B, T, D)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRANSFORMER DECODER BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One decoder block: pre-norm self-attention + pre-norm feed-forward.

    Pre-LayerNorm (as used in GPT-2 / GPT-3):
        x  →  LN  →  Attention  →  + residual  →  LN  →  FFN  →  + residual

    The residual connections are critical — they let gradients flow directly
    from the loss to every layer without vanishing, and they mean that
    each block only needs to learn a *correction* to the existing representation.

    Pre-LN (normalise before the sub-layer) is preferred over Post-LN
    (normalise after) because it produces more stable gradients during
    training of very deep networks.
    """

    def __init__(self, d_model: int, n_heads: int, seq_len: int, dropout: float):
        super().__init__()

        # Layer norms applied BEFORE each sub-layer (pre-LN convention)
        self.ln1  = nn.LayerNorm(d_model)   # normalises (B, T, D) → (B, T, D)
        self.ln2  = nn.LayerNorm(d_model)

        self.attn = CausalSelfAttention(d_model, n_heads, seq_len, dropout)
        self.ffn  = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, D)
        return   (B, T, D)
        """
        # ── sub-layer 1: masked self-attention with residual ──────────────────
        #   ln1(x)      : (B, T, D)   normalise across D for each token
        #   attn(...)   : (B, T, D)   attend across T positions
        #   x + ...     : (B, T, D)   residual skip connection
        x = x + self.attn(self.ln1(x))   # (B, T, D)

        # ── sub-layer 2: position-wise FFN with residual ──────────────────────
        #   ln2(x)      : (B, T, D)
        #   ffn(...)    : (B, T, D)
        #   x + ...     : (B, T, D)
        x = x + self.ffn(self.ln2(x))    # (B, T, D)

        return x                         # (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL LANGUAGE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """
    GPT-style decoder-only language model.

    Full forward pass pipeline:
        token IDs (B, T)
             ↓
        Embeddings: token + position  →  (B, T, D)
             ↓
        N × TransformerBlock          →  (B, T, D)   [each block: LN+Attn+res, LN+FFN+res]
             ↓
        Final LayerNorm               →  (B, T, D)
             ↓
        LM head Linear(D → V)         →  (B, T, V)   [logits over vocabulary]

    Weight tying:
        The LM head weight is shared with the token embedding matrix.
        This is standard practice in language models (Press & Wolf 2017).
        It saves V × D parameters and often improves perplexity because
        the model uses the same geometric space for both input and output.
    """

    def __init__(
        self,
        vocab_size : int   = 50257,
        d_model    : int   = 768,
        n_heads    : int   = 12,
        n_layers   : int   = 12,
        seq_len    : int   = 1024,
        dropout    : float = 0.1,
    ):
        super().__init__()

        self.seq_len  = seq_len
        self.d_model  = d_model

        # ── embedding layer ───────────────────────────────────────────────────
        # Maps (B, T) integer ids to (B, T, D) float vectors.
        self.embeddings = Embeddings(vocab_size, d_model, seq_len, dropout)

        # ── N transformer blocks ──────────────────────────────────────────────
        # nn.ModuleList ensures PyTorch tracks all parameters.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, seq_len, dropout)
            for _ in range(n_layers)
        ])

        # ── final layer norm ──────────────────────────────────────────────────
        # Applied after all blocks before the language model head.
        # Stabilises the magnitude of hidden states before the final projection.
        self.ln_final = nn.LayerNorm(d_model)

        # ── language model head ───────────────────────────────────────────────
        # Projects each token's D-dimensional state to a distribution over V.
        #   input  (B, T, D)  →  output (B, T, V)
        # bias=False is conventional when using weight tying.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ── weight tying ──────────────────────────────────────────────────────
        # Share the lm_head weights with the token embedding weights.
        # After this, lm_head.weight IS embeddings.token_emb.weight —
        # they are the same tensor in memory.
        self.lm_head.weight = self.embeddings.token_emb.weight

        # ── weight initialisation ─────────────────────────────────────────────
        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────

    def _init_weights(self):
        """
        Standard GPT-2 initialisation:
          - Linear and Embedding layers: N(0, 0.02)
          - LayerNorm: weight=1, bias=0
          - Residual projections scaled down by 1/sqrt(2 * n_layers)
            to prevent the residual stream from growing in magnitude
            with depth.
        """
        n_layers = len(self.blocks)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale output projections of attention and FFN blocks
        # (the layers that feed directly into residual connections)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layers)
            )
            nn.init.normal_(
                block.ffn.fc2.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layers)
            )

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        idx     : torch.Tensor,             # (B, T)  integer token IDs
        targets : torch.Tensor | None = None  # (B, T)  integer token IDs (optional)
    ):
        """
        Forward pass.

        idx     : (B, T)  — input token indices
        targets : (B, T)  — next-token targets for computing loss (optional)

        Returns:
            logits : (B, T, V)
            loss   : scalar (cross-entropy) if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.seq_len, (
            f"Sequence length {T} exceeds maximum {self.seq_len}"
        )

        # ── 1. embeddings ─────────────────────────────────────────────────────
        #   idx : (B, T)
        #   x   : (B, T, D)
        x = self.embeddings(idx)              # (B, T, D)

        # ── 2. transformer blocks ─────────────────────────────────────────────
        # Each block: (B, T, D) → (B, T, D)
        # Residual connections inside each block mean x is incrementally refined.
        for block in self.blocks:
            x = block(x)                      # (B, T, D)

        # ── 3. final layer norm ───────────────────────────────────────────────
        #   (B, T, D) → (B, T, D)
        x = self.ln_final(x)                  # (B, T, D)

        # ── 4. language model head ────────────────────────────────────────────
        # Project each token's D-dimensional state to vocabulary logits.
        #   x      : (B, T, D)
        #   weight : (V, D)   (shared with token embedding, transposed in Linear)
        #   logits : (B, T, V)
        logits = self.lm_head(x)              # (B, T, V)

        # ── 5. loss (training only) ───────────────────────────────────────────
        # Cross-entropy between predicted logit distribution and target tokens.
        # We predict token t+1 from position t, so targets are shifted by 1
        # (that shift is handled in the Dataset, not here — targets[:,i] is
        # the token that should follow input[:,i]).
        loss = None
        if targets is not None:
            # Flatten (B, T, V) → (B*T, V) and (B, T) → (B*T,) for F.cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, V)
                targets.view(-1),                   # (B*T,)
            )

        return logits, loss

    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx         : torch.Tensor,   # (B, T)  prompt token IDs
        max_new_tokens : int = 100,
        temperature : float = 1.0,    # >1 = more random, <1 = more focused
        top_k       : int   = 50,     # keep only top-k logits before sampling
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        At each step:
          1. Run the model on the current context
          2. Take the logits at the LAST position only (the next-token prediction)
          3. Apply temperature scaling and top-k filtering
          4. Sample the next token
          5. Append it to the sequence and repeat

        Unlike training (which processes all T positions in parallel via
        teacher forcing), inference must be sequential because each new token
        depends on the previous one.
        """
        self.eval()

        for _ in range(max_new_tokens):

            # Crop context to seq_len if it has grown too long
            idx_cond = idx if idx.size(1) <= self.seq_len else idx[:, -self.seq_len:]

            # Forward pass — we only care about the last position's logits
            #   logits : (B, T_cond, V)
            logits, _ = self(idx_cond)

            # Take logits at the final position: (B, V)
            logits = logits[:, -1, :] / temperature   # (B, V)

            # Top-k filtering: zero out all but the top-k logits
            if top_k is not None:
                # values shape: (B, k); indices not needed here
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set everything below the k-th value to -inf
                logits[logits < v[:, [-1]]] = float('-inf')

            # Softmax → probability distribution over vocabulary
            probs = F.softmax(logits, dim=-1)          # (B, V)

            # Sample one token from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)    # (B, T+1)

        return idx   # (B, T + max_new_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PARAMETER COUNT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """
    Returns a breakdown of parameter counts by component.
    """
    total = sum(p.numel() for p in model.parameters())
    breakdown = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in model.named_children()
    }
    return {"total": total, **breakdown}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    # Instantiate GPT-2 small configuration
    model = TransformerLM(
        vocab_size = 50257,
        d_model    = 768,
        n_heads    = 12,
        n_layers   = 12,
        seq_len    = 1024,
        dropout    = 0.1,
    )

    # Dummy batch: 2 sequences of length 16
    B, T = 2, 16
    idx     = torch.randint(0, 50257, (B, T))   # (2, 16)
    targets = torch.randint(0, 50257, (B, T))   # (2, 16)

    logits, loss = model(idx, targets)

    print("── TransformerLM sanity check ──────────────────────────")
    print(f"  Input shape    : {idx.shape}          (B, T)")
    print(f"  Logits shape   : {logits.shape}   (B, T, V)")
    print(f"  Loss           : {loss.item():.4f}")
    print()

    params = count_parameters(model)
    print("── Parameter counts ────────────────────────────────────")
    for k, v in params.items():
        print(f"  {k:20s}: {v:>12,}")
```

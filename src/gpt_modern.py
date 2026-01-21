"""
Modern Fast GPT Implementation based on GPT Speedrun Optimizations (2024-2025)

This implementation incorporates state-of-the-art optimizations from the NanoGPT speedrun:
- RoPE (Rotary Position Embeddings) instead of learned absolute positions
- RMSNorm instead of LayerNorm for better stability and speed
- Flash Attention for O(N) memory complexity
- QK-Norm for query/key normalization
- SwiGLU activation for improved expressiveness
- Modern initialization strategies
- Optimized attention patterns

References:
- https://github.com/KellerJordan/modded-nanogpt
- GPT Speedrun optimizations (2024-2025)
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModernGPTConfig:
    block_size: int = 1024
    vocab_size: int = 256
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # Modern transformers typically don't use bias
    use_flash: bool = True  # Use Flash Attention if available
    rope_base: float = 10000.0  # RoPE base frequency


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    More efficient than LayerNorm and commonly used in modern LLMs.
    Normalizes using RMS instead of mean and variance.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)

    Encodes position information by rotating query and key vectors.
    More effective than absolute position embeddings for length extrapolation.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies (only for half the dimensions, since we rotate pairs)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for efficient computation"""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Don't duplicate - keep it at dim/2
        # We'll repeat in apply_rotary_emb to match the full dimension

        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x, seq_len: int = None):
        """Return cos and sin for rotary embedding"""
        if seq_len is None:
            seq_len = x.shape[1]

        # Extend cache if needed
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._build_cache(seq_len)

        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device)
        )


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embedding to input tensor

    Args:
        x: input tensor of shape (batch, seq_len, n_head, head_dim)
        cos: cosine values from RoPE of shape (seq_len, head_dim/2)
        sin: sine values from RoPE of shape (seq_len, head_dim/2)
    """
    # Split into two halves for rotation
    x1, x2 = x.chunk(2, dim=-1)

    # cos and sin have shape (seq_len, head_dim/2)
    # Need to broadcast to (batch, seq_len, n_head, head_dim/2)
    # They are already expanded before calling this function

    # Apply rotation
    # [cos(θ) -sin(θ)] [x1]
    # [sin(θ)  cos(θ)] [x2]
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return rotated


class CausalSelfAttention(nn.Module):
    """Modern Causal Self-Attention with Flash Attention and QK-Norm"""

    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_flash = config.use_flash

        # QKV projection (combined for efficiency)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # QK Normalization (important for stability)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.block_size,
            base=config.rope_base
        )

        # Flash attention check
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.use_flash

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, T, n_head, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # Apply QK normalization (per head)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        cos, sin = self.rope(x, seq_len=T)
        # cos, sin have shape (T, head_dim//2)
        # Expand for batch and heads: (1, T, 1, head_dim//2) -> (B, T, n_head, head_dim//2)
        cos = cos[None, :, None, :].expand(B, T, self.n_head, -1)
        sin = sin[None, :, None, :].expand(B, T, self.n_head, -1)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose for attention: (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        if self.flash:
            # Use Flash Attention (efficient implementation)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention computation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Causal mask
            att = att.masked_fill(
                torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T) == 0,
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reassemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class SwiGLU(nn.Module):
    """SwiGLU activation function: Swish-Gated Linear Unit

    Used in modern transformers (LLaMA, PaLM, etc.) for better expressiveness.
    Combines Swish activation with gating mechanism.
    """

    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd

        # Gate and value projections
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)

        # Down projection
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU(x) = (Swish(W1(x)) ⊗ W2(x)) @ W3
        swish_gate = F.silu(self.w1(x))  # Swish = SiLU
        x = swish_gate * self.w2(x)  # Gating
        x = self.w3(x)  # Down projection
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """Feed-forward network with ReLU² activation (alternative to SwiGLU)"""

    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd

        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU² activation (from speedrun)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-normalization"""

    def __init__(self, config: ModernGPTConfig, use_swiglu: bool = True):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)

        # Choose between SwiGLU or ReLU² MLP
        if use_swiglu:
            self.mlp = SwiGLU(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        # Pre-norm architecture (modern standard)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ModernGPT(nn.Module):
    """
    Modern Fast GPT with state-of-the-art optimizations.

    Key features:
    - RoPE (Rotary Position Embeddings)
    - RMSNorm for stability
    - Flash Attention for efficiency
    - QK-Norm for better training dynamics
    - SwiGLU or ReLU² activation
    - Pre-norm architecture
    """

    def __init__(self, config: ModernGPTConfig, use_swiglu: bool = True):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Token embeddings (no positional embeddings - using RoPE instead)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, use_swiglu) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (share embeddings with output layer)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print("ModernGPT model initialized")
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=False):
        """Return the number of parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights with modern best practices"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass

        Args:
            idx: input token indices (B, T)
            targets: target token indices for loss calculation (B, T)

        Returns:
            logits: output logits (B, T, vocab_size)
            loss: cross-entropy loss if targets provided, else None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = self.transformer.drop(tok_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Language model head
        if targets is not None:
            # Only compute logits for positions we need (efficient)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            # Inference: only need last position
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    def full_loss(self, idx, with_grad=False):
        """
        Compute loss with optional gradient computation.
        Compatible with compressor.py interface.

        Args:
            idx: input sequence (B, T)
            with_grad: whether to compute gradients

        Returns:
            loss: scalar loss value
        """
        # Create input and targets
        x = idx[:, :-1]
        y = idx[:, 1:]

        # Forward pass
        if with_grad:
            logits, loss = self.forward(x, y)
        else:
            with torch.no_grad():
                logits, loss = self.forward(x, y)

        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively

        Args:
            idx: conditioning sequence (B, T)
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top k tokens

        Returns:
            generated sequence (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self.forward(idx_cond)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Example configurations
def get_config_small():
    """Small configuration for fast training/testing"""
    return ModernGPTConfig(
        block_size=256,
        vocab_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
    )


def get_config_medium():
    """Medium configuration (similar to GPT-2 small)"""
    return ModernGPTConfig(
        block_size=1024,
        vocab_size=256,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False,
    )


def get_config_large():
    """Large configuration for better performance"""
    return ModernGPTConfig(
        block_size=1024,
        vocab_size=256,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        dropout=0.0,
        bias=False,
    )


if __name__ == "__main__":
    # Test the model
    config = get_config_small()
    model = ModernGPT(config, use_swiglu=True)

    # Create dummy input
    x = torch.randint(0, config.vocab_size, (2, 64))

    # Forward pass
    logits, loss = model.forward(x[:, :-1], x[:, 1:])
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

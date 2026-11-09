"""Minimal decoder-only transformer shared by the hand-written and trained models.

No LayerNorm, ReLU MLPs, separate per-head Q/K/V/O matrices, causal mask,
scores scaled by 1/sqrt(d_head) (hand-written weights compensate for this).
Each layer may have its own number of heads and an optional MLP; the
hand-written and trained versions of a task always use the identical config.
"""
import math
import torch
import torch.nn as nn


class Attn(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        self.W_Q = nn.Parameter(torch.zeros(n_heads, d_model, d_head))
        self.W_K = nn.Parameter(torch.zeros(n_heads, d_model, d_head))
        self.W_V = nn.Parameter(torch.zeros(n_heads, d_model, d_head))
        self.W_O = nn.Parameter(torch.zeros(n_heads, d_head, d_model))
        for p in (self.W_Q, self.W_K, self.W_V):
            nn.init.normal_(p, std=1 / math.sqrt(d_model))
        nn.init.normal_(self.W_O, std=1 / math.sqrt(n_heads * d_head))

    def forward(self, x, return_pattern=False):
        T = x.shape[1]
        q = torch.einsum("btd,hde->bhte", x, self.W_Q)
        k = torch.einsum("btd,hde->bhte", x, self.W_K)
        v = torch.einsum("btd,hde->bhte", x, self.W_V)
        s = torch.einsum("bhte,bhse->bhts", q, k) / math.sqrt(self.d_head)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
        a = s.softmax(-1)
        z = torch.einsum("bhts,bhse->bhte", a, v)
        out = torch.einsum("bhte,hed->btd", z, self.W_O)
        return (out, a) if return_pattern else out


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_mlp))
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return torch.relu(x @ self.W_in + self.b_in) @ self.W_out + self.b_out


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_mlp):
        super().__init__()
        self.attn = Attn(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_mlp) if d_mlp else None

    def forward(self, x):
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    """layers: list of dicts {n_heads, d_head, d_mlp}."""

    def __init__(self, vocab, d_model, max_len, layers):
        super().__init__()
        self.cfg = dict(vocab=vocab, d_model=d_model, max_len=max_len, layers=layers)
        self.tok = nn.Parameter(torch.randn(vocab, d_model) * 0.5)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
        self.blocks = nn.ModuleList(
            Block(d_model, l["n_heads"], l["d_head"], l["d_mlp"]) for l in layers
        )
        self.W_U = nn.Parameter(torch.randn(d_model, vocab) / math.sqrt(d_model))
        self.b_U = nn.Parameter(torch.zeros(vocab))

    def residual(self, idx, upto=None):
        x = self.tok[idx] + self.pos[: idx.shape[1]]
        for i, b in enumerate(self.blocks):
            if upto is not None and i == upto:
                return x
            x = b(x)
        return x

    def forward(self, idx):
        return self.residual(idx) @ self.W_U + self.b_U

    def zero_(self):
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()
        return self


def n_params(m):
    return sum(p.numel() for p in m.parameters())

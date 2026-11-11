"""model.py: a small GPT with RoPE, RMSNorm and a SwiGLU MLP (31,465,984 parameters)."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    block_size: int = 512  # context length in tokens
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512  # numbers per token


def rope_tables(cfg, base=10000):
    hd = cfg.d_model // cfg.n_head  # 64 numbers per head
    speeds = base ** (-torch.arange(0, hd, 2) / hd)  # 32 rotation speeds
    pos = torch.arange(cfg.block_size)
    angles = torch.outer(pos, speeds)  # (512, 32)
    return angles.cos(), angles.sin()


def rope(x, cos, sin):  # x: (B, heads, T, 64)
    x1, x2 = x[..., 0::2], x[..., 1::2]  # 32 pairs per head
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape  # (32, 512, 512)
        q, k, v = self.qkv(x).split(C, dim=-1)
        # 8 heads of 64 numbers each: (B, 8, T, 64)
        q, k, v = [t.view(B, T, self.n_head, -1).transpose(1, 2) for t in (q, k, v)]
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        # same as: softmax(q @ k^T / sqrt(64) + causal mask) @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)  # merge the heads
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg.d_model, 3 * cfg.d_model  # 512 -> 1536 -> 512
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)  # tokens share context
        x = x + self.mlp(self.norm2(x))  # each token updates itself
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # (32, 512) token ids
        x = self.tok_emb(idx)  # (32, 512, 512)
        cos, sin = self.cos[:T], self.sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.head(self.norm(x))  # (32, 512, 8192)
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )


@torch.no_grad()
def generate(model, idx, max_new, eot, temperature=0.8, top_k=50):
    for _ in range(max_new):
        ctx = idx[:, -model.cfg.block_size :]  # last 512 tokens
        logits = model(ctx)[:, -1, :] / temperature  # only the last position
        kth = torch.topk(logits, top_k).values[:, [-1]]
        logits[logits < kth] = float("-inf")  # keep the 50 most likely
        probs = logits.softmax(dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
        if nxt.item() == eot:
            break
    return idx

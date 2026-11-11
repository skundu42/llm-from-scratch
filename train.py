"""train.py: pretrain the model on TinyStories (20,000 steps x 32,768 tokens)."""
import math

import numpy as np
import torch

from model import GPT, Config

device = "cuda" if torch.cuda.is_available() else "mps"
torch.manual_seed(0)
cfg = Config()
raw = GPT(cfg).to(device)
print(f"{sum(p.numel() for p in raw.parameters()):,} parameters")  # 31,465,984
model = torch.compile(raw) if device == "cuda" else raw  # fuse small ops into fewer kernels

B, T = 32, cfg.block_size
data = {s: np.memmap(f"{s}.bin", dtype=np.uint16, mode="r") for s in ["train", "validation"]}


def get_batch(split):
    d = data[split]
    ix = torch.randint(len(d) - T - 1, (B,)).tolist()
    buf = torch.stack([torch.from_numpy(d[i : i + T + 1].astype(np.int64)) for i in ix])
    x, y = buf[:, :-1], buf[:, 1:]  # y is x shifted left by one
    return x.to(device), y.to(device)


max_steps, warmup = 20_000, 1_000
max_lr, min_lr = 1e-3, 1e-4


def lr_at(step):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    p = (step - warmup) / (max_steps - warmup)  # 0 -> 1
    cos = 0.5 * (1 + math.cos(math.pi * p))  # 1 -> 0
    return min_lr + (max_lr - min_lr) * cos


decay = [p for p in model.parameters() if p.dim() >= 2]  # matrices
other = [p for p in model.parameters() if p.dim() < 2]  # norm gains
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
    lr=max_lr,
    betas=(0.9, 0.95),
)


@torch.no_grad()
def val_loss(n=50):
    model.eval()
    total = 0.0
    for _ in range(n):
        x, y = get_batch("validation")
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            total += model(x, y).item()
    model.train()
    return total / n


accum = 2  # 2 x 32 x 512 = 32,768 tokens per optimizer step
for step in range(max_steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    for _ in range(accum):
        x, y = get_batch("train")
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = model(x, y) / accum
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 500 == 0:
        print(f"step {step}  train {loss.item() * accum:.3f}  val {val_loss():.3f}")

torch.save(raw.state_dict(), "storygpt.pt")  # raw: no torch.compile key prefix

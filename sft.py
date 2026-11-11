"""sft.py: teach the story model to follow instructions (TinyStoriesInstruct), then chat."""
import random

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from model import GPT, Config, generate

device = "cuda" if torch.cuda.is_available() else "mps"
tok = Tokenizer.from_file("tokenizer.json")
U, A, EOT = [tok.token_to_id(s) for s in ["<|user|>", "<|assistant|>", "<|endoftext|>"]]

# TinyStoriesInstruct: instruction lines (Features/Words/Summary), then "Story:", then the story
lines = load_dataset("roneneldan/TinyStoriesInstruct", split="train[:5%]")["text"]
records = "\n".join(lines).split("<|endoftext|>")
pairs = [r.split("Story:", 1) for r in records if "Story:" in r]


def example(prompt, story):
    p = [U] + tok.encode(prompt.strip()).ids + [A]
    r = tok.encode(story.strip()).ids + [EOT]
    ids, labels = p + r, [-1] * len(p) + r  # no loss on the prompt
    return ids[:-1], labels[1:]  # shift by one, as in pretraining


examples = [example(p, s) for p, s in pairs]


def batch(exs, T=512):
    T = min(T, max(len(x) for x, _ in exs))
    X = torch.full((len(exs), T), EOT)
    Y = torch.full((len(exs), T), -1)  # padding is never graded
    for i, (x, y) in enumerate(exs):
        x, y = x[:T], y[:T]
        X[i, : len(x)] = torch.tensor(x)
        Y[i, : len(y)] = torch.tensor(y)
    return X.to(device), Y.to(device)


model = GPT(Config()).to(device)
model.load_state_dict(torch.load("storygpt.pt", map_location=device))
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)  # 10x lower
random.seed(0)
for step in range(2_000):
    x, y = batch(random.sample(examples, 32))
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 200 == 0:
        print(f"step {step}  loss {loss.item():.3f}")
torch.save(model.state_dict(), "storygpt-instruct.pt")

# Talk to it
model.eval()
prompt = "Words: dragon, cake, brave\nFeatures: Dialogue"
ids = [U] + tok.encode(prompt).ids + [A]
out = generate(model, torch.tensor([ids], device=device), 400, eot=EOT)
print(tok.decode(out[0, len(ids) :].tolist()))

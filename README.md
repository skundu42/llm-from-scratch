# Build an LLM From Scratch

The complete code from the video **"Build an LLM From Scratch in 30 Minutes"**: a small GPT-style language model, built in four files and 236 lines of PyTorch. It is trained on children's stories, then fine-tuned to follow instructions.

| File | What it does |
|---|---|
| `prepare.py` | Downloads [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), trains an 8,192-token byte-level BPE tokenizer and writes the tokens to `train.bin` / `validation.bin` |
| `model.py` | The model: token embeddings, 8 transformer blocks (RoPE attention, RMSNorm, SwiGLU MLP), tied output head, plus `generate()` |
| `train.py` | Pretraining: 20,000 steps x 32,768 tokens, AdamW, warmup + cosine learning rate, saves `storygpt.pt` |
| `sft.py` | Instruction tuning on [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct), saves `storygpt-instruct.pt`, then asks it for a story |

## The model

| | |
|---|---|
| Parameters | 31,465,984 |
| Layers | 8 |
| Model width | 512 |
| Attention heads | 8 (64 dims each) |
| Context length | 512 tokens |
| MLP | SwiGLU, 512 -> 1536 -> 512 |
| Vocabulary | 8,192 (byte-level BPE) |
| Pretraining tokens | 20,000 steps x 32,768 = ~655M |

## Run it

You need an NVIDIA GPU (CUDA) or an Apple Silicon Mac (MPS), and Python 3.10+.

```bash
pip install -r requirements.txt

python prepare.py   # 1. data + tokenizer
python train.py     # 2. pretrain the base model
python sft.py       # 3. instruction-tune it and generate a story
```

Run the scripts in order from this folder. Each one reads the files the previous one wrote.

## Going further

The papers and datasets mentioned in the video:

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) for pretraining on general web text
- [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) for chat-style fine-tuning

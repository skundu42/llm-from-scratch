"""prepare.py: download TinyStories, train a BPE tokenizer, write token files."""
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ds = load_dataset("roneneldan/TinyStories")  # 2.12M train, 22k validation stories

# 1. Byte-level BPE tokenizer with an 8,192-token vocabulary
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=8192,  # 256 bytes + 3 special tokens + 7,933 merges
    special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)


def text_batches(split, n=1000):
    for i in range(0, len(ds[split]), n):
        yield ds[split][i : i + n]["text"]


tok.train_from_iterator(text_batches("train"), trainer=trainer)
tok.save("tokenizer.json")
EOT = tok.token_to_id("<|endoftext|>")


# 2. Encode every story, end each one with <|endoftext|>, save as uint16
def encode(batch):
    enc = tok.encode_batch(batch["text"])
    return {"ids": [e.ids + [EOT] for e in enc]}


for split in ["train", "validation"]:
    rows = ds[split].map(encode, batched=True, remove_columns=["text"])
    ids = np.fromiter((t for r in rows["ids"] for t in r), dtype=np.uint16)
    ids.tofile(f"{split}.bin")  # ids < 65,536, so 2 bytes each: ~0.9 GB for train

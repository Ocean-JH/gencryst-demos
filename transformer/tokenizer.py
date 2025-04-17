#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from collections import Counter


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

class Tokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, "r") as f:
            self.token2id = json.load(f)
        self.id2token = {i: t for t, i in self.token2id.items()}

    def encode(self, tokens):
        return [self.token2id.get(t, self.token2id["<unk>"]) for t in tokens]

    def decode(self, ids):
        return [self.id2token.get(i, "<unk>") for i in ids]

    def vocab_size(self):
        return len(self.token2id)


def build_vocab(jsonl_file, vocab_file="vocab.json", min_freq=1):
    counter = Counter()
    with open(jsonl_file, "r") as f:
        for line in f:
            tokens = json.loads(line.strip())
            counter.update(tokens)

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    idx = len(vocab)

    for token, count in counter.items():
        if token not in vocab and count >= min_freq:
            vocab[token] = idx
            idx += 1

    with open(vocab_file, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"[vocab] Saved to {vocab_file} ({len(vocab)} tokens)")
    return vocab

if __name__ == "__main__":
    jsonl_file = "tokens.jsonl"
    vocab_file = "vocab.json"
    min_freq = 1
    build_vocab(jsonl_file, vocab_file, min_freq)


    vocab_file = "vocab.json"
    tokenizer = Tokenizer(vocab_file)

    # 测试编码和解码
    tokens = ["<bos>", "[SG#225]", "(Te,a,0.005,0.087,0.418)", "<eos>"]
    ids = tokenizer.encode(tokens)
    print(f"Encoded: {ids}")

    decoded_tokens = tokenizer.decode(ids)
    print(f"Decoded: {decoded_tokens}")

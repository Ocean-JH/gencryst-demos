#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from evaluation import check_wyckoff_validity, check_atomic_clashes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def generate_structure(model, tokenizer, max_len=64, temperature=1.0, device=device):
    model.eval()
    bos = tokenizer.token2id["<bos>"]
    eos = tokenizer.token2id["<eos>"]

    tokens = [bos]
    for _ in range(max_len):
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # shape (1, T)
        logits = model(x)  # (1, T, V)
        next_token_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        tokens.append(next_token)
        if next_token == eos:
            break

    return tokens


if __name__ == "__main__":
    from tokenizer import Tokenizer
    from mini_crystalformer import MiniCrystalFormer
    # Load the model and tokenizer
    tokenizer = Tokenizer("vocab.json")
    model = MiniCrystalFormer(vocab_size=tokenizer.vocab_size()).to(device)

    # Generate a structure
    generated_tokens = generate_structure(model, tokenizer)
    generated_structure = tokenizer.decode(generated_tokens)

    # Check Wyckoff validity and atomic clashes
    # is_valid_wyckoff = check_wyckoff_validity(generated_structure, target_sgnum=225)
    # has_clashes = check_atomic_clashes(generated_structure)

    print(f"Generated Structure: {generated_structure}")
    # print(f"Wyckoff Validity: {is_valid_wyckoff}")
    # print(f"Atomic Clashes: {has_clashes}")
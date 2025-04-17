#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from tokenizer import Tokenizer, build_vocab
from data_loader import dataloader, tokens_to_structure
from mini_crystalformer import MiniCrystalFormer
from train import train_model
from generate import generate_structure
from pymatgen.io.cif import CifWriter

import torch

def main_pipeline(
    data_dir="data/",
    vocab_path="vocab.json",
    save_dir="logs/",
    batch_size=32,
    epochs=10,
    n_generate=20,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(save_dir, exist_ok=True)

    print("📦 Step 1: Load data & construct vocabulary")
    dataloader(data_dir)
    build_vocab("tokens.jsonl", vocab_path, min_freq=1)
    tokenizer = Tokenizer("vocab.json")

    print("📚 Step 2: Create model")
    model = MiniCrystalFormer(vocab_size=tokenizer.vocab_size()).to(device)

    print("🎯 Step 3: Train model")
    train_model(model, tokenizer, device=device, batch_size=batch_size, epochs=epochs)

    print("🎉 Step 4: Generate structures")
    model.load_state_dict(torch.load(f"logs/model_epoch{epochs}.pt", map_location=device))
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    for i in range(n_generate):
        token_ids = generate_structure(model, tokenizer, max_len=64, temperature=1.0)
        tokens = tokenizer.decode(token_ids)
        structure = tokens_to_structure(tokens)
        if structure:
            filename = os.path.join(save_dir, f"struct_{i:03d}.cif")
            CifWriter(structure).write_file(filename)
            print(f"✅ Saved: {filename}")
        else:
            print(f"❌ Failed to generate valid structure #{i}")

    print("✅ Pipeline Done！")

if __name__ == "__main__":
    main_pipeline()

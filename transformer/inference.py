#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tokenizer import Tokenizer
from mini_crystalformer import MiniCrystalFormer
from generate import generate_structure
import torch

# 加载
tokenizer = Tokenizer("vocab.json")
model = MiniCrystalFormer(vocab_size=tokenizer.vocab_size())
model.load_state_dict(torch.load("logs/model_epoch10.pt", map_location="cpu"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# 生成结构
token_ids = generate_structure(model, tokenizer, max_len=64, temperature=1.0)
tokens = tokenizer.decode(token_ids)
print("Generated structure tokens:")
print(tokens)

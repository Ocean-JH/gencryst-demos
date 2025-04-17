#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset

class CrystalDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, "r") as f:
            for line in f:
                tokens = json.loads(line.strip())
                ids = tokenizer.encode(tokens)
                if len(ids) <= max_len:
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)  # input
        y = torch.tensor(ids[1:], dtype=torch.long)   # target (next token)
        return x, y

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    max_len = max([len(x) for x in x_batch])
    pad_token = 0  # assuming <pad> = 0

    x_pad = [torch.cat([x, torch.full((max_len - len(x),), pad_token)]) for x in x_batch]
    y_pad = [torch.cat([y, torch.full((max_len - len(y),), -100)]) for y in y_batch]  # -100 用于忽略 loss

    x = torch.stack(x_pad).long()
    y = torch.stack(y_pad).long()
    return x, y

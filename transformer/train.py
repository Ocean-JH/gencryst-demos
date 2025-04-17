#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from dataset import CrystalDataset, collate_fn

def train_model(model, tokenizer, device, batch_size=32, epochs=10):

    # 初始化
    dataset = CrystalDataset("tokens.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss(ignore_index=-100)

    # 训练 loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)  # (B, T, V)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")

        # 保存模型
        torch.save(model.state_dict(), f"logs/model_epoch{epoch+1}.pt")

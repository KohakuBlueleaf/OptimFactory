"""Minimal OptimFactory example.

Trains a tiny MLP on random data to demonstrate:
- µP initialization
- µP-scaled param groups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from optimfactory import mup_init, mup_init_output, mup_param_group


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(32, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    mup_init(model.parameters())
    mup_init_output(model[-1].weight)

    groups = mup_param_group(
        model.parameters(),
        base_lr=1e-3,
        base_dim=128,
        weight_decay=0.01,
        weight_decay_scale=True,
    )
    optimizer = optim.AdamW(groups, betas=(0.9, 0.98))

    for step in range(200):
        x = torch.randn(256, 32, device=device)
        y = torch.randint(0, 10, (256,), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

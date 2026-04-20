from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] = [256, 256, 256],
        output_size: int = 10,
    ) -> None:
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.linears = nn.ModuleList()
        prev = input_size
        for h in hidden_sizes:
            self.linears.append(nn.Linear(prev, h))
            prev = h
        self.output_layer = nn.Linear(prev, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for linear in self.linears:
            h = torch.relu(linear(h))
        return self.output_layer(h)

    def get_layer_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return post-ReLU activations at each hidden layer (detached)."""
        acts = []
        h = x
        with torch.no_grad():
            for linear in self.linears:
                h = torch.relu(linear(h))
                acts.append(h.detach())
        return acts

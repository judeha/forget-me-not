from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MaskedMLP(nn.Module):
    """MLP with per-task learnable soft masks on hidden activations.

    h_l = relu(W_l h_{l-1} + b_l) * sigmoid(beta * alpha_l^(t))

    Mask alphas are stored per task in task_alphas (nn.ModuleDict).
    Past-task alphas are frozen after each task completes.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] = [256, 256, 256],
        output_size: int = 10,
        beta: float = 5.0,
    ) -> None:
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.beta = beta

        self.linears = nn.ModuleList()
        prev = input_size
        for h in hidden_sizes:
            self.linears.append(nn.Linear(prev, h))
            prev = h
        self.output_layer = nn.Linear(prev, output_size)

        # task_alphas[str(task_id)] -> ParameterList of alpha vectors (one per hidden layer)
        self.task_alphas: nn.ModuleDict = nn.ModuleDict()
        self._current_task: int = 0

    @property
    def n_mask_layers(self) -> int:
        return len(self.hidden_sizes)

    def add_task(self, task_id: int, device: torch.device) -> None:
        key = str(task_id)
        if key not in self.task_alphas:
            self.task_alphas[key] = nn.ParameterList([
                nn.Parameter(torch.zeros(h, device=device))
                for h in self.hidden_sizes
            ])

    def masks(self, task_id: int) -> list[torch.Tensor]:
        """Return sigmoid mask vectors for task_id; shape [h_l] per layer."""
        return [torch.sigmoid(self.beta * a) for a in self.task_alphas[str(task_id)]]

    def freeze_task(self, task_id: int) -> None:
        for a in self.task_alphas[str(task_id)]:
            a.requires_grad_(False)

    def task_params(self, task_id: int) -> list[nn.Parameter]:
        return list(self.task_alphas[str(task_id)])

    def backbone_params(self) -> list[nn.Parameter]:
        return list(self.linears.parameters()) + list(self.output_layer.parameters())

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        if task_id is None:
            task_id = self._current_task
        masks = self.masks(task_id)
        h = x
        for i, linear in enumerate(self.linears):
            h = torch.relu(linear(h)) * masks[i]
        return self.output_layer(h)

    def get_layer_activations(
        self, x: torch.Tensor, task_id: Optional[int] = None
    ) -> list[torch.Tensor]:
        """Return post-mask activations at each hidden layer (detached)."""
        if task_id is None:
            task_id = self._current_task
        masks = self.masks(task_id)
        acts = []
        h = x
        for i, linear in enumerate(self.linears):
            h = torch.relu(linear(h)) * masks[i]
            acts.append(h.detach())
        return acts

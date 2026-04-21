from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadMLP(nn.Module):
    """Shared backbone MLP with per-task output heads (task-incremental)."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] = [256, 256, 256],
    ) -> None:
        super().__init__()
        self.hidden_sizes = hidden_sizes
        sizes = [input_size] + list(hidden_sizes)
        self.linears = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(hidden_sizes))
        ])
        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self._current_task: int = 0

    def add_task_head(self, task_id: int, n_classes: int, device: torch.device) -> None:
        key = str(task_id)
        if key not in self.task_heads:
            self.task_heads[key] = nn.Linear(self.hidden_sizes[-1], n_classes).to(device)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        h = x.flatten(1)
        for linear in self.linears:
            h = F.relu(linear(h))
        return h

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        if task_id is None:
            task_id = self._current_task
        return self.task_heads[str(task_id)](self._backbone(x))

    def get_layer_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        acts = []
        h = x.flatten(1)
        with torch.no_grad():
            for linear in self.linears:
                h = F.relu(linear(h))
                acts.append(h.detach())
        return acts


class MaskedMultiHeadMLP(nn.Module):
    """Masked backbone MLP with per-task heads + gradient masking.

    Per-task sigmoid masks gate hidden activations (same as MaskedMLP).
    After each task, update_ownership() computes which neurons are claimed
    by past tasks.  apply_gradient_mask() then zeros gradients for those
    neurons on subsequent backward passes, preventing weight overwriting.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] = [256, 256, 256],
        beta: float = 5.0,
    ) -> None:
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.beta = beta
        sizes = [input_size] + list(hidden_sizes)
        self.linears = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(hidden_sizes))
        ])
        self.task_alphas: nn.ModuleDict = nn.ModuleDict()
        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self._current_task: int = 0
        self._ownership: Optional[list[torch.Tensor]] = None

    @property
    def n_mask_layers(self) -> int:
        return len(self.hidden_sizes)

    def add_task(self, task_id: int, n_classes: int, device: torch.device) -> None:
        key = str(task_id)
        if key not in self.task_alphas:
            self.task_alphas[key] = nn.ParameterList([
                nn.Parameter(torch.zeros(h, device=device))
                for h in self.hidden_sizes
            ])
        if key not in self.task_heads:
            self.task_heads[key] = nn.Linear(self.hidden_sizes[-1], n_classes).to(device)

    def masks(self, task_id: int) -> list[torch.Tensor]:
        return [torch.sigmoid(self.beta * a) for a in self.task_alphas[str(task_id)]]

    def freeze_task(self, task_id: int) -> None:
        for a in self.task_alphas[str(task_id)]:
            a.requires_grad_(False)

    def update_ownership(self) -> None:
        """Recompute per-layer ownership as element-wise max mask across all frozen tasks."""
        frozen = [k for k, pl in self.task_alphas.items() if not pl[0].requires_grad]
        if not frozen:
            self._ownership = None
            return
        self._ownership = [
            torch.stack([
                torch.sigmoid(self.beta * self.task_alphas[k][l]).detach()
                for k in frozen
            ]).max(0).values
            for l in range(len(self.hidden_sizes))
        ]

    def apply_gradient_mask(self, threshold: float = 0.5) -> None:
        """Zero gradients for neurons whose ownership exceeds threshold."""
        if self._ownership is None:
            return
        for l, linear in enumerate(self.linears):
            owned = self._ownership[l] > threshold
            if linear.weight.grad is not None:
                linear.weight.grad.data[owned] = 0.0
            if linear.bias is not None and linear.bias.grad is not None:
                linear.bias.grad.data[owned] = 0.0

    def backbone_params(self) -> list[nn.Parameter]:
        return list(self.linears.parameters())

    def task_params(self, task_id: int) -> list[nn.Parameter]:
        key = str(task_id)
        return (list(self.task_alphas[key].parameters())
                + list(self.task_heads[key].parameters()))

    def _backbone(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        masks = self.masks(task_id)
        h = x.flatten(1)
        for i, linear in enumerate(self.linears):
            h = F.relu(linear(h)) * masks[i]
        return h

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        if task_id is None:
            task_id = self._current_task
        return self.task_heads[str(task_id)](self._backbone(x, task_id))

    def get_layer_activations(
        self, x: torch.Tensor, task_id: Optional[int] = None
    ) -> list[torch.Tensor]:
        if task_id is None:
            task_id = self._current_task
        masks = self.masks(task_id)
        acts = []
        h = x.flatten(1)
        with torch.no_grad():
            for i, linear in enumerate(self.linears):
                h = F.relu(linear(h)) * masks[i]
                acts.append(h.detach())
        return acts

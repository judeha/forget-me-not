from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MultiHeadCNN(nn.Module):
    """CNN backbone with per-task output heads (task-incremental)."""

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: list[int] = [32, 64, 128],
        input_hw: int = 32,
    ) -> None:
        super().__init__()
        self.conv_channels = conv_channels
        self.conv_blocks = nn.ModuleList()
        prev = in_channels
        for ch in conv_channels:
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(prev, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))
            prev = ch
        spatial = input_hw // (2 ** len(conv_channels))
        self._flat_size = conv_channels[-1] * spatial * spatial
        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self._current_task: int = 0

    def add_task_head(self, task_id: int, n_classes: int, device: torch.device) -> None:
        key = str(task_id)
        if key not in self.task_heads:
            self.task_heads[key] = nn.Linear(self._flat_size, n_classes).to(device)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.conv_blocks:
            h = block(h)
        return h.flatten(1)

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        if task_id is None:
            task_id = self._current_task
        return self.task_heads[str(task_id)](self._backbone(x))

    def get_layer_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        acts = []
        h = x
        with torch.no_grad():
            for block in self.conv_blocks:
                h = block(h)
                acts.append(h.mean(dim=(2, 3)).detach())
        return acts


class MaskedMultiHeadCNN(nn.Module):
    """Masked CNN backbone with per-task heads + gradient masking.

    Channel masks gate conv activations per task; gradient masking zeros
    gradients for channels claimed by past tasks after ownership update.
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: list[int] = [32, 64, 128],
        input_hw: int = 32,
        beta: float = 5.0,
    ) -> None:
        super().__init__()
        self.conv_channels = conv_channels
        self.beta = beta

        self.conv_blocks = nn.ModuleList()
        prev = in_channels
        for ch in conv_channels:
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(prev, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))
            prev = ch
        spatial = input_hw // (2 ** len(conv_channels))
        self._flat_size = conv_channels[-1] * spatial * spatial

        self.task_alphas: nn.ModuleDict = nn.ModuleDict()
        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self._current_task: int = 0
        self._ownership: Optional[list[torch.Tensor]] = None

    @property
    def n_mask_layers(self) -> int:
        return len(self.conv_channels)

    def add_task(self, task_id: int, n_classes: int, device: torch.device) -> None:
        key = str(task_id)
        if key not in self.task_alphas:
            self.task_alphas[key] = nn.ParameterList([
                nn.Parameter(torch.zeros(ch, device=device))
                for ch in self.conv_channels
            ])
        if key not in self.task_heads:
            self.task_heads[key] = nn.Linear(self._flat_size, n_classes).to(device)

    def masks(self, task_id: int) -> list[torch.Tensor]:
        return [torch.sigmoid(self.beta * a) for a in self.task_alphas[str(task_id)]]

    def freeze_task(self, task_id: int) -> None:
        for a in self.task_alphas[str(task_id)]:
            a.requires_grad_(False)

    def update_ownership(self) -> None:
        frozen = [k for k, pl in self.task_alphas.items() if not pl[0].requires_grad]
        if not frozen:
            self._ownership = None
            return
        self._ownership = [
            torch.stack([
                torch.sigmoid(self.beta * self.task_alphas[k][l]).detach()
                for k in frozen
            ]).max(0).values
            for l in range(len(self.conv_channels))
        ]

    def apply_gradient_mask(self, threshold: float = 0.5) -> None:
        """Zero gradients for channels owned by past tasks in each conv block."""
        if self._ownership is None:
            return
        for l, block in enumerate(self.conv_blocks):
            owned = self._ownership[l] > threshold  # (C_l,) bool
            conv = block[0]  # nn.Conv2d is first in Sequential
            bn   = block[1]  # nn.BatchNorm2d is second
            if conv.weight.grad is not None:
                conv.weight.grad.data[owned] = 0.0
            if conv.bias is not None and conv.bias.grad is not None:
                conv.bias.grad.data[owned] = 0.0
            if bn.weight.grad is not None:
                bn.weight.grad.data[owned] = 0.0
            if bn.bias.grad is not None:
                bn.bias.grad.data[owned] = 0.0

    def backbone_params(self) -> list[nn.Parameter]:
        return list(self.conv_blocks.parameters())

    def task_params(self, task_id: int) -> list[nn.Parameter]:
        key = str(task_id)
        return (list(self.task_alphas[key].parameters())
                + list(self.task_heads[key].parameters()))

    def _backbone(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        masks = self.masks(task_id)
        h = x
        for i, block in enumerate(self.conv_blocks):
            h = block(h) * masks[i].view(1, -1, 1, 1)
        return h.flatten(1)

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
        h = x
        with torch.no_grad():
            for i, block in enumerate(self.conv_blocks):
                h = block(h) * masks[i].view(1, -1, 1, 1)
                acts.append(h.mean(dim=(2, 3)).detach())
        return acts

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CNN(nn.Module):
    """Small CNN: 3 conv blocks + FC head, for CIFAR-100."""

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: list[int] = [32, 64, 128],
        output_size: int = 100,
        input_hw: int = 32,
    ) -> None:
        super().__init__()
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
        self.fc = nn.Linear(conv_channels[-1] * spatial * spatial, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.conv_blocks:
            h = block(h)
        return self.fc(h.flatten(1))

    def get_layer_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return global-avg-pooled activations at each conv block (detached)."""
        acts = []
        h = x
        with torch.no_grad():
            for block in self.conv_blocks:
                h = block(h)
                acts.append(h.mean(dim=(2, 3)).detach())
        return acts


class MaskedCNN(nn.Module):
    """CNN with per-task learnable channel masks on conv activations.

    h_l = conv_block_l(h_{l-1}) * sigmoid(beta * alpha_l^(t)).view(1, C_l, 1, 1)
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: list[int] = [32, 64, 128],
        output_size: int = 100,
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
        self.fc = nn.Linear(conv_channels[-1] * spatial * spatial, output_size)

        self.task_alphas: nn.ModuleDict = nn.ModuleDict()
        self._current_task: int = 0

    @property
    def n_mask_layers(self) -> int:
        return len(self.conv_channels)

    def add_task(self, task_id: int, device: torch.device) -> None:
        key = str(task_id)
        if key not in self.task_alphas:
            self.task_alphas[key] = nn.ParameterList([
                nn.Parameter(torch.zeros(ch, device=device))
                for ch in self.conv_channels
            ])

    def masks(self, task_id: int) -> list[torch.Tensor]:
        return [torch.sigmoid(self.beta * a) for a in self.task_alphas[str(task_id)]]

    def freeze_task(self, task_id: int) -> None:
        for a in self.task_alphas[str(task_id)]:
            a.requires_grad_(False)

    def task_params(self, task_id: int) -> list[nn.Parameter]:
        return list(self.task_alphas[str(task_id)])

    def backbone_params(self) -> list[nn.Parameter]:
        return list(self.conv_blocks.parameters()) + list(self.fc.parameters())

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        if task_id is None:
            task_id = self._current_task
        masks = self.masks(task_id)
        h = x
        for i, block in enumerate(self.conv_blocks):
            h = block(h) * masks[i].view(1, -1, 1, 1)
        return self.fc(h.flatten(1))

    def get_layer_activations(
        self, x: torch.Tensor, task_id: Optional[int] = None
    ) -> list[torch.Tensor]:
        """Return global-avg-pooled post-mask activations at each conv block (detached)."""
        if task_id is None:
            task_id = self._current_task
        masks = self.masks(task_id)
        acts = []
        h = x
        for i, block in enumerate(self.conv_blocks):
            h = block(h) * masks[i].view(1, -1, 1, 1)
            acts.append(h.mean(dim=(2, 3)).detach())
        return acts

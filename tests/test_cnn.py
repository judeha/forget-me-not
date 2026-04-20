import torch
import pytest

from src.models.cnn import CNN, MaskedCNN
from src.utils.seed import set_seed


@pytest.fixture
def dummy_batch():
    set_seed(0)
    return torch.randn(4, 3, 32, 32)


def test_cnn_forward_shape(dummy_batch):
    model = CNN(in_channels=3, conv_channels=[8, 16], output_size=10, input_hw=32)
    out = model(dummy_batch)
    assert out.shape == (4, 10)


def test_masked_cnn_forward_shape(dummy_batch):
    model = MaskedCNN(in_channels=3, conv_channels=[8, 16], output_size=10, input_hw=32)
    model.add_task(0, torch.device("cpu"))
    model._current_task = 0
    out = model(dummy_batch)
    assert out.shape == (4, 10)


def test_masked_cnn_freeze(dummy_batch):
    model = MaskedCNN(in_channels=3, conv_channels=[8, 16], output_size=10, input_hw=32)
    model.add_task(0, torch.device("cpu"))
    model.freeze_task(0)
    for a in model.task_alphas["0"]:
        assert not a.requires_grad


def test_masked_cnn_layer_activations(dummy_batch):
    model = MaskedCNN(in_channels=3, conv_channels=[8, 16], output_size=10, input_hw=32)
    model.add_task(0, torch.device("cpu"))
    acts = model.get_layer_activations(dummy_batch, task_id=0)
    assert len(acts) == 2  # one per conv block
    assert acts[0].shape == (4, 8)   # [B, C] after global avg pool
    assert acts[1].shape == (4, 16)


def test_cnn_layer_activations(dummy_batch):
    model = CNN(in_channels=3, conv_channels=[8, 16], output_size=10, input_hw=32)
    acts = model.get_layer_activations(dummy_batch)
    assert len(acts) == 2
    assert acts[0].shape[0] == 4

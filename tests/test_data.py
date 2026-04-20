import pytest
import torch
from src.data.split_mnist import get_split_mnist, TASK_CLASSES


def test_n_tasks():
    tasks = get_split_mnist(data_dir="data/mnist", batch_size=256, subset_size=50)
    assert len(tasks) == 5


def test_task_classes_disjoint():
    all_classes = [c for pair in TASK_CLASSES for c in pair]
    assert len(all_classes) == len(set(all_classes)), "Task classes must be disjoint"


def test_task_labels_correct():
    tasks = get_split_mnist(data_dir="data/mnist", batch_size=256, subset_size=50)
    for task in tasks:
        for x, y in task["train"]:
            labels = set(y.tolist())
            assert labels.issubset(set(task["classes"])), (
                f"Expected only classes {task['classes']}, got {labels}"
            )
            break  # check first batch only


def test_input_shape():
    tasks = get_split_mnist(data_dir="data/mnist", batch_size=32, subset_size=32)
    x, y = next(iter(tasks[0]["train"]))
    assert x.shape[1] == 784

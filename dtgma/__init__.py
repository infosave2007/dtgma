"""
DTG-MA: Dynamic Task-Graph Masked Attention

A Python library for continual learning with zero catastrophic forgetting
using attention-based task isolation.

Based on the paper:
"Dynamic Task-Graph Masked Attention: An Architectural Approach to
Continual Learning Without Catastrophic Forgetting"

Quick Start:
    >>> from dtgma import DTGMAModel, train_continual
    >>> model = DTGMAModel(input_dim=784, hidden_dim=256, num_classes=2)
    >>> results = train_continual(model, tasks_data)
"""

from .model import (
    DTGMAModel,
    TaskGraphAttention,
    DTGMALayer,
    DTGMABlock,
    create_task_isolation_mask,
)
from .training import train_task, evaluate, train_continual

__version__ = "1.0.0"
__author__ = "Kirichenko Oleg Yu."

__all__ = [
    # Models
    "DTGMAModel",
    "TaskGraphAttention",
    "DTGMALayer",
    "DTGMABlock",
    "create_task_isolation_mask",
    # Training
    "train_task",
    "evaluate",
    "train_continual",
]

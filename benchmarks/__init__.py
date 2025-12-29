"""
Standard Continual Learning Benchmarks for DTG-MA

Split MNIST - 5 binary classification tasks
Permuted MNIST - 10 tasks with different pixel permutations  
CIFAR-100 Split - 10 tasks, 10 classes each
CORe50 - 50 object recognition, 10 scenarios
miniImageNet - 100 classes for few-shot/continual learning
Omniglot - 1623 characters from 50 alphabets
"""

from .loaders import (
    get_split_mnist,
    get_permuted_mnist,
    get_split_cifar100,
    get_core50,
    get_mini_imagenet,
    get_omniglot,
    BENCHMARK_INFO,
)

__all__ = [
    "get_split_mnist",
    "get_permuted_mnist", 
    "get_split_cifar100",
    "get_core50",
    "get_mini_imagenet",
    "get_omniglot",
    "BENCHMARK_INFO",
]

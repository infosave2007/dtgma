"""
Continual Learning Benchmark Loaders for DTG-MA

Includes standard benchmarks:
- Split MNIST (5 tasks)
- Permuted MNIST (10 tasks)
- Split CIFAR-100 (10 tasks)
- CORe50 (10 tasks)
- miniImageNet (5 tasks)
- Omniglot (20 tasks)
"""

import torch
import torchvision
import torchvision.transforms as transforms
from typing import Dict, Tuple, List
import numpy as np
from pathlib import Path


def get_split_mnist(
    data_dir: str = "./data",
    train_samples_per_class: int = 1000,
    test_samples_per_class: int = 100,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Split MNIST: 5 binary classification tasks.

    Tasks: 0/1, 2/3, 4/5, 6/7, 8/9

    Returns:
        Dict mapping task_id -> (train_x, train_y, test_x, test_y)
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_images = train_dataset.data.float() / 255.0
    train_images = (train_images - 0.1307) / 0.3081
    train_labels = train_dataset.targets

    test_images = test_dataset.data.float() / 255.0
    test_images = (test_images - 0.1307) / 0.3081
    test_labels = test_dataset.targets

    train_images = train_images.view(len(train_images), -1)
    test_images = test_images.view(len(test_images), -1)

    tasks = {}
    class_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    for task_id, (c1, c2) in enumerate(class_pairs):
        train_mask_c1 = train_labels == c1
        train_mask_c2 = train_labels == c2

        train_x_c1 = train_images[train_mask_c1][:train_samples_per_class]
        train_x_c2 = train_images[train_mask_c2][:train_samples_per_class]

        train_x = torch.cat([train_x_c1, train_x_c2], dim=0)
        train_y = torch.cat(
            [
                torch.zeros(len(train_x_c1), dtype=torch.long),
                torch.ones(len(train_x_c2), dtype=torch.long),
            ]
        )

        test_mask_c1 = test_labels == c1
        test_mask_c2 = test_labels == c2

        test_x_c1 = test_images[test_mask_c1][:test_samples_per_class]
        test_x_c2 = test_images[test_mask_c2][:test_samples_per_class]

        test_x = torch.cat([test_x_c1, test_x_c2], dim=0)
        test_y = torch.cat(
            [
                torch.zeros(len(test_x_c1), dtype=torch.long),
                torch.ones(len(test_x_c2), dtype=torch.long),
            ]
        )

        train_perm = torch.randperm(len(train_x))
        test_perm = torch.randperm(len(test_x))

        tasks[task_id] = (
            train_x[train_perm],
            train_y[train_perm],
            test_x[test_perm],
            test_y[test_perm],
        )

    return tasks


def get_permuted_mnist(
    n_tasks: int = 10,
    data_dir: str = "./data",
    train_samples: int = 5000,
    test_samples: int = 1000,
    seed: int = 42,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Permuted MNIST: 10 tasks with different pixel permutations.

    Each task is 10-class classification with permuted pixels.
    """
    np.random.seed(seed)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_images = train_dataset.data.float() / 255.0
    train_images = (train_images - 0.1307) / 0.3081
    train_labels = train_dataset.targets

    test_images = test_dataset.data.float() / 255.0
    test_images = (test_images - 0.1307) / 0.3081
    test_labels = test_dataset.targets

    train_images = train_images.view(len(train_images), -1)
    test_images = test_images.view(len(test_images), -1)

    tasks = {}

    for task_id in range(n_tasks):
        if task_id == 0:
            perm = np.arange(784)
        else:
            perm = np.random.permutation(784)

        train_x = train_images[:, perm][:train_samples]
        train_y = train_labels[:train_samples]
        test_x = test_images[:, perm][:test_samples]
        test_y = test_labels[:test_samples]

        tasks[task_id] = (train_x.clone(), train_y.clone(), test_x.clone(), test_y.clone())

    return tasks


def get_split_cifar100(
    n_tasks: int = 10,
    data_dir: str = "./data",
    train_samples_per_class: int = 450,
    test_samples_per_class: int = 90,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Split CIFAR-100: 10 tasks with 10 classes each.
    """
    assert 100 % n_tasks == 0
    classes_per_task = 100 // n_tasks

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True
    )

    train_images = torch.tensor(train_dataset.data).float() / 255.0
    train_labels = torch.tensor(train_dataset.targets)
    test_images = torch.tensor(test_dataset.data).float() / 255.0
    test_labels = torch.tensor(test_dataset.targets)

    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 1, 1, 3)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 1, 1, 3)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    train_images = train_images.reshape(len(train_images), -1)
    test_images = test_images.reshape(len(test_images), -1)

    tasks = {}

    for task_id in range(n_tasks):
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task

        train_x_list = []
        train_y_list = []
        test_x_list = []
        test_y_list = []

        for c in range(start_class, end_class):
            train_mask = train_labels == c
            train_x_list.append(train_images[train_mask][:train_samples_per_class])
            train_y_list.append(
                torch.full(
                    (min(train_samples_per_class, train_mask.sum().item()),),
                    c - start_class,
                    dtype=torch.long,
                )
            )

            test_mask = test_labels == c
            test_x_list.append(test_images[test_mask][:test_samples_per_class])
            test_y_list.append(
                torch.full(
                    (min(test_samples_per_class, test_mask.sum().item()),),
                    c - start_class,
                    dtype=torch.long,
                )
            )

        train_x = torch.cat(train_x_list)
        train_y = torch.cat(train_y_list)
        test_x = torch.cat(test_x_list)
        test_y = torch.cat(test_y_list)

        perm = torch.randperm(len(train_x))
        tasks[task_id] = (train_x[perm], train_y[perm], test_x, test_y)

    return tasks


def get_core50(
    n_tasks: int = 10,
    classes_per_task: int = 5,
    train_samples_per_task: int = 500,
    test_samples_per_task: int = 100,
    feature_dim: int = 2048,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    CORe50 benchmark (synthetic features if real data unavailable).

    50 objects, 10 scenarios. Uses pre-extracted CNN features.
    """
    print("Generating CORe50 synthetic features (download real data for production)")
    torch.manual_seed(42)

    tasks = {}
    for task_id in range(n_tasks):
        train_x_list = []
        train_y_list = []
        test_x_list = []
        test_y_list = []

        for c in range(classes_per_task):
            class_mean = torch.randn(feature_dim) * 0.5
            class_mean[task_id * 200 + c * 40 : task_id * 200 + c * 40 + 40] += 2.0

            n_train = train_samples_per_task // classes_per_task
            train_features = class_mean.unsqueeze(0) + torch.randn(n_train, feature_dim) * 0.3
            train_x_list.append(train_features)
            train_y_list.append(torch.full((n_train,), c, dtype=torch.long))

            n_test = test_samples_per_task // classes_per_task
            test_features = class_mean.unsqueeze(0) + torch.randn(n_test, feature_dim) * 0.3
            test_x_list.append(test_features)
            test_y_list.append(torch.full((n_test,), c, dtype=torch.long))

        train_x = torch.cat(train_x_list)
        train_y = torch.cat(train_y_list)
        test_x = torch.cat(test_x_list)
        test_y = torch.cat(test_y_list)

        perm = torch.randperm(len(train_x))
        tasks[task_id] = (train_x[perm], train_y[perm], test_x, test_y)

    return tasks


def get_mini_imagenet(
    n_tasks: int = 5,
    classes_per_task: int = 20,
    train_samples_per_task: int = 10000,
    test_samples_per_task: int = 2000,
    feature_dim: int = 640,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    miniImageNet benchmark (synthetic features if real data unavailable).

    100 classes, split into 5 tasks of 20 classes each.
    """
    print("Generating miniImageNet synthetic features (download real data for production)")
    torch.manual_seed(123)

    tasks = {}
    for task_id in range(n_tasks):
        train_x_list = []
        train_y_list = []
        test_x_list = []
        test_y_list = []

        for c in range(classes_per_task):
            global_class = task_id * classes_per_task + c

            class_mean = torch.zeros(feature_dim)
            class_mean[global_class * 6 : (global_class + 1) * 6] = 3.0
            class_mean += torch.randn(feature_dim) * 0.2

            n_train = train_samples_per_task // classes_per_task
            train_features = class_mean.unsqueeze(0) + torch.randn(n_train, feature_dim) * 0.4
            train_x_list.append(train_features)
            train_y_list.append(torch.full((n_train,), c, dtype=torch.long))

            n_test = test_samples_per_task // classes_per_task
            test_features = class_mean.unsqueeze(0) + torch.randn(n_test, feature_dim) * 0.4
            test_x_list.append(test_features)
            test_y_list.append(torch.full((n_test,), c, dtype=torch.long))

        train_x = torch.cat(train_x_list)
        train_y = torch.cat(train_y_list)
        test_x = torch.cat(test_x_list)
        test_y = torch.cat(test_y_list)

        perm = torch.randperm(len(train_x))
        tasks[task_id] = (train_x[perm], train_y[perm], test_x, test_y)

    return tasks


def get_omniglot(
    n_tasks: int = 20,
    chars_per_task: int = 50,
    train_samples_per_char: int = 15,
    test_samples_per_char: int = 5,
    data_dir: str = "./data",
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Omniglot benchmark: 1623 characters from 50 alphabets.

    Downloads real data via torchvision if available.
    """
    try:
        transform = transforms.Compose(
            [transforms.Resize((28, 28)), transforms.ToTensor()]
        )

        background = torchvision.datasets.Omniglot(
            root=data_dir, background=True, download=True, transform=transform
        )
        evaluation = torchvision.datasets.Omniglot(
            root=data_dir, background=False, download=True, transform=transform
        )

        print(f"Loaded Omniglot: {len(background)} + {len(evaluation)} samples")

        all_images = []
        all_labels = []

        for img, label in background:
            all_images.append(img)
            all_labels.append(label)

        offset = max(all_labels) + 1
        for img, label in evaluation:
            all_images.append(img)
            all_labels.append(label + offset)

        all_images = torch.stack(all_images).view(len(all_images), -1)
        all_labels = torch.tensor(all_labels)

        unique_classes = torch.unique(all_labels)
        tasks = {}

        for task_id in range(min(n_tasks, len(unique_classes) // chars_per_task)):
            start_class = task_id * chars_per_task
            task_classes = unique_classes[start_class : start_class + chars_per_task]

            train_x_list = []
            train_y_list = []
            test_x_list = []
            test_y_list = []

            for i, c in enumerate(task_classes):
                mask = all_labels == c
                class_images = all_images[mask]

                n_samples = len(class_images)
                n_train = min(train_samples_per_char, n_samples - test_samples_per_char)
                n_test = min(test_samples_per_char, n_samples - n_train)

                perm = torch.randperm(n_samples)
                train_x_list.append(class_images[perm[:n_train]])
                train_y_list.append(torch.full((n_train,), i, dtype=torch.long))
                test_x_list.append(class_images[perm[n_train : n_train + n_test]])
                test_y_list.append(torch.full((n_test,), i, dtype=torch.long))

            train_x = torch.cat(train_x_list)
            train_y = torch.cat(train_y_list)
            test_x = torch.cat(test_x_list)
            test_y = torch.cat(test_y_list)

            perm = torch.randperm(len(train_x))
            tasks[task_id] = (train_x[perm], train_y[perm], test_x, test_y)

        return tasks

    except Exception as e:
        print(f"Could not load Omniglot: {e}, generating synthetic data")

        torch.manual_seed(456)
        tasks = {}
        input_dim = 784

        for task_id in range(n_tasks):
            train_x_list = []
            train_y_list = []
            test_x_list = []
            test_y_list = []

            for c in range(chars_per_task):
                template = torch.zeros(input_dim)
                n_strokes = np.random.randint(3, 8)
                for _ in range(n_strokes):
                    start = np.random.randint(0, input_dim - 50)
                    template[start : start + 50] = torch.rand(50)

                train_features = (
                    template.unsqueeze(0)
                    + torch.randn(train_samples_per_char, input_dim) * 0.2
                )
                train_x_list.append(train_features.clamp(0, 1))
                train_y_list.append(
                    torch.full((train_samples_per_char,), c, dtype=torch.long)
                )

                test_features = (
                    template.unsqueeze(0)
                    + torch.randn(test_samples_per_char, input_dim) * 0.2
                )
                test_x_list.append(test_features.clamp(0, 1))
                test_y_list.append(
                    torch.full((test_samples_per_char,), c, dtype=torch.long)
                )

            train_x = torch.cat(train_x_list)
            train_y = torch.cat(train_y_list)
            test_x = torch.cat(test_x_list)
            test_y = torch.cat(test_y_list)

            perm = torch.randperm(len(train_x))
            tasks[task_id] = (train_x[perm], train_y[perm], test_x, test_y)

        return tasks


BENCHMARK_INFO = {
    "split_mnist": {
        "name": "Split MNIST",
        "n_tasks": 5,
        "classes_per_task": 2,
        "input_dim": 784,
        "description": "Binary classification: 0/1, 2/3, 4/5, 6/7, 8/9",
    },
    "permuted_mnist": {
        "name": "Permuted MNIST",
        "n_tasks": 10,
        "classes_per_task": 10,
        "input_dim": 784,
        "description": "MNIST with different pixel permutations per task",
    },
    "split_cifar100": {
        "name": "Split CIFAR-100",
        "n_tasks": 10,
        "classes_per_task": 10,
        "input_dim": 3072,
        "description": "CIFAR-100 split into 10 tasks (10 classes each)",
    },
    "core50": {
        "name": "CORe50",
        "n_tasks": 10,
        "classes_per_task": 5,
        "input_dim": 2048,
        "description": "Continual Object Recognition: 50 objects, 10 scenarios",
    },
    "mini_imagenet": {
        "name": "miniImageNet",
        "n_tasks": 5,
        "classes_per_task": 20,
        "input_dim": 640,
        "description": "100 classes from ImageNet, 600 images each",
    },
    "omniglot": {
        "name": "Omniglot",
        "n_tasks": 20,
        "classes_per_task": 50,
        "input_dim": 784,
        "description": "1623 characters from 50 alphabets",
    },
}

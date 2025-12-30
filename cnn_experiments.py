#!/usr/bin/env python3
"""
DTG-MA CNN/ResNet Experiments
=============================

Fair comparison with literature using CNN/ResNet backbones instead of flattened MLP.

Usage:
    python arxiv_cnn_experiments.py --device mps --epochs 30 --backbone resnet

Author: Kirichenko Oleg Yu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))

from dtgma.cnn_backends import (
    SimpleCNN,
    CIFAR_CNN,
    ResNet18,
    ResNet18_MNIST,
    DTGMAWithCNN,
    CNNFineTune,
    CNNEWC,
    CNNHAT,
    CNNPackNet,
    CNNDERPP,
)


# =============================================================================
# Data Loaders (Image Format)
# =============================================================================


def get_split_mnist_images(
    data_dir: str = "./data",
    train_samples_per_class: int = 1000,
    test_samples_per_class: int = 100,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Split MNIST returning images in (N, 1, 28, 28) format."""
    import torchvision
    import torchvision.transforms as transforms

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True
    )

    # Normalize
    train_images = train_dataset.data.float().unsqueeze(1) / 255.0
    train_images = (train_images - 0.1307) / 0.3081
    train_labels = train_dataset.targets

    test_images = test_dataset.data.float().unsqueeze(1) / 255.0
    test_images = (test_images - 0.1307) / 0.3081
    test_labels = test_dataset.targets

    tasks = {}
    class_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    for task_id, (c1, c2) in enumerate(class_pairs):
        train_mask_c1 = train_labels == c1
        train_mask_c2 = train_labels == c2

        train_x_c1 = train_images[train_mask_c1][:train_samples_per_class]
        train_x_c2 = train_images[train_mask_c2][:train_samples_per_class]

        train_x = torch.cat([train_x_c1, train_x_c2], dim=0)
        train_y = torch.cat([
            torch.zeros(len(train_x_c1), dtype=torch.long),
            torch.ones(len(train_x_c2), dtype=torch.long),
        ])

        test_mask_c1 = test_labels == c1
        test_mask_c2 = test_labels == c2

        test_x_c1 = test_images[test_mask_c1][:test_samples_per_class]
        test_x_c2 = test_images[test_mask_c2][:test_samples_per_class]

        test_x = torch.cat([test_x_c1, test_x_c2], dim=0)
        test_y = torch.cat([
            torch.zeros(len(test_x_c1), dtype=torch.long),
            torch.ones(len(test_x_c2), dtype=torch.long),
        ])

        perm = torch.randperm(len(train_x))
        tasks[task_id] = (train_x[perm], train_y[perm], test_x, test_y)

    return tasks


def get_split_cifar100_images(
    n_tasks: int = 10,
    data_dir: str = "./data",
    train_samples_per_class: int = 450,
    test_samples_per_class: int = 90,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Split CIFAR-100 returning images in (N, 3, 32, 32) format."""
    import torchvision

    classes_per_task = 100 // n_tasks

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True
    )

    # (N, H, W, C) -> (N, C, H, W) and normalize
    train_images = torch.tensor(train_dataset.data).float().permute(0, 3, 1, 2) / 255.0
    train_labels = torch.tensor(train_dataset.targets)
    test_images = torch.tensor(test_dataset.data).float().permute(0, 3, 1, 2) / 255.0
    test_labels = torch.tensor(test_dataset.targets)

    # CIFAR-100 normalization
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

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
            n_train = min(train_samples_per_class, train_mask.sum().item())
            train_x_list.append(train_images[train_mask][:n_train])
            train_y_list.append(torch.full((n_train,), c - start_class, dtype=torch.long))

            test_mask = test_labels == c
            n_test = min(test_samples_per_class, test_mask.sum().item())
            test_x_list.append(test_images[test_mask][:n_test])
            test_y_list.append(torch.full((n_test,), c - start_class, dtype=torch.long))

        train_x = torch.cat(train_x_list)
        train_y = torch.cat(train_y_list)
        test_x = torch.cat(test_x_list)
        test_y = torch.cat(test_y_list)

        perm = torch.randperm(len(train_x))
        tasks[task_id] = (train_x[perm], train_y[perm], test_x, test_y)

    return tasks


def get_omniglot_images(
    n_tasks: int = 10,
    chars_per_task: int = 20,
    data_dir: str = "./data",
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Omniglot returning images in (N, 1, 28, 28) format."""
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    background = torchvision.datasets.Omniglot(
        root=data_dir, background=True, download=True, transform=transform
    )
    evaluation = torchvision.datasets.Omniglot(
        root=data_dir, background=False, download=True, transform=transform
    )

    all_images = []
    all_labels = []

    for img, label in background:
        all_images.append(img)
        all_labels.append(label)

    offset = max(all_labels) + 1
    for img, label in evaluation:
        all_images.append(img)
        all_labels.append(label + offset)

    all_images = torch.stack(all_images)  # (N, 1, 28, 28)
    all_labels = torch.tensor(all_labels)

    unique_classes = torch.unique(all_labels)
    tasks = {}

    for task_id in range(min(n_tasks, len(unique_classes) // chars_per_task)):
        start_class = task_id * chars_per_task
        task_classes = unique_classes[start_class:start_class + chars_per_task]

        train_x_list = []
        train_y_list = []
        test_x_list = []
        test_y_list = []

        for i, c in enumerate(task_classes):
            mask = all_labels == c
            class_images = all_images[mask]

            n_samples = len(class_images)
            n_train = max(1, n_samples - 5)
            n_test = min(5, n_samples - 1)

            perm = torch.randperm(n_samples)
            train_x_list.append(class_images[perm[:n_train]])
            train_y_list.append(torch.full((n_train,), i, dtype=torch.long))
            test_x_list.append(class_images[perm[n_train:n_train + n_test]])
            test_y_list.append(torch.full((n_test,), i, dtype=torch.long))

        train_x = torch.cat(train_x_list)
        train_y = torch.cat(train_y_list)
        test_x = torch.cat(test_x_list)
        test_y = torch.cat(test_y_list)

        perm = torch.randperm(len(train_x))
        tasks[task_id] = (train_x[perm], train_y[perm], test_x, test_y)

    return tasks


# =============================================================================
# Training Functions
# =============================================================================


def train_dtgma_cnn(
    model: DTGMAWithCNN,
    tasks_data: Dict,
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 64,
    verbose: bool = True,
    device: str = "cpu",
) -> Dict:
    """Train DTG-MA with CNN backbone."""
    results = {"accuracies": {}, "forgetting": {}, "times": {}}
    task_ids = sorted(tasks_data.keys())

    model = model.to(device)

    for i, task_id in enumerate(task_ids):
        train_x, train_y, _, _ = tasks_data[task_id]
        train_x, train_y = train_x.to(device), train_y.to(device)

        model.register_task(task_id)

        # Get task parameters only (backbone already trained on first task)
        if i == 0:
            params = list(model.parameters())
        else:
            params = model.get_task_parameters(task_id)

        optimizer = torch.optim.Adam(params, lr=lr)

        start = time.time()
        n_samples = len(train_x)

        for epoch in range(epochs):
            model.train()
            indices = torch.randperm(n_samples, device=device)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_x = train_x[indices[start_idx:end_idx]]
                batch_y = train_y[indices[start_idx:end_idx]]

                optimizer.zero_grad()
                loss = F.cross_entropy(model(batch_x, task_id), batch_y)
                loss.backward()
                optimizer.step()

        results["times"][task_id] = time.time() - start

        # Freeze after first task if backbone should be shared
        if i == 0 and not model.freeze_backbone:
            for p in model.backbone.parameters():
                p.requires_grad = False
            model.freeze_backbone = True

        model.freeze_task(task_id)

        # Evaluate
        model.eval()
        for prev_id in task_ids[:i + 1]:
            _, _, tx, ty = tasks_data[prev_id]
            with torch.no_grad():
                preds = model(tx.to(device), prev_id).argmax(1)
                results["accuracies"][(task_id, prev_id)] = (
                    (preds == ty.to(device)).float().mean().item()
                )

        if verbose:
            print(f"  Task {task_id}: {results['accuracies'][(task_id, task_id)]*100:.1f}%")

    # Compute forgetting
    for tid in task_ids[:-1]:
        initial = results["accuracies"][(tid, tid)]
        final = results["accuracies"][(task_ids[-1], tid)]
        results["forgetting"][tid] = max(0, initial - final)

    final_accs = [results["accuracies"][(task_ids[-1], tid)] for tid in task_ids]
    results["avg_accuracy"] = np.mean(final_accs)
    results["avg_forgetting"] = (
        np.mean(list(results["forgetting"].values())) if results["forgetting"] else 0
    )
    results["total_time"] = sum(results["times"].values())

    return results


def train_cnn_baseline(
    model: nn.Module,
    tasks_data: Dict,
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 64,
    verbose: bool = True,
    device: str = "cpu",
) -> Dict:
    """Train CNN baseline (Fine-tune, EWC)."""
    results = {"accuracies": {}, "forgetting": {}, "times": {}}
    task_ids = sorted(tasks_data.keys())

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i, task_id in enumerate(task_ids):
        train_x, train_y, _, _ = tasks_data[task_id]
        train_x, train_y = train_x.to(device), train_y.to(device)

        start = time.time()
        n_samples = len(train_x)

        for epoch in range(epochs):
            model.train()
            indices = torch.randperm(n_samples, device=device)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_x = train_x[indices[start_idx:end_idx]]
                batch_y = train_y[indices[start_idx:end_idx]]

                optimizer.zero_grad()
                loss = F.cross_entropy(model(batch_x), batch_y)

                # EWC regularization
                if hasattr(model, 'ewc_loss'):
                    loss = loss + model.ewc_loss()

                loss.backward()
                optimizer.step()

        results["times"][task_id] = time.time() - start

        # EWC consolidation
        if hasattr(model, 'consolidate'):
            model.consolidate(train_x, train_y)

        # Evaluate
        model.eval()
        for prev_id in task_ids[:i + 1]:
            _, _, tx, ty = tasks_data[prev_id]
            with torch.no_grad():
                preds = model(tx.to(device)).argmax(1)
                results["accuracies"][(task_id, prev_id)] = (
                    (preds == ty.to(device)).float().mean().item()
                )

        if verbose:
            print(f"  Task {task_id}: {results['accuracies'][(task_id, task_id)]*100:.1f}%")

    for tid in task_ids[:-1]:
        initial = results["accuracies"][(tid, tid)]
        final = results["accuracies"][(task_ids[-1], tid)]
        results["forgetting"][tid] = max(0, initial - final)

    final_accs = [results["accuracies"][(task_ids[-1], tid)] for tid in task_ids]
    results["avg_accuracy"] = np.mean(final_accs)
    results["avg_forgetting"] = (
        np.mean(list(results["forgetting"].values())) if results["forgetting"] else 0
    )
    results["total_time"] = sum(results["times"].values())

    return results


def train_cnn_hat(
    model: CNNHAT,
    tasks_data: Dict,
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 64,
    verbose: bool = True,
    device: str = "cpu",
) -> Dict:
    """Train HAT with CNN backbone."""
    results = {"accuracies": {}, "forgetting": {}, "times": {}}
    task_ids = sorted(tasks_data.keys())

    model = model.to(device)

    for i, task_id in enumerate(task_ids):
        train_x, train_y, _, _ = tasks_data[task_id]
        train_x, train_y = train_x.to(device), train_y.to(device)

        model.register_task(task_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        start = time.time()
        n_samples = len(train_x)

        for epoch in range(epochs):
            model.train()
            s = 1.0 + (model.s_max - 1.0) * epoch / epochs
            indices = torch.randperm(n_samples, device=device)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_x = train_x[indices[start_idx:end_idx]]
                batch_y = train_y[indices[start_idx:end_idx]]

                optimizer.zero_grad()
                output = model(batch_x, task_id, s)
                loss = F.cross_entropy(output, batch_y) + model.hat_reg_loss(task_id, s)
                loss.backward()
                optimizer.step()

        results["times"][task_id] = time.time() - start
        model.freeze_task(task_id)

        model.eval()
        for prev_id in task_ids[:i + 1]:
            _, _, tx, ty = tasks_data[prev_id]
            with torch.no_grad():
                preds = model(tx.to(device), prev_id, model.s_max).argmax(1)
                results["accuracies"][(task_id, prev_id)] = (
                    (preds == ty.to(device)).float().mean().item()
                )

        if verbose:
            print(f"  Task {task_id}: {results['accuracies'][(task_id, task_id)]*100:.1f}%")

    for tid in task_ids[:-1]:
        initial = results["accuracies"][(tid, tid)]
        final = results["accuracies"][(task_ids[-1], tid)]
        results["forgetting"][tid] = max(0, initial - final)

    final_accs = [results["accuracies"][(task_ids[-1], tid)] for tid in task_ids]
    results["avg_accuracy"] = np.mean(final_accs)
    results["avg_forgetting"] = (
        np.mean(list(results["forgetting"].values())) if results["forgetting"] else 0
    )
    results["total_time"] = sum(results["times"].values())

    return results


def train_cnn_packnet(
    model: CNNPackNet,
    tasks_data: Dict,
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 64,
    verbose: bool = True,
    device: str = "cpu",
) -> Dict:
    """Train PackNet with CNN backbone."""
    results = {"accuracies": {}, "forgetting": {}, "times": {}}
    task_ids = sorted(tasks_data.keys())

    model = model.to(device)

    for i, task_id in enumerate(task_ids):
        train_x, train_y, _, _ = tasks_data[task_id]
        train_x, train_y = train_x.to(device), train_y.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        start = time.time()
        n_samples = len(train_x)

        for epoch in range(epochs):
            model.train()
            indices = torch.randperm(n_samples, device=device)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_x = train_x[indices[start_idx:end_idx]]
                batch_y = train_y[indices[start_idx:end_idx]]

                optimizer.zero_grad()
                loss = F.cross_entropy(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        results["times"][task_id] = time.time() - start
        model.prune_and_freeze()

        model.eval()
        for prev_id in task_ids[:i + 1]:
            _, _, tx, ty = tasks_data[prev_id]
            with torch.no_grad():
                preds = model(tx.to(device)).argmax(1)
                results["accuracies"][(task_id, prev_id)] = (
                    (preds == ty.to(device)).float().mean().item()
                )

        if verbose:
            print(f"  Task {task_id}: {results['accuracies'][(task_id, task_id)]*100:.1f}%")

    for tid in task_ids[:-1]:
        initial = results["accuracies"][(tid, tid)]
        final = results["accuracies"][(task_ids[-1], tid)]
        results["forgetting"][tid] = max(0, initial - final)

    final_accs = [results["accuracies"][(task_ids[-1], tid)] for tid in task_ids]
    results["avg_accuracy"] = np.mean(final_accs)
    results["avg_forgetting"] = (
        np.mean(list(results["forgetting"].values())) if results["forgetting"] else 0
    )
    results["total_time"] = sum(results["times"].values())

    return results


def train_cnn_derpp(
    model: CNNDERPP,
    tasks_data: Dict,
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 64,
    verbose: bool = True,
    device: str = "cpu",
) -> Dict:
    """Train DER++ with CNN backbone."""
    results = {"accuracies": {}, "forgetting": {}, "times": {}}
    task_ids = sorted(tasks_data.keys())

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i, task_id in enumerate(task_ids):
        train_x, train_y, _, _ = tasks_data[task_id]
        train_x, train_y = train_x.to(device), train_y.to(device)

        start = time.time()
        n_samples = len(train_x)

        for epoch in range(epochs):
            model.train()
            indices = torch.randperm(n_samples, device=device)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_x = train_x[indices[start_idx:end_idx]]
                batch_y = train_y[indices[start_idx:end_idx]]

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = F.cross_entropy(logits, batch_y)

                # Replay
                if model.buffer_count > 0:
                    buf_x, buf_y, buf_logits = model.sample_buffer(batch_size // 2)
                    buf_x = buf_x.to(device)
                    buf_y = buf_y.to(device)
                    buf_logits = buf_logits.to(device)

                    replay_logits = model(buf_x)
                    loss += model.beta * F.cross_entropy(replay_logits, buf_y)
                    loss += model.alpha * F.mse_loss(replay_logits, buf_logits)

                loss.backward()
                optimizer.step()

                # Add to buffer
                model.add_to_buffer(batch_x, batch_y, logits.detach())

        results["times"][task_id] = time.time() - start

        model.eval()
        for prev_id in task_ids[:i + 1]:
            _, _, tx, ty = tasks_data[prev_id]
            with torch.no_grad():
                preds = model(tx.to(device)).argmax(1)
                results["accuracies"][(task_id, prev_id)] = (
                    (preds == ty.to(device)).float().mean().item()
                )

        if verbose:
            print(f"  Task {task_id}: {results['accuracies'][(task_id, task_id)]*100:.1f}%")

    for tid in task_ids[:-1]:
        initial = results["accuracies"][(tid, tid)]
        final = results["accuracies"][(task_ids[-1], tid)]
        results["forgetting"][tid] = max(0, initial - final)

    final_accs = [results["accuracies"][(task_ids[-1], tid)] for tid in task_ids]
    results["avg_accuracy"] = np.mean(final_accs)
    results["avg_forgetting"] = (
        np.mean(list(results["forgetting"].values())) if results["forgetting"] else 0
    )
    results["total_time"] = sum(results["times"].values())

    return results


# =============================================================================
# Main Experiment Runner
# =============================================================================


def run_cnn_benchmark(
    name: str,
    tasks: Dict,
    backbone_type: str,
    num_classes: int,
    n_tasks: int,
    epochs: int,
    device: str,
    hidden_dim: int = 256,
    input_shape: Tuple[int, ...] = (1, 28, 28),
) -> List[Dict]:
    """Run all CNN methods on one benchmark."""
    print(f"\n{'='*60}")
    print(f"  {name} with {backbone_type.upper()} backbone")
    print(f"  Tasks: {n_tasks}, Classes: {num_classes}, Epochs: {epochs}")
    print(f"{'='*60}")

    results = []

    # Create backbone factory
    if backbone_type == "resnet":
        if input_shape[0] == 1:  # Grayscale
            backbone_fn = lambda: ResNet18_MNIST()
        else:
            backbone_fn = lambda: ResNet18(in_channels=3)
    else:  # simple cnn
        if input_shape[0] == 1:
            backbone_fn = lambda: SimpleCNN(in_channels=1)
        else:
            backbone_fn = lambda: CIFAR_CNN()

    methods = [
        (
            f"DTG-MA+{backbone_type.upper()}",
            lambda: DTGMAWithCNN(
                backbone_fn(),
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                freeze_backbone=False,
            ),
            train_dtgma_cnn,
        ),
        (
            f"Fine-tune+{backbone_type.upper()}",
            lambda: CNNFineTune(backbone_fn(), num_classes),
            train_cnn_baseline,
        ),
        (
            f"EWC+{backbone_type.upper()}",
            lambda: CNNEWC(backbone_fn(), num_classes),
            train_cnn_baseline,
        ),
        (
            f"HAT+{backbone_type.upper()}",
            lambda: CNNHAT(backbone_fn(), num_classes),
            train_cnn_hat,
        ),
        (
            f"PackNet+{backbone_type.upper()}",
            lambda: CNNPackNet(backbone_fn(), num_classes),
            train_cnn_packnet,
        ),
        (
            f"DER+++{backbone_type.upper()}",
            lambda: CNNDERPP(backbone_fn(), num_classes, input_shape=input_shape),
            train_cnn_derpp,
        ),
    ]

    for method_name, model_fn, train_fn in methods:
        print(f"\n[{method_name}]")

        try:
            torch.manual_seed(42)
            np.random.seed(42)

            model = model_fn()
            result = train_fn(
                model, tasks, epochs=epochs, lr=0.001, verbose=True, device=device
            )

            n_params = sum(p.numel() for p in model.parameters())

            results.append({
                "method": method_name,
                "benchmark": name,
                "accuracy": result["avg_accuracy"],
                "forgetting": result["avg_forgetting"],
                "time": result["total_time"],
                "params": n_params,
            })

            print(f"  → Acc: {result['avg_accuracy']*100:.1f}%, Forget: {result['avg_forgetting']*100:.1f}%")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "method": method_name,
                "benchmark": name,
                "accuracy": 0,
                "forgetting": 1.0,
                "time": 0,
                "params": 0,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="DTG-MA CNN Experiments")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="resnet", choices=["cnn", "resnet"])
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="mnist,cifar100,omniglot,all",
    )
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Backbone: {args.backbone.upper()}")
    print(f"Epochs: {args.epochs}")

    all_results = []
    benchmarks = args.benchmarks.lower().split(",")
    run_all = "all" in benchmarks

    # Split MNIST
    if run_all or "mnist" in benchmarks:
        tasks = get_split_mnist_images()
        results = run_cnn_benchmark(
            "Split MNIST",
            tasks,
            args.backbone,
            num_classes=2,
            n_tasks=5,
            epochs=args.epochs,
            device=device,
            hidden_dim=args.hidden,
            input_shape=(1, 28, 28),
        )
        all_results.extend(results)

    # Split CIFAR-100
    if run_all or "cifar100" in benchmarks:
        tasks = get_split_cifar100_images()
        results = run_cnn_benchmark(
            "Split CIFAR-100",
            tasks,
            args.backbone,
            num_classes=10,
            n_tasks=10,
            epochs=args.epochs,
            device=device,
            hidden_dim=args.hidden,
            input_shape=(3, 32, 32),
        )
        all_results.extend(results)

    # Omniglot
    if run_all or "omniglot" in benchmarks:
        tasks = get_omniglot_images()
        results = run_cnn_benchmark(
            "Omniglot",
            tasks,
            args.backbone,
            num_classes=20,
            n_tasks=10,
            epochs=args.epochs,
            device=device,
            hidden_dim=args.hidden,
            input_shape=(1, 28, 28),
        )
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY ({args.backbone.upper()} backbone)")
    print("=" * 80)

    benchmarks_list = sorted(set(r["benchmark"] for r in all_results))

    for bench in benchmarks_list:
        print(f"\n{bench}:")
        print(f"{'Method':<25} {'Accuracy':>12} {'Forgetting':>12} {'Time':>10} {'Params':>12}")
        print("-" * 75)

        bench_results = [r for r in all_results if r["benchmark"] == bench]
        for r in sorted(bench_results, key=lambda x: -x["accuracy"]):
            print(
                f"{r['method']:<25} {r['accuracy']*100:>11.1f}% {r['forgetting']*100:>11.1f}% "
                f"{r['time']:>9.1f}s {r['params']:>12,}"
            )

    # Save results
    output = Path(f"CNN_RESULTS_{args.backbone.upper()}.md")
    with open(output, "w") as f:
        f.write(f"# DTG-MA CNN Experiment Results ({args.backbone.upper()} backbone)\n\n")
        f.write(f"**Device**: {device}\n")
        f.write(f"**Backbone**: {args.backbone.upper()}\n")
        f.write(f"**Epochs**: {args.epochs}\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n\n")

        for bench in benchmarks_list:
            f.write(f"\n## {bench}\n\n")
            f.write("| Method | Accuracy | Forgetting | Time | Params |\n")
            f.write("|--------|----------|------------|------|--------|\n")

            bench_results = [r for r in all_results if r["benchmark"] == bench]
            for r in sorted(bench_results, key=lambda x: -x["accuracy"]):
                f.write(
                    f"| {r['method']} | {r['accuracy']*100:.1f}% | "
                    f"{r['forgetting']*100:.1f}% | {r['time']:.1f}s | {r['params']:,} |\n"
                )

    print(f"\n✓ Results saved to {output}")


if __name__ == "__main__":
    main()

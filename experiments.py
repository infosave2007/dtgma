#!/usr/bin/env python3
"""
DTG-MA arXiv Experiments
========================

Full comparison on standard continual learning benchmarks.

Usage:
    python arxiv_experiments.py --device mps --epochs 50

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
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from benchmarks import (
    get_split_mnist,
    get_permuted_mnist,
    get_split_cifar100,
    get_core50,
    get_mini_imagenet,
    get_omniglot,
)
from dtgma import DTGMAModel, train_dtgma_continual
from dtgma.baselines import (
    EWCModel,
    FineTuneModel,
    HATModel,
    PackNetModel,
    DERPPModel,
    train_continual_baseline,
    train_hat_continual,
    train_packnet_continual,
    train_derpp_continual,
)


# =============================================================================
# LoRA Baseline
# =============================================================================


class LoRAModel(nn.Module):
    """LoRA adapter baseline for continual learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_tasks: int = 10,
        rank: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()

        for t in range(num_tasks):
            self.lora_A[str(t)] = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
            self.lora_B[str(t)] = nn.Parameter(torch.zeros(rank, hidden_dim))

        self.current_task = 0
        self.name = f"LoRA-r{rank}"

    def forward(self, x: torch.Tensor, task_id: int = None) -> torch.Tensor:
        if task_id is None:
            task_id = self.current_task

        lora = x @ self.lora_A[str(task_id)] @ self.lora_B[str(task_id)]
        h = F.relu(self.fc1(x) + lora)
        h = F.relu(self.fc2(h))
        return self.classifier(h)


def train_lora_continual(
    model: LoRAModel,
    tasks_data: Dict,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 64,
    verbose: bool = True,
    device: str = "cpu",
) -> Dict:
    """Train LoRA model on sequential tasks."""
    results = {"accuracies": {}, "forgetting": {}, "times": {}}
    task_ids = sorted(tasks_data.keys())

    model = model.to(device)

    for i, task_id in enumerate(task_ids):
        train_x, train_y, _, _ = tasks_data[task_id]
        train_x, train_y = train_x.to(device), train_y.to(device)

        model.current_task = task_id

        if i == 0:
            params = model.parameters()
        else:
            params = (
                [model.lora_A[str(task_id)], model.lora_B[str(task_id)]]
                + list(model.classifier.parameters())
            )

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

        if i == 0:
            for p in model.fc1.parameters():
                p.requires_grad = False
            for p in model.fc2.parameters():
                p.requires_grad = False

        model.eval()
        for prev_id in task_ids[: i + 1]:
            _, _, tx, ty = tasks_data[prev_id]
            with torch.no_grad():
                preds = model(tx.to(device), prev_id).argmax(1)
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
# Adapter Baseline
# =============================================================================


class AdapterModel(nn.Module):
    """Bottleneck adapter baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_tasks: int = 10,
        bottleneck: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.adapters = nn.ModuleDict()
        for t in range(num_tasks):
            self.adapters[str(t)] = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, hidden_dim),
            )

        self.current_task = 0
        self.name = f"Adapter-b{bottleneck}"

    def forward(self, x: torch.Tensor, task_id: int = None) -> torch.Tensor:
        if task_id is None:
            task_id = self.current_task

        h = F.relu(self.fc1(x))
        h = h + self.adapters[str(task_id)](h)
        h = F.relu(self.fc2(h))
        return self.classifier(h)


def train_adapter_continual(
    model: AdapterModel,
    tasks_data: Dict,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 64,
    verbose: bool = True,
    device: str = "cpu",
) -> Dict:
    """Train Adapter model on sequential tasks."""
    results = {"accuracies": {}, "forgetting": {}, "times": {}}
    task_ids = sorted(tasks_data.keys())

    model = model.to(device)

    for i, task_id in enumerate(task_ids):
        train_x, train_y, _, _ = tasks_data[task_id]
        train_x, train_y = train_x.to(device), train_y.to(device)

        model.current_task = task_id

        if i == 0:
            params = model.parameters()
        else:
            params = list(model.adapters[str(task_id)].parameters()) + list(
                model.classifier.parameters()
            )

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

        if i == 0:
            for p in model.fc1.parameters():
                p.requires_grad = False
            for p in model.fc2.parameters():
                p.requires_grad = False

        model.eval()
        for prev_id in task_ids[: i + 1]:
            _, _, tx, ty = tasks_data[prev_id]
            with torch.no_grad():
                preds = model(tx.to(device), prev_id).argmax(1)
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


def run_benchmark(
    name: str,
    tasks: Dict,
    input_dim: int,
    num_classes: int,
    n_tasks: int,
    epochs: int,
    device: str,
    hidden_dim: int = 256,
    n_runs: int = 1,
) -> List[Dict]:
    """Run all methods on one benchmark."""
    print(f"\n{'='*60}")
    print(f"  {name}: {n_tasks} tasks, {num_classes} classes, dim={input_dim}")
    print(f"  Epochs: {epochs}, Runs: {n_runs}, Device: {device}")
    print(f"{'='*60}")

    results = []

    methods = [
        ("DTG-MA", lambda: DTGMAModel(input_dim, hidden_dim, num_classes), train_dtgma_continual),
        (
            "Fine-tune",
            lambda: FineTuneModel(input_dim, hidden_dim, num_classes),
            lambda m, d, **kw: train_continual_baseline(m, d, **kw),
        ),
        (
            "EWC",
            lambda: EWCModel(input_dim, hidden_dim, num_classes, ewc_lambda=1000),
            lambda m, d, **kw: train_continual_baseline(m, d, **kw),
        ),
        ("HAT", lambda: HATModel(input_dim, hidden_dim, num_classes), train_hat_continual),
        (
            "PackNet",
            lambda: PackNetModel(input_dim, hidden_dim, num_classes),
            train_packnet_continual,
        ),
        (
            "DER++",
            lambda: DERPPModel(input_dim, hidden_dim, num_classes, buffer_size=500),
            train_derpp_continual,
        ),
        (
            "LoRA",
            lambda: LoRAModel(input_dim, hidden_dim, num_classes, num_tasks=n_tasks),
            train_lora_continual,
        ),
        (
            "Adapter",
            lambda: AdapterModel(input_dim, hidden_dim, num_classes, num_tasks=n_tasks),
            train_adapter_continual,
        ),
    ]

    for method_name, model_cls, train_fn in methods:
        print(f"\n[{method_name}]")

        accs = []
        forgets = []
        times = []

        for run in range(n_runs):
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)

            try:
                model = model_cls()
                result = train_fn(
                    model, tasks, epochs=epochs, lr=0.001, verbose=(run == 0), device=device
                )

                accs.append(result["avg_accuracy"])
                forgets.append(result["avg_forgetting"])
                times.append(result.get("total_time", 0))

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                accs.append(0)
                forgets.append(1.0)
                times.append(0)

        model = model_cls()
        n_params = sum(p.numel() for p in model.parameters())

        results.append(
            {
                "method": method_name,
                "benchmark": name,
                "accuracy": np.mean(accs),
                "accuracy_std": np.std(accs) if n_runs > 1 else 0,
                "forgetting": np.mean(forgets),
                "forgetting_std": np.std(forgets) if n_runs > 1 else 0,
                "time": np.mean(times),
                "params": n_params,
            }
        )

        if n_runs > 1:
            print(
                f"  → Acc: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%, "
                f"Forget: {np.mean(forgets)*100:.1f}% ± {np.std(forgets)*100:.1f}%"
            )
        else:
            print(f"  → Acc: {np.mean(accs)*100:.1f}%, Forget: {np.mean(forgets)*100:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="DTG-MA arXiv Experiments")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, mps, auto")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per task")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--runs", type=int, default=3, help="Runs per method")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Benchmarks: mnist, permuted, cifar, core50, mini, omniglot, all",
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
    print(f"Epochs: {args.epochs}")
    print(f"Hidden: {args.hidden}")
    print(f"Runs: {args.runs}")

    all_results = []

    benchmarks_to_run = args.benchmarks.lower().split(",")
    run_all = "all" in benchmarks_to_run

    # 1. Split MNIST
    if run_all or "mnist" in benchmarks_to_run:
        tasks = get_split_mnist(data_dir="./data")
        results = run_benchmark(
            "Split MNIST", tasks, 784, 2, 5, args.epochs, device, args.hidden, args.runs
        )
        all_results.extend(results)

    # 2. Permuted MNIST
    if run_all or "permuted" in benchmarks_to_run:
        tasks = get_permuted_mnist(n_tasks=10, data_dir="./data")
        results = run_benchmark(
            "Permuted MNIST", tasks, 784, 10, 10, args.epochs, device, args.hidden, args.runs
        )
        all_results.extend(results)

    # 3. Split CIFAR-100
    if run_all or "cifar" in benchmarks_to_run:
        tasks = get_split_cifar100(n_tasks=10, data_dir="./data")
        results = run_benchmark(
            "Split CIFAR-100", tasks, 3072, 10, 10, args.epochs, device, args.hidden, args.runs
        )
        all_results.extend(results)

    # 4. CORe50
    if run_all or "core50" in benchmarks_to_run:
        tasks = get_core50(n_tasks=10)
        results = run_benchmark(
            "CORe50", tasks, 2048, 5, 10, args.epochs, device, args.hidden, args.runs
        )
        all_results.extend(results)

    # 5. miniImageNet
    if run_all or "mini" in benchmarks_to_run:
        tasks = get_mini_imagenet(n_tasks=5)
        results = run_benchmark(
            "miniImageNet", tasks, 640, 20, 5, args.epochs, device, args.hidden, args.runs
        )
        all_results.extend(results)

    # 6. Omniglot
    if run_all or "omniglot" in benchmarks_to_run:
        tasks = get_omniglot(n_tasks=10, chars_per_task=20)
        results = run_benchmark(
            "Omniglot", tasks, 784, 20, 10, args.epochs, device, args.hidden, args.runs
        )
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    benchmarks = sorted(set(r["benchmark"] for r in all_results))

    for bench in benchmarks:
        print(f"\n{bench}:")
        print(f"{'Method':<12} {'Accuracy':>12} {'Forgetting':>14} {'Time':>10} {'Params':>12}")
        print("-" * 65)

        bench_results = [r for r in all_results if r["benchmark"] == bench]
        for r in sorted(bench_results, key=lambda x: -x["accuracy"]):
            if r["accuracy_std"] > 0:
                acc_str = f"{r['accuracy']*100:.1f}±{r['accuracy_std']*100:.1f}%"
                fgt_str = f"{r['forgetting']*100:.1f}±{r['forgetting_std']*100:.1f}%"
            else:
                acc_str = f"{r['accuracy']*100:.1f}%"
                fgt_str = f"{r['forgetting']*100:.1f}%"

            print(f"{r['method']:<12} {acc_str:>12} {fgt_str:>14} {r['time']:>9.1f}s {r['params']:>12,}")

    # Save results
    output = Path("ARXIV_RESULTS.md")
    with open(output, "w") as f:
        f.write("# DTG-MA arXiv Experiment Results\n\n")
        f.write(f"**Device**: {device}\n")
        f.write(f"**Epochs**: {args.epochs}\n")
        f.write(f"**Runs**: {args.runs}\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n\n")

        for bench in benchmarks:
            f.write(f"\n## {bench}\n\n")
            f.write("| Method | Accuracy | Forgetting | Time | Params |\n")
            f.write("|--------|----------|------------|------|--------|\n")

            bench_results = [r for r in all_results if r["benchmark"] == bench]
            for r in sorted(bench_results, key=lambda x: -x["accuracy"]):
                if r["accuracy_std"] > 0:
                    acc_str = f"{r['accuracy']*100:.1f}% ± {r['accuracy_std']*100:.1f}%"
                    fgt_str = f"{r['forgetting']*100:.1f}% ± {r['forgetting_std']*100:.1f}%"
                else:
                    acc_str = f"{r['accuracy']*100:.1f}%"
                    fgt_str = f"{r['forgetting']*100:.1f}%"

                f.write(f"| {r['method']} | {acc_str} | {fgt_str} | {r['time']:.1f}s | {r['params']:,} |\n")

    print(f"\n✓ Results saved to {output}")


if __name__ == "__main__":
    main()

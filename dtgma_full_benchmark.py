#!/usr/bin/env python3
"""
DTG-MA Full Benchmark Suite

All benchmarks from FCD paper, adapted for DTG-MA:
1. Split MNIST (5 binary tasks)
2. Permuted MNIST (10 tasks, 10 classes each)
3. Split CIFAR-100 (10 tasks, 10 classes each)
4. LLM benchmark on Qwen2.5-1.5B (continual fine-tuning)
5. Ablation study (with/without freezing)
6. Scalability test (T > k)

Usage:
    python dtgma_full_benchmark.py --benchmark split_mnist --device cpu
    python dtgma_full_benchmark.py --benchmark permuted_mnist --device cpu
    python dtgma_full_benchmark.py --benchmark cifar100 --device cpu
    python dtgma_full_benchmark.py --benchmark llm_qwen25 --device cpu
    python dtgma_full_benchmark.py --benchmark ablation --device cpu
    python dtgma_full_benchmark.py --benchmark scalability --device cpu
    python dtgma_full_benchmark.py --benchmark all --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from dtgma import DTGMAModel, train_continual, train_task, evaluate


# ============================================================================
# BENCHMARK LOADERS
# ============================================================================


def get_split_mnist(
    train_samples_per_class: int = 1000,
    test_samples_per_class: int = 200,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Split MNIST: 5 binary tasks (0vs1, 2vs3, 4vs5, 6vs7, 8vs9)."""
    try:
        from benchmarks import get_split_mnist as _get_split_mnist
        return _get_split_mnist(
            train_samples_per_class=train_samples_per_class,
            test_samples_per_class=test_samples_per_class,
        )
    except ImportError:
        # Fallback: load manually
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

        train_x = train_ds.data.float().view(-1, 784) / 255.0
        train_y = train_ds.targets
        test_x = test_ds.data.float().view(-1, 784) / 255.0
        test_y = test_ds.targets

        tasks = {}
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        for tid, (c1, c2) in enumerate(pairs):
            tr_idx = ((train_y == c1) | (train_y == c2))
            te_idx = ((test_y == c1) | (test_y == c2))

            tr_x = train_x[tr_idx][:train_samples_per_class * 2]
            tr_y = (train_y[tr_idx][:train_samples_per_class * 2] == c2).long()
            te_x = test_x[te_idx][:test_samples_per_class * 2]
            te_y = (test_y[te_idx][:test_samples_per_class * 2] == c2).long()

            tasks[tid] = (tr_x, tr_y, te_x, te_y)
        return tasks


def get_permuted_mnist(
    n_tasks: int = 10,
    train_samples: int = 5000,
    test_samples: int = 1000,
    seed: int = 42,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Permuted MNIST: n_tasks with different pixel permutations, 10 classes each."""
    try:
        from benchmarks import get_permuted_mnist as _get_permuted_mnist
        return _get_permuted_mnist(
            n_tasks=n_tasks,
            train_samples=train_samples,
            test_samples=test_samples,
            seed=seed,
        )
    except ImportError:
        import torchvision

        train_ds = torchvision.datasets.MNIST("./data", train=True, download=True)
        test_ds = torchvision.datasets.MNIST("./data", train=False, download=True)

        train_x = train_ds.data.float().view(-1, 784) / 255.0
        train_y = train_ds.targets
        test_x = test_ds.data.float().view(-1, 784) / 255.0
        test_y = test_ds.targets

        rng = np.random.default_rng(seed)
        tasks = {}

        for tid in range(n_tasks):
            if tid == 0:
                perm = np.arange(784)
            else:
                perm = rng.permutation(784)

            tr_x = train_x[:train_samples, perm]
            tr_y = train_y[:train_samples]
            te_x = test_x[:test_samples, perm]
            te_y = test_y[:test_samples]

            tasks[tid] = (tr_x, tr_y.clone(), te_x, te_y.clone())
        return tasks


def get_split_cifar100(
    n_tasks: int = 10,
    train_samples_per_class: int = 400,
    test_samples_per_class: int = 100,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Split CIFAR-100: n_tasks with 10 classes each."""
    try:
        from benchmarks import get_split_cifar100 as _get_split_cifar100
        return _get_split_cifar100(
            n_tasks=n_tasks,
            train_samples_per_class=train_samples_per_class,
            test_samples_per_class=test_samples_per_class,
        )
    except ImportError:
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform)

        train_x = torch.tensor(train_ds.data).float().permute(0, 3, 1, 2) / 255.0
        train_x = train_x.view(len(train_x), -1)  # Flatten to 3072
        train_y = torch.tensor(train_ds.targets)

        test_x = torch.tensor(test_ds.data).float().permute(0, 3, 1, 2) / 255.0
        test_x = test_x.view(len(test_x), -1)
        test_y = torch.tensor(test_ds.targets)

        classes_per_task = 100 // n_tasks
        tasks = {}

        for tid in range(n_tasks):
            start_class = tid * classes_per_task
            end_class = start_class + classes_per_task

            tr_mask = (train_y >= start_class) & (train_y < end_class)
            te_mask = (test_y >= start_class) & (test_y < end_class)

            tr_x = train_x[tr_mask]
            tr_y = train_y[tr_mask] - start_class  # Remap to 0..classes_per_task-1
            te_x = test_x[te_mask]
            te_y = test_y[te_mask] - start_class

            # Limit samples
            n_tr = min(len(tr_x), train_samples_per_class * classes_per_task)
            n_te = min(len(te_x), test_samples_per_class * classes_per_task)

            tasks[tid] = (tr_x[:n_tr], tr_y[:n_tr], te_x[:n_te], te_y[:n_te])

        return tasks


# ============================================================================
# BENCHMARK RUNNERS
# ============================================================================


def run_split_mnist(
    epochs: int = 100,
    hidden_dim: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    device: str = "cpu",
    runs: int = 3,
) -> Dict[str, Any]:
    """Run Split MNIST benchmark (5 binary tasks)."""
    print("=" * 70)
    print("DTG-MA Benchmark: Split MNIST (5 binary tasks)")
    print("=" * 70)

    all_results = []

    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")
        tasks = get_split_mnist(train_samples_per_class=1000, test_samples_per_class=200)

        model = DTGMAModel(
            input_dim=784,
            hidden_dim=hidden_dim,
            num_classes=2,
            n_layers=n_layers,
            n_heads=n_heads,
            n_tasks_max=5,
            dropout=0.1,
        )

        results = train_continual(
            model=model,
            tasks=tasks,
            epochs=epochs,
            batch_size=64,
            lr=0.01,
            device=device,
            verbose=True,
            freeze_after_training=True,
        )
        all_results.append(results)

    # Aggregate
    avg_acc = np.mean([r["avg_accuracy"] for r in all_results])
    std_acc = np.std([r["avg_accuracy"] for r in all_results])
    avg_forg = np.mean([r["avg_forgetting"] for r in all_results])
    std_forg = np.std([r["avg_forgetting"] for r in all_results])

    print(f"\n{'=' * 70}")
    print(f"SPLIT MNIST RESULTS ({runs} runs)")
    print(f"{'=' * 70}")
    print(f"Average Accuracy:   {avg_acc*100:.1f} ± {std_acc*100:.1f}%")
    print(f"Average Forgetting: {avg_forg*100:.1f} ± {std_forg*100:.1f}%")

    return {
        "benchmark": "split_mnist",
        "avg_accuracy": avg_acc,
        "std_accuracy": std_acc,
        "avg_forgetting": avg_forg,
        "std_forgetting": std_forg,
        "runs": runs,
    }


def run_permuted_mnist(
    n_tasks: int = 10,
    epochs: int = 50,
    hidden_dim: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    device: str = "cpu",
    runs: int = 3,
) -> Dict[str, Any]:
    """Run Permuted MNIST benchmark (10 tasks, 10 classes each)."""
    print("=" * 70)
    print(f"DTG-MA Benchmark: Permuted MNIST ({n_tasks} tasks, 10 classes each)")
    print("=" * 70)

    all_results = []

    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")
        tasks = get_permuted_mnist(n_tasks=n_tasks, train_samples=5000, test_samples=1000, seed=42 + run)

        model = DTGMAModel(
            input_dim=784,
            hidden_dim=hidden_dim,
            num_classes=10,
            n_layers=n_layers,
            n_heads=n_heads,
            n_tasks_max=n_tasks,
            dropout=0.1,
        )

        results = train_continual(
            model=model,
            tasks=tasks,
            epochs=epochs,
            batch_size=64,
            lr=0.01,
            device=device,
            verbose=True,
            freeze_after_training=True,
        )
        all_results.append(results)

    avg_acc = np.mean([r["avg_accuracy"] for r in all_results])
    std_acc = np.std([r["avg_accuracy"] for r in all_results])
    avg_forg = np.mean([r["avg_forgetting"] for r in all_results])
    std_forg = np.std([r["avg_forgetting"] for r in all_results])

    print(f"\n{'=' * 70}")
    print(f"PERMUTED MNIST RESULTS ({runs} runs)")
    print(f"{'=' * 70}")
    print(f"Average Accuracy:   {avg_acc*100:.1f} ± {std_acc*100:.1f}%")
    print(f"Average Forgetting: {avg_forg*100:.1f} ± {std_forg*100:.1f}%")

    return {
        "benchmark": "permuted_mnist",
        "n_tasks": n_tasks,
        "avg_accuracy": avg_acc,
        "std_accuracy": std_acc,
        "avg_forgetting": avg_forg,
        "std_forgetting": std_forg,
        "runs": runs,
    }


def run_split_cifar100(
    n_tasks: int = 10,
    epochs: int = 50,
    hidden_dim: int = 512,
    n_layers: int = 2,
    n_heads: int = 8,
    device: str = "cpu",
    runs: int = 3,
) -> Dict[str, Any]:
    """Run Split CIFAR-100 benchmark (10 tasks, 10 classes each)."""
    print("=" * 70)
    print(f"DTG-MA Benchmark: Split CIFAR-100 ({n_tasks} tasks, 10 classes each)")
    print("=" * 70)

    all_results = []

    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")
        tasks = get_split_cifar100(n_tasks=n_tasks)

        model = DTGMAModel(
            input_dim=3072,  # 32x32x3 flattened
            hidden_dim=hidden_dim,
            num_classes=10,
            n_layers=n_layers,
            n_heads=n_heads,
            n_tasks_max=n_tasks,
            dropout=0.1,
        )

        results = train_continual(
            model=model,
            tasks=tasks,
            epochs=epochs,
            batch_size=64,
            lr=0.01,
            device=device,
            verbose=True,
            freeze_after_training=True,
        )
        all_results.append(results)

    avg_acc = np.mean([r["avg_accuracy"] for r in all_results])
    std_acc = np.std([r["avg_accuracy"] for r in all_results])
    avg_forg = np.mean([r["avg_forgetting"] for r in all_results])
    std_forg = np.std([r["avg_forgetting"] for r in all_results])

    print(f"\n{'=' * 70}")
    print(f"SPLIT CIFAR-100 RESULTS ({runs} runs)")
    print(f"{'=' * 70}")
    print(f"Average Accuracy:   {avg_acc*100:.1f} ± {std_acc*100:.1f}%")
    print(f"Average Forgetting: {avg_forg*100:.1f} ± {std_forg*100:.1f}%")

    return {
        "benchmark": "split_cifar100",
        "n_tasks": n_tasks,
        "avg_accuracy": avg_acc,
        "std_accuracy": std_acc,
        "avg_forgetting": avg_forg,
        "std_forgetting": std_forg,
        "runs": runs,
    }


def run_llm_qwen25(
    n_tasks: int = 3,
    epochs: int = 30,
    train_samples: int = 100,
    test_samples: int = 50,
    hidden_dim: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    device: str = "cpu",
    dtype: str = "float32",
) -> Dict[str, Any]:
    """
    Run LLM continual learning benchmark on Qwen2.5-1.5B.
    
    Similar to FCD's GPT-2 benchmark but using Qwen2.5.
    Tests: Sentiment, Topic, Formality (or more tasks).
    """
    print("=" * 70)
    print("DTG-MA Benchmark: LLM Continual Learning (Qwen2.5-1.5B)")
    print("=" * 70)
    print(f"Tasks: {n_tasks}")
    print(f"Epochs per task: {epochs}")
    print(f"Device: {device}")
    print("=" * 70)

    # Import from existing benchmark
    try:
        from dtgma_qwen25_benchmark import (
            load_qwen_model,
            encode_texts,
            build_text_domain_dataset,
            TEXT_DOMAINS,
        )
    except ImportError:
        print("[ERROR] Cannot import dtgma_qwen25_benchmark. Run from dtgma directory.")
        return {}

    # Load Qwen model
    tokenizer, qwen_model = load_qwen_model(device=device, dtype=dtype)

    # Build tasks
    task_ids = list(range(min(n_tasks, len(TEXT_DOMAINS))))
    tasks = {}

    print("\nBuilding datasets and extracting embeddings...")
    for tid in task_ids:
        domain_name = TEXT_DOMAINS[tid]["name"]
        print(f"  Task {tid}: {domain_name}")

        train_texts, train_labels = build_text_domain_dataset(tid, train_samples, 42)
        test_texts, test_labels = build_text_domain_dataset(tid, test_samples, 999)

        train_emb = encode_texts(qwen_model, tokenizer, train_texts, device, batch_size=8)
        test_emb = encode_texts(qwen_model, tokenizer, test_texts, device, batch_size=8)

        train_y = torch.tensor(train_labels, dtype=torch.long)
        test_y = torch.tensor(test_labels, dtype=torch.long)

        tasks[tid] = (train_emb, train_y, test_emb, test_y)

    input_dim = tasks[0][0].size(1)
    print(f"\nEmbedding dimension: {input_dim}")

    # Create DTG-MA model
    model = DTGMAModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        n_layers=n_layers,
        n_heads=n_heads,
        n_tasks_max=max(task_ids) + 1,
        dropout=0.1,
    )

    print(f"Total params: {model.get_total_params():,}")

    results = train_continual(
        model=model,
        tasks=tasks,
        epochs=epochs,
        batch_size=32,
        lr=0.001,
        device=device,
        verbose=True,
        freeze_after_training=True,
    )

    results["benchmark"] = "llm_qwen25"
    results["encoder"] = "Qwen2.5-1.5B"
    results["task_names"] = {tid: TEXT_DOMAINS[tid]["name"] for tid in task_ids}

    return results


def run_ablation(
    epochs: int = 100,
    hidden_dim: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    device: str = "cpu",
    runs: int = 3,
) -> Dict[str, Any]:
    """
    Ablation study: compare with/without freezing.
    
    Configurations:
    1. Full DTG-MA (attention + FFN per task, freezing)
    2. Without freezing (all tasks share gradients)
    """
    print("=" * 70)
    print("DTG-MA Ablation Study")
    print("=" * 70)

    tasks = get_split_mnist(train_samples_per_class=1000, test_samples_per_class=200)
    configs = [
        ("Full DTG-MA (with freezing)", True),
        ("No freezing (shared gradients)", False),
    ]

    results_all = {}

    for config_name, freeze in configs:
        print(f"\n{'=' * 50}")
        print(f"Configuration: {config_name}")
        print(f"{'=' * 50}")

        run_results = []
        for run in range(runs):
            model = DTGMAModel(
                input_dim=784,
                hidden_dim=hidden_dim,
                num_classes=2,
                n_layers=n_layers,
                n_heads=n_heads,
                n_tasks_max=5,
                dropout=0.1,
            )

            results = train_continual(
                model=model,
                tasks=tasks,
                epochs=epochs,
                batch_size=64,
                lr=0.01,
                device=device,
                verbose=(run == 0),
                freeze_after_training=freeze,
            )
            run_results.append(results)

        avg_acc = np.mean([r["avg_accuracy"] for r in run_results])
        std_acc = np.std([r["avg_accuracy"] for r in run_results])
        avg_forg = np.mean([r["avg_forgetting"] for r in run_results])
        std_forg = np.std([r["avg_forgetting"] for r in run_results])

        results_all[config_name] = {
            "avg_accuracy": avg_acc,
            "std_accuracy": std_acc,
            "avg_forgetting": avg_forg,
            "std_forgetting": std_forg,
        }

        print(f"  Accuracy:   {avg_acc*100:.1f} ± {std_acc*100:.1f}%")
        print(f"  Forgetting: {avg_forg*100:.1f} ± {std_forg*100:.1f}%")

    print(f"\n{'=' * 70}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<35} {'Accuracy':>15} {'Forgetting':>15}")
    print("-" * 70)
    for name, r in results_all.items():
        acc = f"{r['avg_accuracy']*100:.1f} ± {r['std_accuracy']*100:.1f}%"
        forg = f"{r['avg_forgetting']*100:.1f} ± {r['std_forgetting']*100:.1f}%"
        print(f"{name:<35} {acc:>15} {forg:>15}")

    return {"benchmark": "ablation", "results": results_all, "runs": runs}


def run_scalability(
    max_tasks: int = 20,
    epochs: int = 50,
    hidden_dim: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Scalability test: T > k scenario.
    
    Tests how DTG-MA degrades when number of tasks exceeds model capacity.
    """
    print("=" * 70)
    print("DTG-MA Scalability Test (T > k)")
    print("=" * 70)

    task_counts = [5, 10, 16, 20]
    if max_tasks not in task_counts:
        task_counts.append(max_tasks)
    task_counts = sorted([t for t in task_counts if t <= max_tasks])

    results_all = {}

    for n_tasks in task_counts:
        print(f"\n{'=' * 50}")
        print(f"Testing with {n_tasks} tasks")
        print(f"{'=' * 50}")

        tasks = get_permuted_mnist(n_tasks=n_tasks, train_samples=2000, test_samples=500)

        model = DTGMAModel(
            input_dim=784,
            hidden_dim=hidden_dim,
            num_classes=10,
            n_layers=n_layers,
            n_heads=n_heads,
            n_tasks_max=n_tasks,
            dropout=0.1,
        )

        results = train_continual(
            model=model,
            tasks=tasks,
            epochs=epochs,
            batch_size=64,
            lr=0.01,
            device=device,
            verbose=True,
            freeze_after_training=True,
        )

        results_all[n_tasks] = {
            "avg_accuracy": results["avg_accuracy"],
            "avg_forgetting": results["avg_forgetting"],
        }

    print(f"\n{'=' * 70}")
    print("SCALABILITY SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Tasks':<10} {'Accuracy':>15} {'Forgetting':>15}")
    print("-" * 45)
    for n_tasks, r in results_all.items():
        print(f"{n_tasks:<10} {r['avg_accuracy']*100:>14.1f}% {r['avg_forgetting']*100:>14.1f}%")

    return {"benchmark": "scalability", "results": results_all}


# ============================================================================
# REPORT GENERATION
# ============================================================================


def write_full_report(all_results: Dict[str, Any], path: Path):
    """Write comprehensive benchmark report."""
    md = []
    md.append("# DTG-MA Full Benchmark Results")
    md.append("")
    md.append("Comprehensive evaluation of Dynamic Task-Graph Masked Attention")
    md.append("on standard continual learning benchmarks.")
    md.append("")
    md.append("## Summary Table")
    md.append("")
    md.append("| Benchmark | Tasks | Accuracy | Forgetting |")
    md.append("|---|---:|---:|---:|")

    for name, r in all_results.items():
        if "avg_accuracy" in r:
            n_tasks = r.get("n_tasks", r.get("runs", "?"))
            acc = f"{r['avg_accuracy']*100:.1f}%"
            if "std_accuracy" in r:
                acc = f"{r['avg_accuracy']*100:.1f} ± {r['std_accuracy']*100:.1f}%"
            forg = f"{r['avg_forgetting']*100:.1f}%"
            if "std_forgetting" in r:
                forg = f"{r['avg_forgetting']*100:.1f} ± {r['std_forgetting']*100:.1f}%"
            md.append(f"| {name} | {n_tasks} | {acc} | {forg} |")

    md.append("")
    md.append("## Detailed Results")
    md.append("")

    for name, r in all_results.items():
        md.append(f"### {name}")
        md.append("")
        md.append(f"```json")
        import json
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj
        md.append(json.dumps(convert(r), indent=2))
        md.append("```")
        md.append("")

    md.append("---")
    md.append("*Generated by DTG-MA benchmark suite*")

    path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote report: {path.resolve()}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="DTG-MA Full Benchmark Suite")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="split_mnist",
        choices=["split_mnist", "permuted_mnist", "cifar100", "llm_qwen25", "ablation", "scalability", "all"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for statistics")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--report", type=str, default="DTG_MA_FULL_BENCHMARK.md")

    args = parser.parse_args()

    all_results = {}

    benchmarks_to_run = []
    if args.benchmark == "all":
        benchmarks_to_run = ["split_mnist", "permuted_mnist", "cifar100", "llm_qwen25", "ablation", "scalability"]
    else:
        benchmarks_to_run = [args.benchmark]

    for bench in benchmarks_to_run:
        if bench == "split_mnist":
            all_results["split_mnist"] = run_split_mnist(
                epochs=args.epochs,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                device=args.device,
                runs=args.runs,
            )
        elif bench == "permuted_mnist":
            all_results["permuted_mnist"] = run_permuted_mnist(
                n_tasks=10,
                epochs=min(args.epochs, 50),
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                device=args.device,
                runs=args.runs,
            )
        elif bench == "cifar100":
            all_results["split_cifar100"] = run_split_cifar100(
                n_tasks=10,
                epochs=min(args.epochs, 50),
                hidden_dim=512,
                n_layers=args.n_layers,
                n_heads=8,
                device=args.device,
                runs=args.runs,
            )
        elif bench == "llm_qwen25":
            all_results["llm_qwen25"] = run_llm_qwen25(
                n_tasks=3,
                epochs=30,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                device=args.device,
                dtype=args.dtype,
            )
        elif bench == "ablation":
            all_results["ablation"] = run_ablation(
                epochs=args.epochs,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                device=args.device,
                runs=args.runs,
            )
        elif bench == "scalability":
            all_results["scalability"] = run_scalability(
                max_tasks=20,
                epochs=min(args.epochs, 50),
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                device=args.device,
            )

    if all_results:
        write_full_report(all_results, Path(args.report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

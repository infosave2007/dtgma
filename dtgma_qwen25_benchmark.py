#!/usr/bin/env python3
"""
DTG-MA Benchmark: Qwen2.5-1.5B Continual Learning

This script tests DTG-MA (Dynamic Task-Graph Masked Attention) on continual
learning benchmarks using Qwen2.5-1.5B as a frozen feature extractor.

Key difference from FCD demo:
- NO routing — only direct task_id evaluation (as required for scientific paper)
- Tests catastrophic forgetting and task isolation

Usage:
    python dtgma_qwen25_benchmark.py --benchmark split_mnist --device cpu
    python dtgma_qwen25_benchmark.py --benchmark text_domains --tasks 4 --device cpu

The test measures:
1. Average accuracy across all tasks
2. Average forgetting (max_acc - final_acc per task)
3. Task isolation (no interference between task-specific parameters)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from dtgma import DTGMAModel, DTGMABlock, train_continual, train_task, evaluate


# ============================================================================
# TEXT DOMAIN TASKS (synthetic, similar to FCD demo)
# ============================================================================

TEXT_DOMAINS = {
    0: {
        "name": "Sentiment",
        "templates": [
            ("I love this product, it's amazing!", 0),
            ("This is the worst experience ever.", 1),
            ("Absolutely fantastic service!", 0),
            ("Terrible quality, very disappointed.", 1),
            ("Great value for money!", 0),
            ("Complete waste of time and money.", 1),
            ("Highly recommend to everyone!", 0),
            ("Would not recommend to anyone.", 1),
        ],
    },
    1: {
        "name": "Topic",
        "templates": [
            ("The stock market crashed yesterday.", 0),
            ("Scientists discovered a new planet.", 1),
            ("Interest rates are rising again.", 0),
            ("New species found in Amazon rainforest.", 1),
            ("Cryptocurrency prices hit new highs.", 0),
            ("Climate change affects polar bears.", 1),
            ("The Fed announced new policies.", 0),
            ("Researchers develop new vaccine.", 1),
        ],
    },
    2: {
        "name": "Formality",
        "templates": [
            ("Dear Sir/Madam, I am writing to inquire about...", 0),
            ("Hey, what's up? Got a sec?", 1),
            ("We hereby confirm your reservation.", 0),
            ("Yo, that party was lit!", 1),
            ("Please find attached the requested documents.", 0),
            ("LOL that's hilarious bro", 1),
            ("I would like to schedule a meeting.", 0),
            ("Gonna grab some food, wanna come?", 1),
        ],
    },
    3: {
        "name": "Intent",
        "templates": [
            ("Can you help me reset my password?", 0),
            ("I want to buy the premium subscription.", 1),
            ("How do I change my email address?", 0),
            ("Add this item to my shopping cart.", 1),
            ("Why isn't my account working?", 0),
            ("I'd like to upgrade my plan.", 1),
            ("Where can I find my order history?", 0),
            ("Purchase the annual membership.", 1),
        ],
    },
    4: {
        "name": "CodeVsOps",
        "templates": [
            ("def factorial(n): return 1 if n <= 1 else n * factorial(n-1)", 0),
            ("kubectl apply -f deployment.yaml", 1),
            ("class UserService { constructor() {} }", 0),
            ("docker-compose up -d --build", 1),
            ("async function fetchData() { await api.get() }", 0),
            ("terraform init && terraform plan", 1),
            ("const result = array.map(x => x * 2).filter(x => x > 5)", 0),
            ("helm upgrade --install myapp ./chart", 1),
        ],
    },
    5: {
        "name": "FinanceVsBusiness",
        "templates": [
            ("The quarterly earnings exceeded analyst expectations.", 0),
            ("Our team needs to improve cross-functional collaboration.", 1),
            ("Portfolio diversification reduces systematic risk.", 0),
            ("The product roadmap prioritizes customer feedback.", 1),
            ("Bond yields inversely correlate with prices.", 0),
            ("Stakeholder alignment is critical for project success.", 1),
            ("Hedge funds use leverage to amplify returns.", 0),
            ("Agile methodology improves delivery velocity.", 1),
        ],
    },
    6: {
        "name": "LegalVsPolicy",
        "templates": [
            ("The defendant is hereby summoned to appear in court.", 0),
            ("All employees must complete annual compliance training.", 1),
            ("Pursuant to Section 230 of the Communications Act...", 0),
            ("Remote work is permitted with manager approval.", 1),
            ("The plaintiff seeks damages in the amount of...", 0),
            ("Expense reports must be submitted within 30 days.", 1),
            ("Habeas corpus is a fundamental constitutional right.", 0),
            ("Vacation requests require two weeks advance notice.", 1),
        ],
    },
    7: {
        "name": "ShortVsDetailed",
        "templates": [
            ("Yes.", 0),
            ("After careful consideration of all the factors involved, I believe we should proceed.", 1),
            ("OK", 0),
            ("Given the complexity of the situation, let me explain in detail.", 1),
            ("No.", 0),
            ("To properly address your question, I need to provide some background context first.", 1),
            ("Done", 0),
            ("The comprehensive analysis reveals several key insights that warrant discussion.", 1),
        ],
    },
}


def build_text_domain_dataset(
    task_id: int,
    samples_per_class: int = 100,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """Generate synthetic text samples for a domain task."""
    import random

    random.seed(seed + task_id * 1000)

    domain = TEXT_DOMAINS[task_id]
    templates = domain["templates"]

    prompts = []
    labels = []

    for _ in range(samples_per_class * 2):
        template, label = random.choice(templates)
        # Add some variation
        variations = [
            template,
            template.lower(),
            template.upper()[:50] + template[50:] if len(template) > 50 else template,
            " " + template,
            template + " ",
        ]
        prompts.append(random.choice(variations))
        labels.append(label)

    return prompts, labels


# ============================================================================
# FEATURE EXTRACTION (using Qwen2.5-1.5B)
# ============================================================================


def load_qwen_model(device: str = "cpu", dtype: str = "float32"):
    """Load Qwen2.5-1.5B as frozen feature extractor."""
    from transformers import AutoTokenizer, AutoModel

    model_name = "Qwen/Qwen2.5-1.5B"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = getattr(torch, dtype) if dtype != "auto" else torch.float32
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return tokenizer, model


def encode_texts(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cpu",
    max_length: int = 128,
    batch_size: int = 8,
) -> torch.Tensor:
    """Encode texts to embeddings using frozen LLM."""
    embeddings = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            # Mean pooling over sequence
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0)


# ============================================================================
# BENCHMARKS
# ============================================================================


def run_text_domains_benchmark(
    n_tasks: int = 4,
    train_samples: int = 100,
    test_samples: int = 50,
    epochs: int = 50,
    hidden_dim: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    device: str = "cpu",
    dtype: str = "float32",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run DTG-MA on text domain classification tasks.

    Each task is a binary classification in a specific domain.
    Model receives frozen Qwen embeddings and learns task-specific attention paths.
    """
    print("=" * 70)
    print("DTG-MA Benchmark: Text Domains (Qwen2.5-1.5B)")
    print("=" * 70)
    print(f"Tasks: {n_tasks}")
    print(f"Train samples per task: {train_samples * 2}")
    print(f"Test samples per task: {test_samples * 2}")
    print(f"Epochs per task: {epochs}")
    print(f"Device: {device}")
    print("=" * 70)

    # Load Qwen model
    tokenizer, qwen_model = load_qwen_model(device=device, dtype=dtype)

    # Build tasks
    task_ids = list(range(min(n_tasks, len(TEXT_DOMAINS))))
    tasks = {}

    print("\nBuilding datasets and extracting embeddings...")
    for tid in task_ids:
        domain_name = TEXT_DOMAINS[tid]["name"]
        print(f"  Task {tid}: {domain_name}")

        # Generate data
        train_texts, train_labels = build_text_domain_dataset(tid, train_samples, seed)
        test_texts, test_labels = build_text_domain_dataset(tid, test_samples, seed + 999)

        # Extract embeddings
        train_emb = encode_texts(qwen_model, tokenizer, train_texts, device, batch_size=8)
        test_emb = encode_texts(qwen_model, tokenizer, test_texts, device, batch_size=8)

        train_y = torch.tensor(train_labels, dtype=torch.long)
        test_y = torch.tensor(test_labels, dtype=torch.long)

        tasks[tid] = (train_emb, train_y, test_emb, test_y)

    input_dim = tasks[0][0].size(1)
    print(f"\nEmbedding dimension: {input_dim}")

    # Create DTG-MA model
    print("\nCreating DTG-MA model...")
    model = DTGMAModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=2,  # Binary classification
        n_layers=n_layers,
        n_heads=n_heads,
        n_tasks_max=max(task_ids) + 1,
        dropout=0.1,
    )

    print(f"Total params: {model.get_total_params():,}")

    # Train continual
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

    # Add metadata
    results["benchmark"] = "text_domains"
    results["n_tasks"] = n_tasks
    results["model"] = "DTG-MA"
    results["encoder"] = "Qwen2.5-1.5B"
    results["task_names"] = {tid: TEXT_DOMAINS[tid]["name"] for tid in task_ids}

    return results


def run_split_mnist_benchmark(
    epochs: int = 100,
    hidden_dim: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run DTG-MA on Split MNIST (5 binary tasks).
    """
    print("=" * 70)
    print("DTG-MA Benchmark: Split MNIST")
    print("=" * 70)

    # Import benchmark loader
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
        from benchmarks import get_split_mnist
    except ImportError:
        print("[ERROR] Cannot import benchmarks. Run from correct directory.")
        return {}

    tasks = get_split_mnist(train_samples_per_class=500, test_samples_per_class=100)

    input_dim = 784
    n_tasks = len(tasks)

    print(f"Tasks: {n_tasks}")
    print(f"Input dim: {input_dim}")
    print(f"Epochs per task: {epochs}")
    print(f"Device: {device}")
    print("=" * 70)

    # Create DTG-MA model
    model = DTGMAModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        n_layers=n_layers,
        n_heads=n_heads,
        n_tasks_max=n_tasks,
        dropout=0.1,
    )

    print(f"Total params: {model.get_total_params():,}")

    # Train continual
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

    results["benchmark"] = "split_mnist"
    results["n_tasks"] = n_tasks
    results["model"] = "DTG-MA"

    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================


def write_report(results: Dict[str, Any], path: Path):
    """Write benchmark results to Markdown file."""
    md = []
    md.append(f"# DTG-MA Benchmark Results")
    md.append("")
    md.append(f"**Benchmark:** {results.get('benchmark', 'unknown')}")
    md.append(f"**Model:** {results.get('model', 'DTG-MA')}")
    if "encoder" in results:
        md.append(f"**Encoder:** {results['encoder']}")
    md.append(f"**Tasks:** {results.get('n_tasks', '?')}")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- **Average Accuracy:** {results['avg_accuracy']*100:.2f}%")
    md.append(f"- **Average Forgetting:** {results['avg_forgetting']*100:.2f}%")
    md.append(f"- **Time:** {results.get('time', 0):.1f}s")
    md.append("")
    md.append("## Per-Task Results")
    md.append("")
    md.append("| Task | Name | Accuracy | Forgetting |")
    md.append("|---:|---|---:|---:|")

    task_names = results.get("task_names", {})
    per_task_acc = results.get("per_task_accuracy", {})
    per_task_forg = results.get("per_task_forgetting", {})

    for tid in sorted(per_task_acc.keys()):
        name = task_names.get(tid, f"Task {tid}")
        acc = per_task_acc[tid] * 100
        forg = per_task_forg.get(tid, 0) * 100
        md.append(f"| {tid} | {name} | {acc:.1f}% | {forg:.1f}% |")

    md.append("")
    md.append("## Key Findings")
    md.append("")
    if results["avg_forgetting"] < 0.01:
        md.append("✅ **Zero forgetting achieved** — DTG-MA successfully isolates task parameters.")
    elif results["avg_forgetting"] < 0.05:
        md.append("✅ **Near-zero forgetting** — minimal interference between tasks.")
    else:
        md.append(f"⚠️ Some forgetting observed ({results['avg_forgetting']*100:.1f}%).")

    md.append("")
    md.append("---")
    md.append(f"*Generated by DTG-MA benchmark suite*")

    path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote report: {path.resolve()}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="DTG-MA Benchmark (Qwen2.5-1.5B)")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="text_domains",
        choices=["text_domains", "split_mnist"],
        help="Benchmark to run",
    )
    parser.add_argument("--tasks", type=int, default=4, help="Number of tasks (text_domains)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per task")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of DTG-MA layers")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--train-samples", type=int, default=100, help="Train samples per class")
    parser.add_argument("--test-samples", type=int, default=50, help="Test samples per class")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--report", type=str, default="DTG_MA_BENCHMARK_RESULTS.md")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.benchmark == "text_domains":
        results = run_text_domains_benchmark(
            n_tasks=args.tasks,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
        )
    elif args.benchmark == "split_mnist":
        results = run_split_mnist_benchmark(
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            device=args.device,
        )
    else:
        print(f"Unknown benchmark: {args.benchmark}")
        return 1

    if results:
        write_report(results, Path(args.report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

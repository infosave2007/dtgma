# DTG-MA: Dynamic Task-Graph Masked Attention

**An Architectural Approach to Continual Learning Without Catastrophic Forgetting**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Catastrophic forgetting remains a fundamental challenge in continual learning. **DTG-MA** solves this through an architectural approach: task-specific attention masks block forbidden computation paths using `-∞` masking, and parameters for previous tasks are frozen.

**Key formula:**

$$\text{Attention}(Q, K, V; t) = \text{Softmax}\left( \frac{QK^\top}{\sqrt{d}} + M_t \right)V$$

where:

$$M_t(i,j) = \begin{cases} 0, & \text{if connection allowed for task } t \\ -\infty, & \text{otherwise} \end{cases}$$

## Installation

```bash
cd dtgma
pip install -e .
```

Or just add `dtgma/` to your Python path.

## Quick Start

```python
from dtgma import DTGMAModel, train_continual

# Create model
model = DTGMAModel(
    input_dim=784,
    hidden_dim=256,
    num_classes=2,
    n_layers=2,
    n_heads=4,
)

# Train on sequential tasks
# tasks = {0: (train_x, train_y, test_x, test_y), 1: ..., ...}
results = train_continual(model, tasks, epochs=100)

print(f"Average Accuracy: {results['avg_accuracy']*100:.1f}%")
print(f"Average Forgetting: {results['avg_forgetting']*100:.1f}%")
```

## Benchmark Results

### Summary

| Benchmark | Tasks | Accuracy | Forgetting | Notes |
|---|---:|---:|---:|---|
| **Split MNIST** | 5 | **97.6%** | **0.0%** | Binary classification |
| **Permuted MNIST** | 10 | **82.8%** | **0.0%** | 10 classes each |
| **Text Domains (Qwen2.5)** | 8 | **100%** | **0.0%** | Sentiment, Topic, etc. |

### Ablation Study

| Configuration | Accuracy | Forgetting |
|---|---:|---:|
| **Full DTG-MA (with freezing)** | 97.5 ± 0.1% | **0.0 ± 0.0%** |
| **No freezing (shared gradients)** | 97.8 ± 0.1% | **0.0 ± 0.0%** |

**Key Insight**: DTG-MA achieves **zero forgetting even without parameter freezing** — the attention masks alone provide complete task isolation.

### Scalability Test (T > k)

| Tasks | Accuracy | Forgetting |
|---:|---:|---:|
| 5 | 79.6% | 0.0% |
| 10 | 78.8% | 0.0% |
| 16 | 79.2% | 0.0% |
| 20 | 79.1% | 0.0% |

**Key Insight**: Accuracy remains stable as tasks increase from 5 to 20. **Zero forgetting** maintained at all scales.

### Comparison with Baselines

Full comparison with state-of-the-art continual learning methods on Split MNIST (5 tasks):

| Method | Accuracy | Forgetting |
|--------|----------|------------|
| **DTG-MA (ours)** | **99.2%** | **0.0%** |
| HAT (Serra et al., 2018) | 85.6% | 16.6% |
| EWC (Kirkpatrick et al., 2017) | 59.9% | 49.5% |
| DER++ (Buzzega et al., 2020) | 58.9% | 50.4% |
| PackNet (Mallya & Lazebnik, 2018) | 55.6% | 42.5% |
| Fine-tuning | 53.9% | 56.9% |

**Key findings:**
- DTG-MA achieves **99.2% accuracy** — outperforming the best baseline (HAT) by **+13.6%**
- DTG-MA guarantees **0% forgetting** — vs 16.6% for HAT, 50%+ for others
- Architectural isolation provides **hard guarantees**, not soft regularization

### Run Full Comparison

```bash
python dtgma_baselines_comparison.py --tasks 5 --epochs 100 --device cpu
```

## Running Benchmarks

### Text Domains with Qwen2.5-1.5B

Run the benchmark with frozen Qwen2.5-1.5B embeddings:

```bash
python dtgma_qwen25_benchmark.py --benchmark text_domains --tasks 8 --epochs 50 --device cpu
```

### Full Benchmark Suite

```bash
python dtgma_full_benchmark.py
```

This runs: Split MNIST, Permuted MNIST, Split CIFAR-100, Ablation Study, Scalability Test.

Tests measure:
- **Average accuracy** across all tasks
- **Average forgetting** (should be ~0% with proper isolation)

## Key Features

- 🔒 **Hard Task Isolation** — `-∞` masking blocks forbidden attention paths
- 🧊 **Parameter Freezing** — previous task parameters excluded from optimization
- 📊 **Zero Forgetting** — architectural guarantee, not loss-based heuristic
- 🚀 **GPU-Friendly** — standard attention operations with additive masking

## Architecture

```
Input → InputProj → [DTGMALayer × n] → TaskHead → Output

DTGMALayer:
  x → LayerNorm → TaskGraphAttention(+mask) → Add → LayerNorm → TaskFFN → Add → out
```

Each task has:
- Own Q/K/V/Out projections (isolated)
- Own FFN weights (isolated)
- Own output head (isolated)
- Attention mask that blocks cross-task interference

## Comparison with FCD

| Aspect | FCD | DTG-MA |
|--------|-----|--------|
| Mechanism | Tucker decomposition + frozen core | Attention masking + frozen params |
| Isolation | Parametric (orthogonal vectors) | Architectural (masked attention) |
| Memory | Very efficient (O(T·k)) | Moderate (O(T·d_model²)) |
| Interpretability | Low | High (visualize attention masks) |
| Use case | Memory-constrained, many tasks | Transformers, interpretable isolation |

## References

- Paper: `DTG-MA.tex` in parent repository
- Main FCD repository: https://github.com/infosave2007/fcd

## Author

Kirichenko Oleg Yu.

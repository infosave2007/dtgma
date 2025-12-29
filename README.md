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

### From GitHub

```bash
git clone https://github.com/infosave2007/dtgma.git
cd dtgma
pip install -e .
```

### Requirements

```bash
pip install torch>=2.0 numpy
# For LLM benchmarks:
pip install transformers torchvision
```

### Manual Installation

Or just add `dtgma/` to your Python path:

```python
import sys
sys.path.append('/path/to/dtgma')
from dtgma import DTGMAModel, train_continual
```

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

### Summary (arXiv Experiments)

| Benchmark | Tasks | Classes | Accuracy | Forgetting |
|---|---:|---:|---:|---:|
| **Split MNIST** | 5 | 2 per task | **98.9%** | **0.0%** |
| **Split CIFAR-100** | 10 | 10 per task | **52.5%** | **0.0%** |
| **Omniglot** | 10 | 20 per task | **49.6%** | **0.0%** |
| **Text Domains (Qwen2.5)** | 8 | varies | **100%** | **0.0%** |

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

#### Split MNIST (5 tasks, 2 classes each)

| Method | Accuracy | Forgetting | Params |
|--------|----------|------------|--------|
| **DTG-MA (ours)** | **98.9%** | **0.0%** | 203K |
| Adapter | 98.6% | 0.0% | 433K |
| HAT | 87.2% | 4.0% | 267K |
| LoRA | 84.9% | 16.4% | 288K |
| EWC | 74.7% | 28.2% | 267K |
| PackNet | 63.1% | 13.1% | 267K |
| Fine-tuning | 60.1% | 47.9% | 267K |
| DER++ | 58.9% | 50.2% | 267K |

#### Split CIFAR-100 (10 tasks, 10 classes each)

| Method | Accuracy | Forgetting | Params |
|--------|----------|------------|--------|
| **DTG-MA (ours)** | **52.5%** | **0.0%** | 789K |
| Adapter | 29.9% | 21.1% | 1.19M |
| HAT | 23.3% | 14.7% | 855K |
| EWC | 16.2% | 9.7% | 855K |
| LoRA | 14.9% | 38.5% | 988K |
| PackNet | 14.4% | 27.3% | 855K |
| Fine-tuning | 14.0% | 42.7% | 855K |
| DER++ | 13.7% | 40.9% | 855K |

#### Omniglot (10 tasks, 20 classes each)

| Method | Accuracy | Forgetting | Params |
|--------|----------|------------|--------|
| **DTG-MA (ours)** | **49.6%** | **0.0%** | 203K |
| Adapter | 15.2% | 17.0% | 603K |
| LoRA | 10.9% | 21.0% | 313K |
| PackNet | 10.8% | 6.8% | 272K |
| DER++ | 9.5% | 42.9% | 272K |
| Fine-tuning | 9.2% | 14.3% | 272K |
| HAT | 8.1% | 4.2% | 272K |
| EWC | 6.3% | 1.3% | 272K |

**Key findings:**
- DTG-MA achieves **98.9%** on Split MNIST (+11.7% vs HAT)
- DTG-MA achieves **52.5%** on Split CIFAR-100 (+22.6% vs Adapter)
- DTG-MA achieves **49.6%** on Omniglot (+34.4% vs Adapter)
- **0% forgetting** on all benchmarks — architectural guarantee, not soft regularization

### Run arXiv Experiments

Run the complete benchmark suite used in the paper:

```bash
# Run all benchmarks (Split MNIST, Split CIFAR-100, Omniglot)
python arxiv_experiments.py --epochs 30 --runs 1

# Run specific benchmark
python arxiv_experiments.py --benchmarks split_mnist --epochs 30
python arxiv_experiments.py --benchmarks split_cifar100 --epochs 30
python arxiv_experiments.py --benchmarks omniglot --epochs 30

# Run on GPU (if available)
python arxiv_experiments.py --device cuda --epochs 30
```

Results are saved to `ARXIV_RESULTS.md`.

### Run Baselines Comparison

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

- **Paper**: Kirichenko, O. (2025). *Dynamic Task-Graph Masked Attention (DTG-MA): An Architectural Approach to Continual Learning Without Catastrophic Forgetting*. Zenodo. [DOI: 10.5281/zenodo.17921784](https://doi.org/10.5281/zenodo.17921784)
- **FCD repository**: https://github.com/infosave2007/fcd

## Citation

```bibtex
@software{kirichenko2025dtgma,
  author = {Kirichenko, Oleg Yu.},
  title = {DTG-MA: Dynamic Task-Graph Masked Attention},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.17921784},
  url = {https://github.com/infosave2007/dtgma}
}
```

## Author

Kirichenko Oleg Yu.

## License

MIT License - see [LICENSE](LICENSE) for details.

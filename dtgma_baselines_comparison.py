#!/usr/bin/env python3
"""
DTG-MA vs Baselines Comparison

Full comparison with all continual learning baseline methods:
- Fine-tuning (no protection)
- EWC (Elastic Weight Consolidation)
- HAT (Hard Attention to the Task)
- PackNet (Network Pruning)
- DER++ (Dark Experience Replay)
- DTG-MA (ours)

Benchmarks: Split MNIST (5 tasks)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
from typing import Dict, Tuple

# DTG-MA
from dtgma import (
    DTGMAModel,
    train_continual,
    # Baselines
    FineTuneModel,
    EWCModel,
    HATModel,
    PackNetModel,
    DERPPModel,
    train_continual_baseline,
    train_hat_continual,
    train_packnet_continual,
    train_derpp_continual,
)


def load_mnist():
    """Load MNIST dataset."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_x = train_dataset.data.float().view(-1, 784) / 255.0
    train_y = train_dataset.targets
    test_x = test_dataset.data.float().view(-1, 784) / 255.0
    test_y = test_dataset.targets
    
    return train_x, train_y, test_x, test_y


def create_split_mnist_tasks(n_tasks: int = 5) -> Dict[int, Tuple]:
    """Create Split MNIST tasks (binary classification)."""
    train_x, train_y, test_x, test_y = load_mnist()
    
    tasks = {}
    for task_id in range(n_tasks):
        class_a = task_id * 2
        class_b = task_id * 2 + 1
        
        # Train
        train_mask = (train_y == class_a) | (train_y == class_b)
        task_train_x = train_x[train_mask]
        task_train_y = (train_y[train_mask] == class_b).long()
        
        # Test
        test_mask = (test_y == class_a) | (test_y == class_b)
        task_test_x = test_x[test_mask]
        task_test_y = (test_y[test_mask] == class_b).long()
        
        tasks[task_id] = (task_train_x, task_train_y, task_test_x, task_test_y)
    
    return tasks


def run_comparison(
    n_tasks: int = 5,
    epochs: int = 100,
    hidden_dim: int = 256,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict:
    """Run full comparison of all methods."""
    
    print("=" * 60)
    print("DTG-MA vs Baselines Comparison")
    print("=" * 60)
    print(f"Tasks: {n_tasks}, Epochs: {epochs}, Device: {device}")
    print()
    
    # Load data
    print("Loading Split MNIST...")
    tasks_data = create_split_mnist_tasks(n_tasks)
    print(f"Created {len(tasks_data)} binary classification tasks")
    print()
    
    results = {}
    input_dim = 784
    num_classes = 2
    
    # 1. Fine-tuning
    print("-" * 40)
    print("1. Fine-tuning (baseline)")
    print("-" * 40)
    ft_model = FineTuneModel(input_dim, hidden_dim, num_classes)
    t0 = time.time()
    ft_results = train_continual_baseline(
        ft_model, tasks_data, epochs=epochs, lr=0.01, device=device, verbose=verbose
    )
    ft_time = time.time() - t0
    results['Fine-tuning'] = {
        'accuracy': ft_results['avg_accuracy'],
        'forgetting': ft_results['avg_forgetting'],
        'time': ft_time
    }
    print(f"Time: {ft_time:.1f}s")
    print()
    
    # 2. EWC
    print("-" * 40)
    print("2. EWC (Elastic Weight Consolidation)")
    print("-" * 40)
    ewc_model = EWCModel(input_dim, hidden_dim, num_classes, ewc_lambda=1000)
    t0 = time.time()
    ewc_results = train_continual_baseline(
        ewc_model, tasks_data, epochs=epochs, lr=0.01, device=device, verbose=verbose
    )
    ewc_time = time.time() - t0
    results['EWC'] = {
        'accuracy': ewc_results['avg_accuracy'],
        'forgetting': ewc_results['avg_forgetting'],
        'time': ewc_time
    }
    print(f"Time: {ewc_time:.1f}s")
    print()
    
    # 3. HAT
    print("-" * 40)
    print("3. HAT (Hard Attention to the Task)")
    print("-" * 40)
    hat_model = HATModel(input_dim, hidden_dim, num_classes)
    t0 = time.time()
    hat_results = train_hat_continual(
        hat_model, tasks_data, epochs=epochs, lr=0.01, device=device, verbose=verbose
    )
    hat_time = time.time() - t0
    results['HAT'] = {
        'accuracy': hat_results['avg_accuracy'],
        'forgetting': hat_results['avg_forgetting'],
        'time': hat_time
    }
    print(f"Time: {hat_time:.1f}s")
    print()
    
    # 4. PackNet
    print("-" * 40)
    print("4. PackNet (Network Pruning)")
    print("-" * 40)
    packnet_model = PackNetModel(input_dim, hidden_dim, num_classes)
    t0 = time.time()
    packnet_results = train_packnet_continual(
        packnet_model, tasks_data, epochs=epochs, lr=0.01, device=device, verbose=verbose
    )
    packnet_time = time.time() - t0
    results['PackNet'] = {
        'accuracy': packnet_results['avg_accuracy'],
        'forgetting': packnet_results['avg_forgetting'],
        'time': packnet_time
    }
    print(f"Time: {packnet_time:.1f}s")
    print()
    
    # 5. DER++
    print("-" * 40)
    print("5. DER++ (Dark Experience Replay)")
    print("-" * 40)
    derpp_model = DERPPModel(input_dim, hidden_dim, num_classes, buffer_size=500)
    t0 = time.time()
    derpp_results = train_derpp_continual(
        derpp_model, tasks_data, epochs=epochs, lr=0.01, device=device, verbose=verbose
    )
    derpp_time = time.time() - t0
    results['DER++'] = {
        'accuracy': derpp_results['avg_accuracy'],
        'forgetting': derpp_results['avg_forgetting'],
        'time': derpp_time
    }
    print(f"Time: {derpp_time:.1f}s")
    print()
    
    # 6. DTG-MA (ours)
    print("-" * 40)
    print("6. DTG-MA (ours)")
    print("-" * 40)
    dtgma_model = DTGMAModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        n_layers=2,
        n_heads=4,
    )
    t0 = time.time()
    dtgma_results = train_continual(
        dtgma_model, tasks_data, epochs=epochs, lr=0.01, device=device, verbose=verbose
    )
    dtgma_time = time.time() - t0
    results['DTG-MA (ours)'] = {
        'accuracy': dtgma_results['avg_accuracy'],
        'forgetting': dtgma_results['avg_forgetting'],
        'time': dtgma_time
    }
    print(f"Time: {dtgma_time:.1f}s")
    print()
    
    return results


def print_results_table(results: Dict):
    """Print results as markdown table."""
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print()
    print("| Method | Accuracy | Forgetting | Time |")
    print("|--------|----------|------------|------|")
    
    # Sort by accuracy descending
    sorted_methods = sorted(results.keys(), key=lambda x: results[x]['accuracy'], reverse=True)
    
    for method in sorted_methods:
        r = results[method]
        acc = r['accuracy'] * 100
        fgt = r['forgetting'] * 100
        tm = r['time']
        
        # Bold for DTG-MA
        if 'DTG-MA' in method:
            print(f"| **{method}** | **{acc:.1f}%** | **{fgt:.1f}%** | {tm:.1f}s |")
        else:
            print(f"| {method} | {acc:.1f}% | {fgt:.1f}% | {tm:.1f}s |")
    
    print()


def save_results(results: Dict, output_file: str):
    """Save results to markdown file."""
    with open(output_file, 'w') as f:
        f.write("# DTG-MA vs Baselines Comparison\n\n")
        f.write("## Split MNIST (5 tasks, binary classification)\n\n")
        f.write("| Method | Accuracy | Forgetting | Time |\n")
        f.write("|--------|----------|------------|------|\n")
        
        sorted_methods = sorted(results.keys(), key=lambda x: results[x]['accuracy'], reverse=True)
        
        for method in sorted_methods:
            r = results[method]
            acc = r['accuracy'] * 100
            fgt = r['forgetting'] * 100
            tm = r['time']
            
            if 'DTG-MA' in method:
                f.write(f"| **{method}** | **{acc:.1f}%** | **{fgt:.1f}%** | {tm:.1f}s |\n")
            else:
                f.write(f"| {method} | {acc:.1f}% | {fgt:.1f}% | {tm:.1f}s |\n")
        
        f.write("\n## Key Findings\n\n")
        
        dtgma = results.get('DTG-MA (ours)', {})
        if dtgma:
            f.write(f"- **DTG-MA achieves {dtgma['accuracy']*100:.1f}% accuracy with {dtgma['forgetting']*100:.1f}% forgetting**\n")
            
            # Compare with best baseline
            baselines = {k: v for k, v in results.items() if 'DTG-MA' not in k}
            if baselines:
                best_baseline = max(baselines.keys(), key=lambda x: baselines[x]['accuracy'])
                best_acc = baselines[best_baseline]['accuracy']
                best_fgt = baselines[best_baseline]['forgetting']
                
                acc_gain = (dtgma['accuracy'] - best_acc) * 100
                fgt_reduction = (best_fgt - dtgma['forgetting']) * 100
                
                f.write(f"- DTG-MA outperforms {best_baseline} by {acc_gain:+.1f}% accuracy\n")
                if fgt_reduction > 0:
                    f.write(f"- DTG-MA reduces forgetting by {fgt_reduction:.1f}% compared to baselines\n")
        
        f.write("\n---\n*Generated by DTG-MA benchmark suite*\n")
    
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='DTG-MA vs Baselines Comparison')
    parser.add_argument('--tasks', type=int, default=5, help='Number of tasks')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per task')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--output', type=str, default='DTG_MA_BASELINES_COMPARISON.md',
                        help='Output file')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    
    args = parser.parse_args()
    
    results = run_comparison(
        n_tasks=args.tasks,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
        verbose=not args.quiet
    )
    
    print_results_table(results)
    save_results(results, args.output)


if __name__ == '__main__':
    main()

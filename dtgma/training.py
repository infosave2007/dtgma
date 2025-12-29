"""
DTG-MA Training utilities for continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any
import time


def train_task(
    model: nn.Module,
    task_id: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.01,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Train the DTG-MA model on a single task.

    Args:
        model: DTGMAModel instance
        task_id: Task identifier
        train_x: Training features
        train_y: Training labels
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        verbose: Print progress

    Returns:
        Dictionary with training metrics
    """
    model = model.to(device)
    model.train()

    # Register task if needed
    if task_id not in model.registered_tasks:
        model.register_task(task_id)

    # Get task-specific parameters
    task_params = model.get_task_parameters(task_id)
    if not task_params:
        if verbose:
            print(f"  [warn] No trainable parameters for task {task_id}")
        return {"loss": 0.0, "accuracy": 0.0}

    optimizer = torch.optim.Adam(task_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    dataset = TensorDataset(train_x.to(device), train_y.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            logits = model(batch_x, task_id)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_samples += batch_x.size(0)

        total_loss = epoch_loss / epoch_samples
        total_correct = epoch_correct
        total_samples = epoch_samples

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            acc = epoch_correct / epoch_samples * 100
            print(f"    epoch {epoch+1}/{epochs}: loss={total_loss:.4f}, acc={acc:.1f}%")

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {"loss": total_loss, "accuracy": accuracy}


def evaluate(
    model: nn.Module,
    task_id: int,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    """
    Evaluate the model on a task.

    Args:
        model: DTGMAModel instance
        task_id: Task identifier
        test_x: Test features
        test_y: Test labels
        batch_size: Batch size
        device: Device

    Returns:
        Accuracy (0-1)
    """
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(test_x.to(device), test_y.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x, task_id)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

    return correct / total if total > 0 else 0.0


def train_continual(
    model: nn.Module,
    tasks: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.01,
    device: str = "cpu",
    verbose: bool = True,
    freeze_after_training: bool = True,
) -> Dict[str, Any]:
    """
    Train the DTG-MA model on multiple tasks sequentially (continual learning).

    Args:
        model: DTGMAModel instance
        tasks: Dictionary mapping task_id -> (train_x, train_y, test_x, test_y)
        epochs: Epochs per task
        batch_size: Batch size
        lr: Learning rate
        device: Device
        verbose: Print progress
        freeze_after_training: Freeze task parameters after training

    Returns:
        Dictionary with:
            - avg_accuracy: Average accuracy across all tasks
            - avg_forgetting: Average forgetting
            - per_task_accuracy: Accuracy for each task after all training
            - accuracy_matrix: Full accuracy matrix [task_trained][task_evaluated]
    """
    task_ids = sorted(tasks.keys())
    n_tasks = len(task_ids)

    # Accuracy matrix: [after training task i][evaluate on task j]
    accuracy_matrix = torch.zeros(n_tasks, n_tasks)

    if verbose:
        print(f"\n{'='*60}")
        print(f"DTG-MA Continual Learning: {n_tasks} tasks")
        print(f"{'='*60}")

    start_time = time.time()

    for i, tid in enumerate(task_ids):
        train_x, train_y, test_x, test_y = tasks[tid]

        if verbose:
            print(f"\n[Task {tid}] Training ({len(train_x)} samples)...")

        # Train on current task
        train_task(
            model,
            task_id=tid,
            train_x=train_x,
            train_y=train_y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            verbose=verbose,
        )

        # Freeze task parameters to prevent interference
        if freeze_after_training:
            model.freeze_task(tid)
            if verbose:
                print(f"  Task {tid} frozen.")

        # Evaluate on all tasks seen so far
        for j, prev_tid in enumerate(task_ids[: i + 1]):
            _, _, prev_test_x, prev_test_y = tasks[prev_tid]
            acc = evaluate(model, prev_tid, prev_test_x, prev_test_y, device=device)
            accuracy_matrix[i, j] = acc

            if verbose:
                print(f"  Eval task {prev_tid}: {acc*100:.1f}%")

    elapsed = time.time() - start_time

    # Compute metrics
    # Final accuracy per task (after all training)
    final_accuracies = accuracy_matrix[-1, :].tolist()

    # Forgetting: max accuracy - final accuracy
    max_accuracies = accuracy_matrix.max(dim=0).values
    forgetting = (max_accuracies - accuracy_matrix[-1, :]).clamp(min=0.0)

    avg_accuracy = float(accuracy_matrix[-1, :].mean())
    avg_forgetting = float(forgetting.mean())

    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Average Accuracy:   {avg_accuracy*100:.1f}%")
        print(f"Average Forgetting: {avg_forgetting*100:.1f}%")
        print(f"Time: {elapsed:.1f}s")

        print("\nPer-task final accuracy:")
        for j, tid in enumerate(task_ids):
            print(f"  Task {tid}: {final_accuracies[j]*100:.1f}%")

        print("\nPer-task forgetting:")
        for j, tid in enumerate(task_ids):
            print(f"  Task {tid}: {forgetting[j].item()*100:.1f}%")

    return {
        "avg_accuracy": avg_accuracy,
        "avg_forgetting": avg_forgetting,
        "per_task_accuracy": {tid: final_accuracies[j] for j, tid in enumerate(task_ids)},
        "per_task_forgetting": {tid: forgetting[j].item() for j, tid in enumerate(task_ids)},
        "accuracy_matrix": accuracy_matrix.tolist(),
        "time": elapsed,
    }

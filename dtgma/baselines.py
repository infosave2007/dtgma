"""
Baseline Methods for Continual Learning Comparison with DTG-MA

Based on FCD baselines implementation:
- Fine-tuning (No protection)
- EWC (Elastic Weight Consolidation) 
- HAT (Hard Attention to the Task)
- PackNet (Network Pruning)
- DER++ (Dark Experience Replay)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from copy import deepcopy
import numpy as np


class FineTuneModel(nn.Module):
    """
    Simple fine-tuning baseline (no continual learning protection).
    Each new task just overwrites previous knowledge.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.name = "Fine-Tune"
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.classifier(h)


class EWCModel(nn.Module):
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).
    
    Protects important weights from previous tasks using Fisher information.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_classes: int,
        ewc_lambda: float = 1000.0
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.ewc_lambda = ewc_lambda
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optpar_dict: Dict[str, torch.Tensor] = {}
        self.name = f"EWC-λ{ewc_lambda}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.classifier(h)
    
    def compute_fisher(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        n_samples: int = None
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix diagonal."""
        self.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        
        n_samples = n_samples or len(x)
        
        for i in range(min(n_samples, len(x))):
            self.zero_grad()
            output = self(x[i:i+1])
            log_prob = F.log_softmax(output, dim=1)
            loss = -log_prob[0, y[i]]
            loss.backward()
            
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        
        for n in fisher:
            fisher[n] /= n_samples
            
        return fisher
    
    def consolidate(self, x: torch.Tensor, y: torch.Tensor, n_samples: int = None):
        """Store optimal parameters and Fisher information after training on a task."""
        new_fisher = self.compute_fisher(x, y, n_samples)
        
        for n, p in self.named_parameters():
            if p.requires_grad:
                if n in self.fisher_dict:
                    self.fisher_dict[n] = self.fisher_dict[n] + new_fisher[n]
                else:
                    self.fisher_dict[n] = new_fisher[n].clone()
                self.optpar_dict[n] = p.data.clone()
    
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self.fisher_dict:
            return torch.tensor(0.0)
        
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self.fisher_dict:
                loss += (self.fisher_dict[n] * (p - self.optpar_dict[n]) ** 2).sum()
        
        return self.ewc_lambda * loss


class HATModel(nn.Module):
    """
    Hard Attention to the Task (Serra et al., ICML 2018).
    
    Uses learnable attention masks per task to protect previous knowledge.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, s_max: float = 400.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.s_max = s_max
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.task_embeddings: Dict[int, Dict[str, nn.Parameter]] = {}
        self.masks: Dict[int, Dict[str, torch.Tensor]] = {}
        
        self.current_task = None
        self.name = "HAT"
    
    def register_task(self, task_id: int):
        """Register attention embeddings for new task."""
        if task_id in self.task_embeddings:
            return

        device = next(self.parameters()).device
        
        self.task_embeddings[task_id] = {
            'e1': nn.Parameter(torch.randn(self.hidden_dim, device=device) * 0.1),
            'e2': nn.Parameter(torch.randn(self.hidden_dim, device=device) * 0.1)
        }
        for name, param in self.task_embeddings[task_id].items():
            self.register_parameter(f'task_{task_id}_{name}', param)
        
        self.current_task = task_id
    
    def get_attention(self, task_id: int, s: float = 1.0) -> Dict[str, torch.Tensor]:
        """Get attention masks for task."""
        embeddings = self.task_embeddings[task_id]
        return {
            'a1': torch.sigmoid(s * embeddings['e1']),
            'a2': torch.sigmoid(s * embeddings['e2'])
        }
    
    def forward(self, x: torch.Tensor, task_id: int = None, s: float = 1.0) -> torch.Tensor:
        if task_id is None:
            task_id = self.current_task
        
        att = self.get_attention(task_id, s)
        
        h = F.relu(self.fc1(x)) * att['a1']
        h = F.relu(self.fc2(h)) * att['a2']
        return self.classifier(h)
    
    def freeze_task(self, task_id: int):
        """Store hard masks after task training."""
        with torch.no_grad():
            att = self.get_attention(task_id, self.s_max)
            self.masks[task_id] = {
                'a1': (att['a1'] > 0.5).float(),
                'a2': (att['a2'] > 0.5).float()
            }
    
    def hat_reg_loss(self, task_id: int, s: float) -> torch.Tensor:
        """Regularization to encourage sparse masks."""
        att = self.get_attention(task_id, s)
        reg = att['a1'].sum() + att['a2'].sum()
        
        overlap = 0.0
        for prev_id, prev_mask in self.masks.items():
            if prev_id != task_id:
                overlap += (att['a1'] * prev_mask['a1']).sum()
                overlap += (att['a2'] * prev_mask['a2']).sum()
        
        return 0.01 * reg + 10.0 * overlap


class PackNetModel(nn.Module):
    """
    PackNet: Pruning-based continual learning (Mallya & Lazebnik, 2018).
    
    After each task, prunes unimportant weights and freezes the rest.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, prune_ratio: float = 0.5):
        super().__init__()
        self.prune_ratio = prune_ratio
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.register_buffer('mask_fc1', torch.ones_like(self.fc1.weight))
        self.register_buffer('mask_fc2', torch.ones_like(self.fc2.weight))
        self.register_buffer('frozen_mask_fc1', torch.zeros_like(self.fc1.weight))
        self.register_buffer('frozen_mask_fc2', torch.zeros_like(self.fc2.weight))
        
        self.name = "PackNet"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self.fc1.weight * self.mask_fc1
        w2 = self.fc2.weight * self.mask_fc2
        
        h = F.relu(F.linear(x, w1, self.fc1.bias))
        h = F.relu(F.linear(h, w2, self.fc2.bias))
        return self.classifier(h)
    
    def prune_and_freeze(self):
        """Prune smallest weights and freeze important ones."""
        for name, param in [('fc1', self.fc1.weight), ('fc2', self.fc2.weight)]:
            if name == 'fc1':
                mask = self.mask_fc1
                frozen = self.frozen_mask_fc1
            else:
                mask = self.mask_fc2
                frozen = self.frozen_mask_fc2

            available = mask * (1 - frozen)
            
            if available.sum() == 0:
                continue
            
            magnitudes = (param.abs() * available).flatten()
            nonzero_mags = magnitudes[magnitudes > 0]
            
            if len(nonzero_mags) == 0:
                continue
            
            k = int(len(nonzero_mags) * self.prune_ratio)
            if k > 0:
                threshold = torch.kthvalue(nonzero_mags.detach().to('cpu'), k).values.to(param.device)
                prune_mask = (param.abs() <= threshold) & (available > 0)
                mask[prune_mask] = 0
            
            keep_mask = (mask > 0) & (available > 0)
            frozen[keep_mask] = 1


class ReplayBuffer:
    """Experience replay buffer storing samples and logits."""
    
    def __init__(self, capacity: int, input_dim: int, num_classes: int):
        self.capacity = capacity
        self.buffer_x = torch.zeros(capacity, input_dim)
        self.buffer_y = torch.zeros(capacity, dtype=torch.long)
        self.buffer_logits = torch.zeros(capacity, num_classes)
        self.count = 0
        self.ptr = 0
    
    def add(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor):
        """Add samples to buffer."""
        batch_size = x.size(0)
        for i in range(batch_size):
            self.buffer_x[self.ptr] = x[i].detach().to('cpu')
            self.buffer_y[self.ptr] = y[i].detach().to('cpu')
            self.buffer_logits[self.ptr] = logits[i].detach().to('cpu')
            self.ptr = (self.ptr + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from buffer."""
        idx = np.random.choice(self.count, min(batch_size, self.count), replace=False)
        return self.buffer_x[idx], self.buffer_y[idx], self.buffer_logits[idx]


class DERPPModel(nn.Module):
    """
    DER++: Dark Experience Replay (Buzzega et al., NeurIPS 2020).
    
    Stores past samples with their logits and replays them during training.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_classes: int,
        buffer_size: int = 500,
        alpha: float = 0.5,
        beta: float = 0.5
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.buffer = ReplayBuffer(buffer_size, input_dim, num_classes)
        self.name = "DER++"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
    
    def observe(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Training step with replay."""
        self.train()
        optimizer.zero_grad()
        
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        if self.buffer.count > 0:
            buf_x, buf_y, buf_logits = self.buffer.sample(min(32, self.buffer.count))
            buf_x = buf_x.to(x.device)
            buf_y = buf_y.to(x.device)
            buf_logits = buf_logits.to(x.device)

            replay_logits = self(buf_x)
            replay_ce = F.cross_entropy(replay_logits, buf_y)
            replay_mse = F.mse_loss(replay_logits, buf_logits)
            
            loss += self.beta * replay_ce + self.alpha * replay_mse
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            self.buffer.add(x.detach(), y.detach(), logits.detach())
        
        return loss.item()


# =============================================================================
# Training Functions
# =============================================================================

def train_baseline(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = None,
    verbose: bool = False,
    device: str = 'cpu'
) -> Dict:
    """Train a baseline model on a single task."""
    model = model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_samples = len(train_x)
    batch_size = batch_size or n_samples
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]
            
            optimizer.zero_grad()
            
            output = model(batch_x)
            task_loss = F.cross_entropy(output, batch_y)
            
            ewc_loss = torch.tensor(0.0)
            if hasattr(model, 'ewc_loss'):
                ewc_loss = model.ewc_loss()
            
            total_loss = task_loss + ewc_loss
            total_loss.backward()
            optimizer.step()
    
    return {}


def evaluate_baseline(
    model: nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: str = 'cpu'
) -> float:
    """Evaluate model accuracy."""
    model = model.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(test_x)
        preds = output.argmax(dim=1)
        return (preds == test_y).float().mean().item()


def train_continual_baseline(
    model: nn.Module,
    tasks_data: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = None,
    verbose: bool = True,
    device: str = 'cpu'
) -> Dict:
    """Train baseline model on multiple tasks sequentially."""
    results = {
        'accuracies': {},
        'forgetting': {},
        'final_accuracies': {},
    }
    
    task_ids = sorted(tasks_data.keys())
    
    for i, task_id in enumerate(task_ids):
        train_x, train_y, test_x, test_y = tasks_data[task_id]
        
        if verbose:
            print(f"\n[{model.name}] Training Task {task_id}")
        
        train_baseline(
            model, train_x, train_y,
            epochs=epochs, lr=lr, batch_size=batch_size,
            verbose=verbose, device=device
        )
        
        if hasattr(model, 'consolidate'):
            model.consolidate(train_x.to(device), train_y.to(device))
        
        for prev_id in task_ids[:i+1]:
            _, _, prev_test_x, prev_test_y = tasks_data[prev_id]
            acc = evaluate_baseline(model, prev_test_x, prev_test_y, device)
            results['accuracies'][(task_id, prev_id)] = acc
            
            if verbose:
                print(f"  Task {prev_id} accuracy: {acc*100:.1f}%")
    
    for task_id in task_ids[:-1]:
        initial = results['accuracies'][(task_id, task_id)]
        final = results['accuracies'][(task_ids[-1], task_id)]
        results['forgetting'][task_id] = max(0, initial - final)
    
    for task_id in task_ids:
        results['final_accuracies'][task_id] = results['accuracies'][(task_ids[-1], task_id)]
    
    avg_acc = sum(results['final_accuracies'].values()) / len(task_ids)
    avg_forget = sum(results['forgetting'].values()) / len(results['forgetting']) if results['forgetting'] else 0
    
    results['avg_accuracy'] = avg_acc
    results['avg_forgetting'] = avg_forget
    
    if verbose:
        print(f"\n[{model.name}] Summary: Acc={avg_acc*100:.1f}%, Forget={avg_forget*100:.1f}%")
    
    return results


def train_hat_continual(
    model: HATModel,
    tasks_data: Dict[int, Tuple],
    epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
    device: str = 'cpu'
) -> Dict:
    """Train HAT model on sequential tasks."""
    results = {'accuracies': {}, 'forgetting': {}}
    task_ids = sorted(tasks_data.keys())

    model = model.to(device)
    
    for i, task_id in enumerate(task_ids):
        train_x, train_y, test_x, test_y = tasks_data[task_id]
        model.register_task(task_id)
        
        params = list(model.fc1.parameters()) + list(model.fc2.parameters()) + \
                 list(model.classifier.parameters()) + \
                 list(model.task_embeddings[task_id].values())
        
        optimizer = torch.optim.Adam(params, lr=lr)
        
        for epoch in range(epochs):
            model.train()
            s = 1.0 + (model.s_max - 1.0) * epoch / epochs
            
            optimizer.zero_grad()
            output = model(train_x.to(device), task_id, s)
            ce_loss = F.cross_entropy(output, train_y.to(device))
            hat_loss = model.hat_reg_loss(task_id, s)
            
            loss = ce_loss + hat_loss
            loss.backward()
            
            for prev_id, prev_mask in model.masks.items():
                model.fc1.weight.grad *= (1 - prev_mask['a1'].unsqueeze(1))
                model.fc2.weight.grad *= (1 - prev_mask['a2'].unsqueeze(1))
            
            optimizer.step()
        
        model.freeze_task(task_id)
        
        model.eval()
        for prev_id in task_ids[:i+1]:
            _, _, tx, ty = tasks_data[prev_id]
            with torch.no_grad():
                preds = model(tx.to(device), prev_id, model.s_max).argmax(1)
                results['accuracies'][(task_id, prev_id)] = (preds == ty.to(device)).float().mean().item()
        
        if verbose:
            print(f"[HAT] Task {task_id}: {results['accuracies'][(task_id, task_id)]*100:.1f}%")
    
    for tid in task_ids[:-1]:
        initial = results['accuracies'][(tid, tid)]
        final = results['accuracies'][(task_ids[-1], tid)]
        results['forgetting'][tid] = max(0, initial - final)
    
    final_accs = [results['accuracies'][(task_ids[-1], tid)] for tid in task_ids]
    results['avg_accuracy'] = np.mean(final_accs)
    results['avg_forgetting'] = np.mean(list(results['forgetting'].values())) if results['forgetting'] else 0
    
    return results


def train_packnet_continual(
    model: PackNetModel,
    tasks_data: Dict[int, Tuple],
    epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
    device: str = 'cpu'
) -> Dict:
    """Train PackNet on sequential tasks."""
    results = {'accuracies': {}, 'forgetting': {}}
    task_ids = sorted(tasks_data.keys())

    model = model.to(device)
    
    for i, task_id in enumerate(task_ids):
        train_x, train_y, test_x, test_y = tasks_data[task_id]
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            output = model(train_x.to(device))
            loss = F.cross_entropy(output, train_y.to(device))
            loss.backward()
            
            if model.fc1.weight.grad is not None:
                model.fc1.weight.grad *= (1 - model.frozen_mask_fc1)
            if model.fc2.weight.grad is not None:
                model.fc2.weight.grad *= (1 - model.frozen_mask_fc2)
            
            optimizer.step()
        
        model.prune_and_freeze()
        
        model.eval()
        for prev_id in task_ids[:i+1]:
            _, _, tx, ty = tasks_data[prev_id]
            with torch.no_grad():
                preds = model(tx.to(device)).argmax(1)
                results['accuracies'][(task_id, prev_id)] = (preds == ty.to(device)).float().mean().item()
        
        if verbose:
            print(f"[PackNet] Task {task_id}: {results['accuracies'][(task_id, task_id)]*100:.1f}%")
    
    for tid in task_ids[:-1]:
        initial = results['accuracies'][(tid, tid)]
        final = results['accuracies'][(task_ids[-1], tid)]
        results['forgetting'][tid] = max(0, initial - final)
    
    final_accs = [results['accuracies'][(task_ids[-1], tid)] for tid in task_ids]
    results['avg_accuracy'] = np.mean(final_accs)
    results['avg_forgetting'] = np.mean(list(results['forgetting'].values())) if results['forgetting'] else 0
    
    return results


def train_derpp_continual(
    model: DERPPModel,
    tasks_data: Dict[int, Tuple],
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 32,
    verbose: bool = True,
    device: str = 'cpu'
) -> Dict:
    """Train DER++ on sequential tasks."""
    results = {'accuracies': {}, 'forgetting': {}}
    task_ids = sorted(tasks_data.keys())
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i, task_id in enumerate(task_ids):
        train_x, train_y, test_x, test_y = tasks_data[task_id]
        n_samples = len(train_x)
        
        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_x = train_x[indices[start:end]].to(device)
                batch_y = train_y[indices[start:end]].to(device)
                model.observe(batch_x, batch_y, optimizer)
        
        model.eval()
        for prev_id in task_ids[:i+1]:
            _, _, tx, ty = tasks_data[prev_id]
            with torch.no_grad():
                preds = model(tx.to(device)).argmax(1)
                results['accuracies'][(task_id, prev_id)] = (preds == ty.to(device)).float().mean().item()
        
        if verbose:
            print(f"[DER++] Task {task_id}: {results['accuracies'][(task_id, task_id)]*100:.1f}%")
    
    for tid in task_ids[:-1]:
        initial = results['accuracies'][(tid, tid)]
        final = results['accuracies'][(task_ids[-1], tid)]
        results['forgetting'][tid] = max(0, initial - final)
    
    final_accs = [results['accuracies'][(task_ids[-1], tid)] for tid in task_ids]
    results['avg_accuracy'] = np.mean(final_accs)
    results['avg_forgetting'] = np.mean(list(results['forgetting'].values())) if results['forgetting'] else 0
    
    return results

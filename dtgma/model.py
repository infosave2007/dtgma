"""
DTG-MA Core Model Components

Dynamic Task-Graph Masked Attention for Continual Learning.

Key principles:
1. Parameters as task-annotated computation graph
2. Task-specific attention masking with -inf for forbidden paths
3. Freezing of previous task parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class TaskGraphAttention(nn.Module):
    """
    Attention module with task-specific masking using -inf for forbidden paths.

    Implements:
        Attention(Q, K, V; t) = Softmax((QK^T / √d) + M_t) V

    where M_t(i,j) = 0 if edge (i,j) allowed for task t, else -inf

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_tasks_max: Maximum number of tasks
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_tasks_max: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_tasks_max = n_tasks_max
        self.dropout = nn.Dropout(dropout)

        # Task-specific Q, K, V projections
        self.Q_projections: nn.ModuleDict = nn.ModuleDict()
        self.K_projections: nn.ModuleDict = nn.ModuleDict()
        self.V_projections: nn.ModuleDict = nn.ModuleDict()
        self.out_projections: nn.ModuleDict = nn.ModuleDict()

        # Task masks: initialized to -inf (all blocked)
        self.register_buffer(
            "task_masks", torch.full((n_tasks_max, n_heads, 1, 1), float("-inf"))
        )

        self.registered_tasks: List[int] = []

    def register_task(self, task_id: int, seq_len: int = 1):
        """Register a new task with its own projection matrices."""
        if str(task_id) in self.Q_projections:
            return

        device = self.task_masks.device

        self.Q_projections[str(task_id)] = nn.Linear(self.d_model, self.d_model).to(device)
        self.K_projections[str(task_id)] = nn.Linear(self.d_model, self.d_model).to(device)
        self.V_projections[str(task_id)] = nn.Linear(self.d_model, self.d_model).to(device)
        self.out_projections[str(task_id)] = nn.Linear(self.d_model, self.d_model).to(device)

        # Initialize mask for this task (allow all connections)
        self.task_masks[task_id, :, :seq_len, :seq_len] = 0.0

        self.registered_tasks.append(task_id)

    def set_mask(self, task_id: int, mask: torch.Tensor):
        """Set custom binary mask (1=allow, 0=block) for a task."""
        additive_mask = torch.where(
            mask.bool(),
            torch.tensor(0.0, device=mask.device),
            torch.tensor(float("-inf"), device=mask.device),
        )
        seq_len = mask.size(0)
        self.task_masks[task_id, :, :seq_len, :seq_len] = additive_mask.unsqueeze(0)

    def forward(
        self, x: torch.Tensor, task_id: int, return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with task-specific masking.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            task_id: Task identifier
            return_attention: Whether to return attention weights

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()

        Q = self.Q_projections[str(task_id)](x)
        K = self.K_projections[str(task_id)](x)
        V = self.V_projections[str(task_id)](x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply task-specific mask
        mask = self.task_masks[task_id, :, :seq_len, :seq_len]
        scores = scores + mask

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended = torch.matmul(attention_weights, V)

        # Reshape back
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        output = self.out_projections[str(task_id)](attended)

        if return_attention:
            return output, attention_weights
        return output

    def freeze_task(self, task_id: int):
        """Freeze all parameters for a specific task."""
        for param in self.Q_projections[str(task_id)].parameters():
            param.requires_grad = False
        for param in self.K_projections[str(task_id)].parameters():
            param.requires_grad = False
        for param in self.V_projections[str(task_id)].parameters():
            param.requires_grad = False
        for param in self.out_projections[str(task_id)].parameters():
            param.requires_grad = False

    def get_task_parameters(self, task_id: int) -> List[nn.Parameter]:
        """Get all trainable parameters for a specific task."""
        params = []
        params.extend(self.Q_projections[str(task_id)].parameters())
        params.extend(self.K_projections[str(task_id)].parameters())
        params.extend(self.V_projections[str(task_id)].parameters())
        params.extend(self.out_projections[str(task_id)].parameters())
        return [p for p in params if p.requires_grad]


class DTGMALayer(nn.Module):
    """
    Full DTG-MA layer combining attention with feedforward network.

    Architecture:
        x → LayerNorm → TaskGraphAttention → Add & Norm → FFN → Add & Norm → out
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int = 4,
        n_tasks_max: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.attention = TaskGraphAttention(d_model, n_heads, n_tasks_max, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Task-specific feedforward networks
        self.ffn: nn.ModuleDict = nn.ModuleDict()

    def register_task(self, task_id: int, seq_len: int = 1):
        """Register a new task."""
        self.attention.register_task(task_id, seq_len)

        device = self.norm1.weight.device

        self.ffn[str(task_id)] = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(self.dropout.p),
        ).to(device)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Attention block with residual
        normed = self.norm1(x)
        attended = self.attention(normed, task_id)
        x = x + self.dropout(attended)

        # FFN block with residual
        normed = self.norm2(x)
        ffn_out = self.ffn[str(task_id)](normed)
        x = x + ffn_out

        return x

    def freeze_task(self, task_id: int):
        """Freeze all parameters for a specific task."""
        self.attention.freeze_task(task_id)
        for param in self.ffn[str(task_id)].parameters():
            param.requires_grad = False

    def get_task_parameters(self, task_id: int) -> List[nn.Parameter]:
        """Get all trainable parameters for a task."""
        params = self.attention.get_task_parameters(task_id)
        params.extend([p for p in self.ffn[str(task_id)].parameters() if p.requires_grad])
        return params


class DTGMABlock(nn.Module):
    """
    DTG-MA block for standalone use or as adapter on top of frozen encoder.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension (d_model)
        output_dim: Output dimension (num_classes)
        n_layers: Number of DTG-MA layers
        n_heads: Number of attention heads
        n_tasks_max: Maximum number of tasks
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        n_tasks_max: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        # Input projection (shared)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of DTG-MA layers
        self.layers = nn.ModuleList(
            [
                DTGMALayer(hidden_dim, hidden_dim * 4, n_heads, n_tasks_max, dropout)
                for _ in range(n_layers)
            ]
        )

        # Task-specific output heads
        self.output_heads: nn.ModuleDict = nn.ModuleDict()
        self.registered_tasks: List[int] = []

    def register_task(self, task_id: int, seq_len: int = 1):
        """Register a new task."""
        for layer in self.layers:
            layer.register_task(task_id, seq_len)

        device = self.input_proj.weight.device
        self.output_heads[str(task_id)] = nn.Linear(self.hidden_dim, self.output_dim).to(device)
        self.registered_tasks.append(task_id)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (batch, input_dim) or (batch, seq_len, input_dim)
            task_id: Task identifier

        Returns:
            Output (batch, output_dim)
        """
        h = self.input_proj(x)

        # Ensure 3D for attention
        if h.dim() == 2:
            h = h.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        for layer in self.layers:
            h = layer(h, task_id)

        if squeeze_output:
            h = h.squeeze(1)

        output = self.output_heads[str(task_id)](h)
        return output

    def freeze_task(self, task_id: int):
        """Freeze task parameters."""
        for layer in self.layers:
            layer.freeze_task(task_id)
        for param in self.output_heads[str(task_id)].parameters():
            param.requires_grad = False

    def get_task_parameters(self, task_id: int) -> List[nn.Parameter]:
        """Get all trainable parameters for a task."""
        params = []
        for layer in self.layers:
            params.extend(layer.get_task_parameters(task_id))
        params.extend(
            [p for p in self.output_heads[str(task_id)].parameters() if p.requires_grad]
        )
        return params

    def get_total_params(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DTGMAModel(nn.Module):
    """
    Complete DTG-MA model for continual learning (standalone, MLP-style).

    Architecture:
        Input → InputProj → [DTGMALayer]×n_layers → TaskHead → Output

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes per task
        n_layers: Number of DTG-MA layers
        n_heads: Number of attention heads
        n_tasks_max: Maximum number of tasks
        dropout: Dropout probability

    Example:
        >>> model = DTGMAModel(784, 256, 2)
        >>> model.register_task(0)
        >>> output = model(x, task_id=0)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        n_layers: int = 2,
        n_heads: int = 4,
        n_tasks_max: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.block = DTGMABlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            n_layers=n_layers,
            n_heads=n_heads,
            n_tasks_max=n_tasks_max,
            dropout=dropout,
        )

        self.registered_tasks: List[int] = []

    def register_task(self, task_id: int, seq_len: int = 1):
        """Register a new task."""
        self.block.register_task(task_id, seq_len)
        self.registered_tasks.append(task_id)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass."""
        return self.block(x, task_id)

    def freeze_task(self, task_id: int):
        """Freeze task parameters after training."""
        self.block.freeze_task(task_id)

    def get_task_parameters(self, task_id: int) -> List[nn.Parameter]:
        """Get all trainable parameters for a task."""
        return self.block.get_task_parameters(task_id)

    def get_total_params(self) -> int:
        """Total number of parameters."""
        return self.block.get_total_params()

    def get_trainable_params(self) -> int:
        """Number of trainable parameters."""
        return self.block.get_trainable_params()


# ========================================
# Utility functions
# ========================================


def create_task_isolation_mask(
    seq_len: int, task_positions: List[int], allow_self_attention: bool = True
) -> torch.Tensor:
    """
    Create a mask that isolates specific positions for a task.

    Args:
        seq_len: Total sequence length
        task_positions: List of positions belonging to this task
        allow_self_attention: Whether positions can attend to each other

    Returns:
        Binary mask (seq_len, seq_len): 1=allow, 0=block
    """
    mask = torch.zeros(seq_len, seq_len)

    if allow_self_attention:
        for i in task_positions:
            for j in task_positions:
                mask[i, j] = 1.0
    else:
        for i in task_positions:
            mask[i, i] = 1.0

    return mask

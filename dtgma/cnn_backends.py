"""
CNN/ResNet Backends for DTG-MA and Baselines

Provides convolutional feature extractors for fair comparison with literature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


# =============================================================================
# Basic CNN Backbone
# =============================================================================


class SimpleCNN(nn.Module):
    """
    Simple CNN backbone for MNIST/Omniglot (28x28 grayscale images).
    
    Architecture: Conv(32) -> Conv(64) -> FC(128)
    Output: 128-dim features
    """
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.output_dim = 128
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)             # (batch, 64*7*7)
        x = F.relu(self.fc(x))                # (batch, 128)
        return x


class CIFAR_CNN(nn.Module):
    """
    CNN backbone for CIFAR-100 (32x32 RGB images).
    
    Architecture: Conv(64) -> Conv(128) -> Conv(256) -> FC(256)
    Output: 256-dim features
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256 * 4 * 4, 256)
        self.output_dim = 256
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, 32, 32)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (batch, 64, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (batch, 128, 8, 8)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (batch, 256, 4, 4)
        x = x.view(x.size(0), -1)                        # (batch, 256*4*4)
        x = F.relu(self.fc(x))                           # (batch, 256)
        return x


# =============================================================================
# ResNet Backbone
# =============================================================================


class BasicBlock(nn.Module):
    """Basic ResNet block."""
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 backbone for CIFAR (32x32 images).
    
    Modified for smaller input size (no initial 7x7 conv, smaller strides).
    Output: 512-dim features
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = None):
        super().__init__()
        self.in_planes = 64
        
        # Initial conv (modified for CIFAR)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 512
        
        # Optional classifier
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Linear(512, num_classes)
    
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, return_features: bool = True) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        
        if return_features or self.classifier is None:
            return features
        return self.classifier(features)


class ResNet18_MNIST(nn.Module):
    """ResNet-18 adapted for MNIST (28x28 grayscale)."""
    
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 512
    
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)


# =============================================================================
# DTG-MA with CNN Backend
# =============================================================================


class DTGMAWithCNN(nn.Module):
    """
    DTG-MA with CNN/ResNet feature extractor.
    
    Architecture:
        Image → CNN/ResNet → DTG-MA Layers → Task Head → Output
    
    The CNN backbone can be frozen or trained end-to-end.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 256,
        num_classes: int = 10,
        n_layers: int = 2,
        n_heads: int = 4,
        n_tasks_max: int = 20,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        # Import here to avoid circular dependency
        from .model import DTGMABlock
        
        self.dtgma = DTGMABlock(
            input_dim=backbone.output_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            n_layers=n_layers,
            n_heads=n_heads,
            n_tasks_max=n_tasks_max,
        )
        
        self.registered_tasks: List[int] = []
    
    def register_task(self, task_id: int, seq_len: int = 1):
        self.dtgma.register_task(task_id, seq_len)
        self.registered_tasks.append(task_id)
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        # x: (batch, C, H, W) for images
        features = self.backbone(x)
        return self.dtgma(features, task_id)
    
    def freeze_task(self, task_id: int):
        self.dtgma.freeze_task(task_id)
    
    def get_task_parameters(self, task_id: int) -> List[nn.Parameter]:
        params = self.dtgma.get_task_parameters(task_id)
        if not self.freeze_backbone:
            params.extend([p for p in self.backbone.parameters() if p.requires_grad])
        return params
    
    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# CNN Baselines
# =============================================================================


class CNNFineTune(nn.Module):
    """Fine-tuning baseline with CNN backbone."""
    
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.output_dim, num_classes)
        self.name = "Fine-tune+CNN"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class CNNEWC(nn.Module):
    """EWC with CNN backbone."""
    
    def __init__(self, backbone: nn.Module, num_classes: int, ewc_lambda: float = 1000.0):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.output_dim, num_classes)
        self.ewc_lambda = ewc_lambda
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optpar_dict: Dict[str, torch.Tensor] = {}
        self.name = f"EWC+CNN"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def compute_fisher(self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 200):
        self.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        
        n_samples = min(n_samples, len(x))
        
        for i in range(n_samples):
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
    
    def consolidate(self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 200):
        new_fisher = self.compute_fisher(x, y, n_samples)
        for n, p in self.named_parameters():
            if p.requires_grad:
                if n in self.fisher_dict:
                    self.fisher_dict[n] = self.fisher_dict[n] + new_fisher[n]
                else:
                    self.fisher_dict[n] = new_fisher[n].clone()
                self.optpar_dict[n] = p.data.clone()
    
    def ewc_loss(self) -> torch.Tensor:
        if not self.fisher_dict:
            return torch.tensor(0.0)
        
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self.fisher_dict:
                loss += (self.fisher_dict[n] * (p - self.optpar_dict[n]) ** 2).sum()
        return self.ewc_lambda * loss


class CNNHAT(nn.Module):
    """HAT with CNN backbone."""
    
    def __init__(self, backbone: nn.Module, num_classes: int, s_max: float = 400.0):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = backbone.output_dim
        self.fc = nn.Linear(backbone.output_dim, backbone.output_dim)
        self.classifier = nn.Linear(backbone.output_dim, num_classes)
        self.s_max = s_max
        
        self.task_embeddings: Dict[int, nn.Parameter] = {}
        self.masks: Dict[int, torch.Tensor] = {}
        self.current_task = None
        self.name = "HAT+CNN"
    
    def register_task(self, task_id: int):
        if task_id in self.task_embeddings:
            return
        device = next(self.parameters()).device
        self.task_embeddings[task_id] = nn.Parameter(
            torch.randn(self.hidden_dim, device=device) * 0.1
        )
        self.register_parameter(f'task_{task_id}_e', self.task_embeddings[task_id])
        self.current_task = task_id
    
    def get_attention(self, task_id: int, s: float = 1.0) -> torch.Tensor:
        return torch.sigmoid(s * self.task_embeddings[task_id])
    
    def forward(self, x: torch.Tensor, task_id: int = None, s: float = 1.0) -> torch.Tensor:
        if task_id is None:
            task_id = self.current_task
        
        features = self.backbone(x)
        att = self.get_attention(task_id, s)
        h = F.relu(self.fc(features)) * att
        return self.classifier(h)
    
    def freeze_task(self, task_id: int):
        with torch.no_grad():
            att = self.get_attention(task_id, self.s_max)
            self.masks[task_id] = (att > 0.5).float()
    
    def hat_reg_loss(self, task_id: int, s: float) -> torch.Tensor:
        att = self.get_attention(task_id, s)
        reg = att.sum()
        
        overlap = 0.0
        for prev_id, prev_mask in self.masks.items():
            if prev_id != task_id:
                overlap += (att * prev_mask).sum()
        
        return 0.01 * reg + 10.0 * overlap


class CNNPackNet(nn.Module):
    """PackNet with CNN backbone."""
    
    def __init__(self, backbone: nn.Module, num_classes: int, prune_ratio: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.output_dim, backbone.output_dim)
        self.classifier = nn.Linear(backbone.output_dim, num_classes)
        self.prune_ratio = prune_ratio
        
        self.register_buffer('mask_fc', torch.ones_like(self.fc.weight))
        self.register_buffer('frozen_mask_fc', torch.zeros_like(self.fc.weight))
        self.name = "PackNet+CNN"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        w = self.fc.weight * self.mask_fc
        h = F.relu(F.linear(features, w, self.fc.bias))
        return self.classifier(h)
    
    def prune_and_freeze(self):
        available = self.mask_fc * (1 - self.frozen_mask_fc)
        if available.sum() == 0:
            return
        
        magnitudes = (self.fc.weight.abs() * available).flatten()
        nonzero = magnitudes[magnitudes > 0]
        
        if len(nonzero) == 0:
            return
        
        k = int(len(nonzero) * self.prune_ratio)
        if k > 0:
            threshold = torch.kthvalue(nonzero.detach().cpu(), k).values.to(self.fc.weight.device)
            prune_mask = (self.fc.weight.abs() <= threshold) & (available > 0)
            self.mask_fc[prune_mask] = 0
        
        keep_mask = (self.mask_fc > 0) & (available > 0)
        self.frozen_mask_fc[keep_mask] = 1


class CNNDERPP(nn.Module):
    """DER++ with CNN backbone."""
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        buffer_size: int = 500,
        alpha: float = 0.5,
        beta: float = 0.5,
        input_shape: Tuple[int, ...] = (3, 32, 32),
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.output_dim, num_classes)
        self.alpha = alpha
        self.beta = beta
        self.name = "DER+++CNN"
        
        # Buffer stores images
        self.buffer_size = buffer_size
        self.input_shape = input_shape
        self.register_buffer('buffer_x', torch.zeros(buffer_size, *input_shape))
        self.register_buffer('buffer_y', torch.zeros(buffer_size, dtype=torch.long))
        self.register_buffer('buffer_logits', torch.zeros(buffer_size, num_classes))
        self.buffer_count = 0
        self.buffer_ptr = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def add_to_buffer(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor):
        batch_size = x.size(0)
        for i in range(batch_size):
            self.buffer_x[self.buffer_ptr] = x[i].detach()
            self.buffer_y[self.buffer_ptr] = y[i].detach()
            self.buffer_logits[self.buffer_ptr] = logits[i].detach()
            self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size
            self.buffer_count = min(self.buffer_count + 1, self.buffer_size)
    
    def sample_buffer(self, batch_size: int):
        idx = torch.randperm(self.buffer_count)[:min(batch_size, self.buffer_count)]
        return self.buffer_x[idx], self.buffer_y[idx], self.buffer_logits[idx]


# =============================================================================
# Factory Functions
# =============================================================================


def get_backbone(dataset: str, use_resnet: bool = False) -> nn.Module:
    """Get appropriate backbone for dataset."""
    if dataset in ['mnist', 'omniglot']:
        if use_resnet:
            return ResNet18_MNIST()
        return SimpleCNN(in_channels=1)
    elif dataset in ['cifar100', 'cifar10']:
        if use_resnet:
            return ResNet18(in_channels=3)
        return CIFAR_CNN()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

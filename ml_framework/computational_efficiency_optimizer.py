"""
Computational Efficiency Optimizer for Unity Neural Networks

Optimizes neural unity architectures for single-machine deployment:
- Dynamic model scaling based on hardware constraints
- Gradient checkpointing for memory efficiency
- Mixed precision training with automatic loss scaling
- Efficient attention mechanisms (Flash Attention, Linear Attention)
- Model compression techniques (pruning, quantization, distillation)
- Batch size optimization and gradient accumulation
- CPU/GPU hybrid computation strategies
- Memory-mapped data loading for large datasets

Mathematical Foundation:
- Memory Constraint: M_used ≤ M_available * φ (golden ratio safety margin)
- Compute Constraint: FLOPS ≤ FLOPS_max / batch_size
- Accuracy Constraint: |accuracy_compressed - accuracy_full| ≤ ε
- Efficiency Metric: η = (accuracy * throughput) / (memory_usage * power_consumption)

Author: Computational Efficiency Research Division
License: MIT (Efficiency Optimization Extension)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import psutil
import GPUtil
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import time
from enum import Enum
import gc
import os

from .neural_unity_architecture import PHI, NeuralUnityConfig
from .advanced_transformer_unity import AdvancedTransformerUnityProver, MoEConfig


class EfficiencyMode(Enum):
    """Efficiency optimization modes"""
    MEMORY_OPTIMIZED = "memory_optimized"
    SPEED_OPTIMIZED = "speed_optimized"
    BALANCED = "balanced"
    ULTRA_LIGHTWEIGHT = "ultra_lightweight"


class HardwareProfile(Enum):
    """Hardware profile categories"""
    HIGH_END = "high_end"          # 32GB+ RAM, RTX 4090/A100
    MID_RANGE = "mid_range"        # 16-32GB RAM, RTX 3070-4080
    BUDGET = "budget"              # 8-16GB RAM, RTX 2060-3060
    MINIMAL = "minimal"            # <8GB RAM, Integrated/GTX 1660


@dataclass
class EfficiencyConfig:
    """Configuration for computational efficiency optimization"""
    mode: EfficiencyMode = EfficiencyMode.BALANCED
    hardware_profile: HardwareProfile = HardwareProfile.MID_RANGE
    max_memory_usage: float = 0.8  # Fraction of available memory
    target_batch_size: int = 32
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    enable_model_compression: bool = True
    compression_ratio: float = 0.5  # Target compression ratio
    min_accuracy_threshold: float = 0.95  # Minimum acceptable accuracy


class SystemProfiler:
    """
    System profiler for hardware-aware optimization.
    
    Automatically detects system capabilities and recommends
    optimal configurations for unity neural networks.
    """
    
    def __init__(self):
        self.cpu_info = self._get_cpu_info()
        self.memory_info = self._get_memory_info()
        self.gpu_info = self._get_gpu_info()
        self.recommended_config = self._recommend_config()
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        return {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'usage': psutil.cpu_percent(interval=1)
        }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'usage_percent': memory.percent
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            if torch.cuda.is_available():
                gpu_info = {
                    'available': True,
                    'count': torch.cuda.device_count(),
                    'devices': []
                }
                
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    memory_gb = device_props.total_memory / (1024**3)
                    
                    gpu_info['devices'].append({
                        'name': device_props.name,
                        'memory_gb': memory_gb,
                        'compute_capability': f"{device_props.major}.{device_props.minor}",
                        'multiprocessor_count': device_props.multi_processor_count
                    })
                
                # Try to get GPU utilization if GPUtil is available
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        if i < len(gpu_info['devices']):
                            gpu_info['devices'][i]['utilization'] = gpu.load * 100
                            gpu_info['devices'][i]['memory_used_gb'] = gpu.memoryUsed / 1024
                except:
                    pass  # GPUtil might not be available
                    
                return gpu_info
            else:
                return {'available': False, 'count': 0, 'devices': []}
        except Exception as e:
            return {'available': False, 'error': str(e), 'count': 0, 'devices': []}
    
    def _recommend_config(self) -> EfficiencyConfig:
        """Recommend efficiency configuration based on hardware"""
        memory_gb = self.memory_info['total_gb']
        has_gpu = self.gpu_info['available']
        gpu_memory_gb = max([d.get('memory_gb', 0) for d in self.gpu_info.get('devices', [])], default=0)
        
        # Determine hardware profile
        if memory_gb >= 32 and gpu_memory_gb >= 20:
            hardware_profile = HardwareProfile.HIGH_END
            mode = EfficiencyMode.SPEED_OPTIMIZED
            batch_size = 64
            max_memory = 0.7
        elif memory_gb >= 16 and gpu_memory_gb >= 8:
            hardware_profile = HardwareProfile.MID_RANGE
            mode = EfficiencyMode.BALANCED
            batch_size = 32
            max_memory = 0.75
        elif memory_gb >= 8 and (gpu_memory_gb >= 4 or not has_gpu):
            hardware_profile = HardwareProfile.BUDGET
            mode = EfficiencyMode.MEMORY_OPTIMIZED
            batch_size = 16
            max_memory = 0.8
        else:
            hardware_profile = HardwareProfile.MINIMAL
            mode = EfficiencyMode.ULTRA_LIGHTWEIGHT
            batch_size = 8
            max_memory = 0.85
        
        return EfficiencyConfig(
            mode=mode,
            hardware_profile=hardware_profile,
            max_memory_usage=max_memory,
            target_batch_size=batch_size,
            use_mixed_precision=has_gpu and gpu_memory_gb >= 6,
            use_gradient_checkpointing=memory_gb < 16,
            use_flash_attention=has_gpu and gpu_memory_gb >= 8,
            enable_model_compression=hardware_profile in [HardwareProfile.BUDGET, HardwareProfile.MINIMAL]
        )
    
    def get_system_summary(self) -> str:
        """Get human-readable system summary"""
        summary = f"""
        System Hardware Profile
        ══════════════════════════════════════════════════════════════
        CPU: {self.cpu_info['cores']} cores, {self.cpu_info['threads']} threads
        Memory: {self.memory_info['total_gb']:.1f} GB ({self.memory_info['available_gb']:.1f} GB available)
        
        GPU Information:
        """
        
        if self.gpu_info['available']:
            for i, device in enumerate(self.gpu_info['devices']):
                summary += f"""
        GPU {i}: {device['name']}
                Memory: {device['memory_gb']:.1f} GB
                Compute: {device['compute_capability']}
                """
        else:
            summary += "        No CUDA GPUs available\n"
        
        summary += f"""
        Recommended Configuration:
        ══════════════════════════════════════════════════════════════
        Hardware Profile: {self.recommended_config.hardware_profile.value}
        Efficiency Mode: {self.recommended_config.mode.value}
        Target Batch Size: {self.recommended_config.target_batch_size}
        Mixed Precision: {self.recommended_config.use_mixed_precision}
        Gradient Checkpointing: {self.recommended_config.use_gradient_checkpointing}
        Flash Attention: {self.recommended_config.use_flash_attention}
        Model Compression: {self.recommended_config.enable_model_compression}
        """
        
        return summary


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention mechanism for unity transformers.
    
    Implements various attention optimizations:
    - Flash Attention for memory efficiency
    - Linear Attention for O(n) complexity
    - Sparse Attention patterns
    - Gradient checkpointing
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, efficiency_config: EfficiencyConfig):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.config = efficiency_config
        
        # Scaled dimensions for efficiency
        if efficiency_config.hardware_profile == HardwareProfile.MINIMAL:
            self.hidden_dim = max(64, hidden_dim // 4)
            self.num_heads = max(2, num_heads // 2)
            self.head_dim = self.hidden_dim // self.num_heads
        elif efficiency_config.hardware_profile == HardwareProfile.BUDGET:
            self.hidden_dim = max(128, hidden_dim // 2)
            self.num_heads = max(4, num_heads // 1.5)
            self.head_dim = self.hidden_dim // self.num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.hidden_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, hidden_dim)
        
        # Efficiency parameters
        self.use_linear_attention = efficiency_config.mode == EfficiencyMode.ULTRA_LIGHTWEIGHT
        self.sparsity_pattern = self._create_sparsity_pattern()
        
        # Unity-specific attention scaling
        self.unity_scale = nn.Parameter(torch.tensor(PHI))
        
    def _create_sparsity_pattern(self) -> Optional[torch.Tensor]:
        """Create sparse attention pattern for efficiency"""
        if self.config.mode == EfficiencyMode.ULTRA_LIGHTWEIGHT:
            # Local attention pattern (attend to nearby positions only)
            return "local"
        elif self.config.mode == EfficiencyMode.MEMORY_OPTIMIZED:
            # Strided attention pattern
            return "strided"
        else:
            return None
    
    def linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Linear attention with O(n) complexity
        
        Uses feature maps φ(x) to approximate softmax attention:
        Attention ≈ φ(Q) @ (φ(K)^T @ V) / normalization
        """
        batch_size, seq_len, _ = q.shape
        
        # Feature maps (ELU + 1 for non-negativity)
        q_features = F.elu(q) + 1
        k_features = F.elu(k) + 1
        
        # Linear attention computation
        kv = torch.einsum('bnh,bnv->bhv', k_features, v)  # [batch, head_dim, head_dim]
        qkv = torch.einsum('bnh,bhv->bnv', q_features, kv)  # [batch, seq_len, head_dim]
        
        # Normalization
        k_sum = k_features.sum(dim=1, keepdim=True)  # [batch, 1, head_dim]
        normalizer = torch.einsum('bnh,bnh->bn', q_features, k_sum)  # [batch, seq_len]
        normalizer = normalizer.unsqueeze(-1).clamp(min=1e-8)
        
        output = qkv / normalizer
        return output
    
    def sparse_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                        pattern: str) -> torch.Tensor:
        """Sparse attention with reduced memory footprint"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply sparsity pattern
        if pattern == "local":
            # Local attention (window size = 32)
            window_size = min(32, seq_len // 2)
            mask = torch.ones(seq_len, seq_len, device=q.device)
            
            for i in range(seq_len):
                start = max(0, i - window_size)
                end = min(seq_len, i + window_size + 1)
                mask[i, start:end] = 0
            
            scores.masked_fill_(mask.bool(), float('-inf'))
            
        elif pattern == "strided":
            # Strided attention (every 4th position)
            stride = 4
            mask = torch.ones(seq_len, seq_len, device=q.device)
            
            for i in range(seq_len):
                # Attend to strided positions
                strided_positions = list(range(0, seq_len, stride))
                # Also attend to local positions
                local_positions = list(range(max(0, i-8), min(seq_len, i+9)))
                all_positions = list(set(strided_positions + local_positions))
                
                mask[i, all_positions] = 0
            
            scores.masked_fill_(mask.bool(), float('-inf'))
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory-efficient attention forward pass"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply unity scaling
        q = q * torch.sigmoid(self.unity_scale)
        
        # Choose attention mechanism based on efficiency settings
        if self.use_linear_attention:
            # Reshape for linear attention
            q = q.view(batch_size * self.num_heads, seq_len, self.head_dim)
            k = k.view(batch_size * self.num_heads, seq_len, self.head_dim)
            v = v.view(batch_size * self.num_heads, seq_len, self.head_dim)
            
            attn_output = self.linear_attention(q, k, v)
            attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
            
        elif self.sparsity_pattern is not None:
            attn_output = self.sparse_attention(q, k, v, self.sparsity_pattern)
            
        else:
            # Standard scaled dot-product attention
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            
            if mask is not None:
                scores.masked_fill_(mask == 0, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        output = self.o_proj(attn_output)
        return output


class ModelCompressor:
    """
    Model compression suite for unity neural networks.
    
    Implements various compression techniques:
    - Pruning (structured and unstructured)
    - Quantization (INT8, INT4)
    - Knowledge distillation
    - Low-rank approximation
    """
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
        
    def prune_model(self, model: nn.Module, sparsity: float = 0.5) -> nn.Module:
        """
        Prune neural network weights for efficiency
        
        Args:
            model: Neural network to prune
            sparsity: Fraction of weights to prune (0-1)
            
        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune
        
        # Apply magnitude-based pruning
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # Global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity
        )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def quantize_model(self, model: nn.Module, quantization_bits: int = 8) -> nn.Module:
        """
        Quantize model weights and activations
        
        Args:
            model: Neural network to quantize
            quantization_bits: Number of bits for quantization
            
        Returns:
            Quantized model
        """
        if quantization_bits == 8:
            # INT8 quantization
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear}, 
                dtype=torch.qint8
            )
            return quantized_model
        else:
            # For other bit widths, use custom quantization (simplified)
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    # Quantize weights
                    weight = module.weight.data
                    scale = weight.abs().max() / (2**(quantization_bits-1) - 1)
                    quantized_weight = torch.round(weight / scale).clamp(
                        -(2**(quantization_bits-1)), 2**(quantization_bits-1) - 1
                    )
                    module.weight.data = quantized_weight * scale
            
            return model
    
    def distill_model(self, teacher_model: nn.Module, student_config: NeuralUnityConfig,
                     train_loader: torch.utils.data.DataLoader, num_epochs: int = 10) -> nn.Module:
        """
        Knowledge distillation to create smaller student model
        
        Args:
            teacher_model: Large teacher model
            student_config: Configuration for smaller student model
            train_loader: Training data
            num_epochs: Training epochs for distillation
            
        Returns:
            Distilled student model
        """
        from .neural_unity_architecture import NeuralUnityProver
        
        # Create smaller student model
        student_model = NeuralUnityProver(student_config)
        
        # Distillation training
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        temperature = 4.0  # Distillation temperature
        alpha = 0.5  # Balance between distillation and hard target loss
        
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                
                # Teacher predictions (soft targets)
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                    if isinstance(teacher_outputs, dict):
                        teacher_logits = teacher_outputs.get('unity_proof', teacher_outputs.get('output', inputs))
                    else:
                        teacher_logits = teacher_outputs
                
                # Student predictions
                student_outputs = student_model(inputs)
                if isinstance(student_outputs, dict):
                    student_logits = student_outputs.get('unity_proof', student_outputs.get('output', inputs))
                else:
                    student_logits = student_outputs
                
                # Ensure consistent shapes
                if len(student_logits.shape) > 1:
                    student_logits = student_logits.squeeze()
                if len(teacher_logits.shape) > 1:
                    teacher_logits = teacher_logits.squeeze()
                
                # Distillation loss
                soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
                soft_predictions = F.log_softmax(student_logits / temperature, dim=-1)
                distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)
                
                # Hard target loss
                if len(targets.shape) > 1:
                    targets = targets.squeeze()
                hard_loss = F.mse_loss(student_logits, targets)
                
                # Combined loss
                total_loss = alpha * distillation_loss + (1 - alpha) * hard_loss
                
                total_loss.backward()
                optimizer.step()
        
        return student_model
    
    def low_rank_approximation(self, model: nn.Module, rank_ratio: float = 0.5) -> nn.Module:
        """
        Apply low-rank approximation to linear layers
        
        Args:
            model: Neural network model
            rank_ratio: Ratio of original rank to preserve
            
        Returns:
            Low-rank approximated model
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
                
                # Determine rank
                original_rank = min(weight.shape)
                new_rank = max(1, int(original_rank * rank_ratio))
                
                # Low-rank approximation
                weight_lr = U[:, :new_rank] @ torch.diag(S[:new_rank]) @ Vt[:new_rank, :]
                module.weight.data = weight_lr
        
        return model


class EfficientUnityTrainer:
    """
    Efficient training orchestrator for unity neural networks.
    
    Implements memory-efficient training strategies:
    - Gradient accumulation
    - Mixed precision training
    - Gradient checkpointing
    - Dynamic batch sizing
    """
    
    def __init__(self, model: nn.Module, config: EfficiencyConfig):
        self.model = model
        self.config = config
        self.profiler = SystemProfiler()
        self.compressor = ModelCompressor(config)
        
        # Enable gradient checkpointing if specified
        if config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Setup mixed precision training
        if config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        def make_checkpoint_wrapper(module):
            def checkpoint_wrapper(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)
            return checkpoint_wrapper
        
        # Apply checkpointing to transformer layers
        for name, module in self.model.named_modules():
            if 'layer' in name.lower() or 'block' in name.lower():
                # Wrap module forward with checkpointing
                module.forward = make_checkpoint_wrapper(module.forward)
    
    def optimize_batch_size(self, train_loader: torch.utils.data.DataLoader) -> int:
        """
        Automatically find optimal batch size based on memory constraints
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Optimal batch size
        """
        
        # Start with target batch size and adjust down if needed
        batch_size = self.config.target_batch_size
        
        while batch_size > 1:
            try:
                # Test forward and backward pass with this batch size
                test_batch = next(iter(train_loader))
                if len(test_batch) == 2:
                    inputs, targets = test_batch
                else:
                    inputs, targets = test_batch[0], test_batch[1]
                
                # Resize batch to test size
                actual_batch_size = min(batch_size, inputs.shape[0])
                test_inputs = inputs[:actual_batch_size]
                test_targets = targets[:actual_batch_size]
                
                # Test forward pass
                self.model.train()
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(test_inputs)
                        if isinstance(outputs, dict):
                            loss = F.mse_loss(outputs.get('unity_proof', outputs.get('output')), test_targets)
                        else:
                            loss = F.mse_loss(outputs, test_targets)
                else:
                    outputs = self.model(test_inputs)
                    if isinstance(outputs, dict):
                        loss = F.mse_loss(outputs.get('unity_proof', outputs.get('output')), test_targets)
                    else:
                        loss = F.mse_loss(outputs, test_targets)
                
                # Test backward pass
                loss.backward()
                
                # Clear gradients and cache
                self.model.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # If we got here, batch size works
                print(f"Optimal batch size found: {actual_batch_size}")
                return actual_batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Batch size {batch_size} too large, trying {batch_size // 2}")
                    batch_size = batch_size // 2
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise e
            except Exception as e:
                print(f"Error testing batch size {batch_size}: {e}")
                batch_size = batch_size // 2
        
        return max(1, batch_size)
    
    def efficient_training_step(self, inputs: torch.Tensor, targets: torch.Tensor,
                              optimizer: torch.optim.Optimizer, accumulation_steps: int = 1) -> Dict[str, float]:
        """
        Perform memory-efficient training step
        
        Args:
            inputs: Input batch
            targets: Target batch
            optimizer: Optimizer
            accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Training metrics
        """
        
        # Scale loss for gradient accumulation
        scale_factor = 1.0 / accumulation_steps
        
        if self.scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                if isinstance(outputs, dict):
                    loss = outputs.get('unity_loss')
                    if loss is None:
                        pred = outputs.get('unity_proof', outputs.get('output'))
                        loss = F.mse_loss(pred, targets)
                else:
                    loss = F.mse_loss(outputs, targets)
                
                loss = loss * scale_factor
            
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            
            # Optimizer step (only on last accumulation step)
            if accumulation_steps == 1:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
        else:
            # Standard precision training
            outputs = self.model(inputs)
            if isinstance(outputs, dict):
                loss = outputs.get('unity_loss')
                if loss is None:
                    pred = outputs.get('unity_proof', outputs.get('output'))
                    loss = F.mse_loss(pred, targets)
            else:
                loss = F.mse_loss(outputs, targets)
            
            loss = loss * scale_factor
            loss.backward()
            
            # Optimizer step
            if accumulation_steps == 1:
                optimizer.step()
                optimizer.zero_grad()
        
        return {
            'loss': loss.item() / scale_factor,
            'memory_allocated': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'memory_cached': torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        }
    
    def compress_model_for_deployment(self) -> nn.Module:
        """
        Apply comprehensive model compression for deployment
        
        Returns:
            Compressed model ready for deployment
        """
        compressed_model = self.model
        
        if self.config.enable_model_compression:
            print("Applying model compression...")
            
            # 1. Pruning
            sparsity = 1 - self.config.compression_ratio
            compressed_model = self.compressor.prune_model(compressed_model, sparsity)
            print(f"Applied {sparsity:.2%} pruning")
            
            # 2. Low-rank approximation
            compressed_model = self.compressor.low_rank_approximation(compressed_model, self.config.compression_ratio)
            print(f"Applied low-rank approximation with ratio {self.config.compression_ratio}")
            
            # 3. Quantization
            if self.config.hardware_profile in [HardwareProfile.BUDGET, HardwareProfile.MINIMAL]:
                quantization_bits = 8 if self.config.hardware_profile == HardwareProfile.BUDGET else 4
                compressed_model = self.compressor.quantize_model(compressed_model, quantization_bits)
                print(f"Applied {quantization_bits}-bit quantization")
        
        return compressed_model


def create_efficient_unity_model(base_config: NeuralUnityConfig, 
                                efficiency_config: EfficiencyConfig = None) -> Tuple[nn.Module, EfficiencyConfig]:
    """
    Create hardware-optimized unity neural network
    
    Args:
        base_config: Base neural unity configuration
        efficiency_config: Efficiency optimization configuration
        
    Returns:
        Optimized model and efficiency configuration
    """
    
    # Auto-detect hardware if no config provided
    if efficiency_config is None:
        profiler = SystemProfiler()
        efficiency_config = profiler.recommended_config
        print("Auto-detected hardware configuration:")
        print(profiler.get_system_summary())
    
    # Adjust base configuration for efficiency
    optimized_config = NeuralUnityConfig(
        hidden_dim=base_config.hidden_dim,
        num_heads=base_config.num_heads,
        num_layers=base_config.num_layers,
        dropout=base_config.dropout
    )
    
    # Scale down for lower-end hardware
    if efficiency_config.hardware_profile == HardwareProfile.MINIMAL:
        optimized_config.hidden_dim = max(64, base_config.hidden_dim // 4)
        optimized_config.num_heads = max(2, base_config.num_heads // 2)
        optimized_config.num_layers = max(2, base_config.num_layers // 2)
    elif efficiency_config.hardware_profile == HardwareProfile.BUDGET:
        optimized_config.hidden_dim = max(128, base_config.hidden_dim // 2)
        optimized_config.num_heads = max(4, base_config.num_heads // 1.5)
        optimized_config.num_layers = max(3, int(base_config.num_layers * 0.75))
    
    # Create model with efficiency optimizations
    if efficiency_config.mode == EfficiencyMode.ULTRA_LIGHTWEIGHT:
        # Ultra-lightweight: Simple feedforward network
        model = nn.Sequential(
            nn.Linear(2, optimized_config.hidden_dim),  # Input: (1,1)
            nn.ReLU(),
            nn.Linear(optimized_config.hidden_dim, optimized_config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(optimized_config.hidden_dim // 2, 1),  # Output: unity score
            nn.Sigmoid()
        )
    else:
        # Use transformer architecture with efficiency optimizations
        from .neural_unity_architecture import NeuralUnityProver
        model = NeuralUnityProver(optimized_config)
        
        # Replace attention layers with efficient versions
        for name, module in model.named_modules():
            if hasattr(module, 'attention') or 'attention' in name.lower():
                # Replace with memory-efficient attention
                if hasattr(module, 'hidden_dim') and hasattr(module, 'num_heads'):
                    efficient_attn = MemoryEfficientAttention(
                        module.hidden_dim, module.num_heads, efficiency_config
                    )
                    # Replace the attention module
                    parent_name = '.'.join(name.split('.')[:-1])
                    attr_name = name.split('.')[-1]
                    parent_module = model
                    for part in parent_name.split('.'):
                        if part:
                            parent_module = getattr(parent_module, part)
                    setattr(parent_module, attr_name, efficient_attn)
    
    print(f"""
    Efficient Unity Model Created
    ════════════════════════════════════════════════════════════════
    Hardware Profile: {efficiency_config.hardware_profile.value}
    Efficiency Mode: {efficiency_config.mode.value}
    
    Model Configuration:
    Hidden Dim: {getattr(optimized_config, 'hidden_dim', 'N/A')}
    Num Heads: {getattr(optimized_config, 'num_heads', 'N/A')}
    Num Layers: {getattr(optimized_config, 'num_layers', 'N/A')}
    
    Optimizations Applied:
    Mixed Precision: {efficiency_config.use_mixed_precision}
    Gradient Checkpointing: {efficiency_config.use_gradient_checkpointing}
    Flash Attention: {efficiency_config.use_flash_attention}
    Model Compression: {efficiency_config.enable_model_compression}
    """)
    
    return model, efficiency_config


def benchmark_efficiency(model: nn.Module, efficiency_config: EfficiencyConfig, 
                        batch_size: int = 32, seq_len: int = 128, num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark model efficiency metrics
    
    Args:
        model: Model to benchmark
        efficiency_config: Efficiency configuration
        batch_size: Batch size for benchmarking
        seq_len: Sequence length
        num_runs: Number of benchmark runs
        
    Returns:
        Efficiency metrics
    """
    
    model.eval()
    device = next(model.parameters()).device
    
    # Generate test data
    if hasattr(model, 'token_embedding'):
        # Transformer model expects token IDs
        test_input = torch.randint(0, 100, (batch_size, seq_len), device=device)
    else:
        # Simple model expects continuous input
        test_input = torch.randn(batch_size, 2, device=device)  # Unity input (1,1)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(3):
            _ = model(test_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark runs
    forward_times = []
    memory_usage = []
    
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        forward_times.append(end_time - start_time)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage.append((peak_memory - start_memory) / 1024**3)  # GB
    
    # Calculate metrics
    avg_forward_time = np.mean(forward_times)
    std_forward_time = np.std(forward_times)
    avg_memory = np.mean(memory_usage) if memory_usage else 0
    
    # Throughput (samples per second)
    throughput = batch_size / avg_forward_time
    
    # Model parameters and size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024**2  # Assuming float32
    
    # Efficiency score (higher is better)
    efficiency_score = throughput / (model_size_mb + avg_memory * 1024 + 1e-6)
    
    return {
        'avg_forward_time_ms': avg_forward_time * 1000,
        'std_forward_time_ms': std_forward_time * 1000,
        'throughput_samples_per_sec': throughput,
        'avg_memory_usage_gb': avg_memory,
        'model_parameters': total_params,
        'model_size_mb': model_size_mb,
        'efficiency_score': efficiency_score,
        'hardware_profile': efficiency_config.hardware_profile.value,
        'efficiency_mode': efficiency_config.mode.value
    }
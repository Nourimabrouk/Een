
"""GPU Utilities for Unity Mathematics"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch available. Using device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("PyTorch not available. Using NumPy backend.")

def to_device(tensor):
    """Move tensor to appropriate device"""
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        return tensor.to(DEVICE)
    return tensor

def ensure_numpy(tensor):
    """Convert tensor to numpy array"""
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return np.asarray(tensor)

def create_tensor(data):
    """Create tensor on appropriate device"""
    if TORCH_AVAILABLE:
        return torch.tensor(data, device=DEVICE, dtype=torch.float32)
    return np.array(data, dtype=np.float32)

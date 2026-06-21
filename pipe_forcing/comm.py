"""Small tensor communication helpers for pipe-forcing inference."""

from __future__ import annotations

import torch
import torch.distributed as dist


def send_tensor(tensor: torch.Tensor, dst: int) -> None:
    """Send a contiguous CUDA tensor to the next pipeline rank."""
    dist.send(tensor.contiguous(), dst=dst)


def recv_tensor(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, src: int) -> torch.Tensor:
    """Receive a CUDA tensor with a known shape from the previous pipeline rank."""
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src)
    return tensor


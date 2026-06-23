from __future__ import annotations

import torch
import torch.distributed as dist


def seq_to_head_parallel(x: torch.Tensor, group=None) -> torch.Tensor:
    """[N, H, Dh] -> [world*N, H/world, Dh]."""
    # minWM/Wan SP 风格：scatter heads，gather sequence。
    # all-to-all 后，每张卡拿到全 sequence，但只负责一部分 heads。
    world_size = dist.get_world_size(group)
    n_tokens, n_heads, head_dim = x.shape
    if n_heads % world_size != 0:
        raise ValueError(f"num_heads={n_heads} must be divisible by world_size={world_size}")
    local_heads = n_heads // world_size
    send = x.reshape(n_tokens, world_size, local_heads, head_dim).permute(1, 0, 2, 3).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.reshape(world_size * n_tokens, local_heads, head_dim).contiguous()


def head_parallel_to_seq(x: torch.Tensor, group=None) -> torch.Tensor:
    """[world*N, H/world, Dh] -> [N, H, Dh]."""
    # attention 完成后做反向 all-to-all：scatter sequence，gather heads 回本地 chunk。
    world_size = dist.get_world_size(group)
    global_tokens, local_heads, head_dim = x.shape
    if global_tokens % world_size != 0:
        raise ValueError(f"global_tokens={global_tokens} must be divisible by world_size={world_size}")
    n_tokens = global_tokens // world_size
    send = x.reshape(world_size, n_tokens, local_heads, head_dim).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.permute(1, 0, 2, 3).reshape(n_tokens, world_size * local_heads, head_dim).contiguous()

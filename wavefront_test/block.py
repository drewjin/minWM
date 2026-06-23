from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight


class DummyWanBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 对齐 Wan 的主耗时结构：LN -> q/k/v -> q/k RMSNorm -> attention -> o -> MLP。
        # 省掉 AdaLN、cross-attn、RoPE/PRoPE，只保留这次 benchmark 关心的算子。
        self.norm1 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.q = nn.Linear(d_model, d_model, bias=True)
        self.k = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, d_model, bias=True)
        self.norm_q = WanRMSNorm(d_model, eps=eps)
        self.norm_k = WanRMSNorm(d_model, eps=eps)
        self.o = nn.Linear(d_model, d_model, bias=True)

        self.norm2 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)

    def compute_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_tokens = x.shape[0]
        h = self.norm1(x)
        q = self.norm_q(self.q(h)).view(n_tokens, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(h)).view(n_tokens, self.num_heads, self.head_dim)
        v = self.v(h).view(n_tokens, self.num_heads, self.head_dim)
        return q.contiguous(), k.contiguous(), v.contiguous()

    def attention_heads(
        self,
        q: torch.Tensor,
        k_list: list[torch.Tensor],
        v_list: list[torch.Tensor],
    ) -> torch.Tensor:
        # 内部 full attention；frontier 的因果性由调用方选择 K/V 来源实现。
        # SDPA 需要 [B, H, N, Dh]，本 benchmark 内部主要用 [N, H, Dh]。
        k = torch.cat(k_list, dim=0)
        v = torch.cat(v_list, dim=0)
        q_sdpa = q.permute(1, 0, 2).unsqueeze(0)
        k_sdpa = k.permute(1, 0, 2).unsqueeze(0)
        v_sdpa = v.permute(1, 0, 2).unsqueeze(0)
        out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False)
        return out.squeeze(0).permute(1, 0, 2).contiguous()

    def project_attention(self, out_heads: torch.Tensor) -> torch.Tensor:
        return self.o(out_heads.reshape(out_heads.shape[0], self.d_model))

    def self_attn(
        self,
        q: torch.Tensor,
        k_list: list[torch.Tensor],
        v_list: list[torch.Tensor],
    ) -> torch.Tensor:
        return self.project_attention(self.attention_heads(q, k_list, v_list))

    def finish_block(self, x: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        x = x + attn_out
        h = self.norm2(x)
        h = self.fc2(F.gelu(self.fc1(h), approximate="tanh"))
        return x + h


def build_blocks(
    layers: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.ModuleList:
    blocks = nn.ModuleList([DummyWanBlock(d_model, num_heads, ffn_dim, eps=eps) for _ in range(layers)])
    return blocks.to(device=device, dtype=dtype).eval()

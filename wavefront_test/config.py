from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HF_WAN13B_CONFIG_URL = "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/raw/main/config.json"


@dataclass(frozen=True)
class WanArchConfig:
    d_model: int
    ffn_dim: int
    num_heads: int
    num_layers: int
    in_dim: int = 16
    out_dim: int = 16
    text_len: int = 512
    eps: float = 1e-6
    patch_size: tuple[int, int, int] = (1, 2, 2)
    source: str = "builtin"

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}")
        return self.d_model // self.num_heads

    @classmethod
    def wan21_t2v_13b(cls, source: str = "builtin") -> "WanArchConfig":
        return cls(
            d_model=1536,
            ffn_dim=8960,
            num_heads=12,
            num_layers=30,
            in_dim=16,
            out_dim=16,
            text_len=512,
            eps=1e-6,
            patch_size=(1, 2, 2),
            source=source,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str) -> "WanArchConfig":
        fallback = cls.wan21_t2v_13b(source=source)
        patch_size = data.get("patch_size", fallback.patch_size)
        if patch_size is None:
            patch_size = fallback.patch_size
        return cls(
            d_model=int(data.get("dim", data.get("d_model", fallback.d_model))),
            ffn_dim=int(data.get("ffn_dim", fallback.ffn_dim)),
            num_heads=int(data.get("num_heads", fallback.num_heads)),
            num_layers=int(data.get("num_layers", fallback.num_layers)),
            in_dim=int(data.get("in_dim", fallback.in_dim)),
            out_dim=int(data.get("out_dim", fallback.out_dim)),
            text_len=int(data.get("text_len", fallback.text_len)),
            eps=float(data.get("eps", fallback.eps)),
            patch_size=tuple(int(x) for x in patch_size),
            source=source,
        )


@dataclass(frozen=True)
class ChunkShapeConfig:
    frames_per_chunk: int
    latent_h: int
    latent_w: int
    latent_channels: int
    patch_size: tuple[int, int, int]
    source: str

    @property
    def tokens_per_chunk(self) -> int:
        pt, ph, pw = self.patch_size
        if self.frames_per_chunk % pt or self.latent_h % ph or self.latent_w % pw:
            raise ValueError(
                "latent chunk shape must be divisible by patch_size: "
                f"frames={self.frames_per_chunk}, h={self.latent_h}, w={self.latent_w}, patch={self.patch_size}"
            )
        return (self.frames_per_chunk // pt) * (self.latent_h // ph) * (self.latent_w // pw)


def load_wan_arch_config(
    wan_config: str = "auto",
    checkpoint_dir: str | None = None,
    allow_hf: bool = True,
) -> WanArchConfig:
    # 配置来源优先级：显式 checkpoint/config -> 本地 Wan 模型目录 -> HF config -> 内置 1.3B fallback。
    # 这里暂时只读架构参数，不加载真实权重，方便后续把 checkpoint 接进来。
    candidates: list[Path] = []
    if checkpoint_dir:
        candidates.append(Path(checkpoint_dir) / "config.json")

    if wan_config not in ("auto", "hf", "builtin"):
        candidates.append(Path(wan_config))

    if wan_config == "auto":
        candidates.append(Path("Wan21/wan_models/Wan2.1-T2V-1.3B/config.json"))

    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return WanArchConfig.from_dict(json.load(f), source=str(path))

    if wan_config in ("auto", "hf") and allow_hf:
        try:
            with urllib.request.urlopen(HF_WAN13B_CONFIG_URL, timeout=3) as resp:
                return WanArchConfig.from_dict(json.loads(resp.read().decode("utf-8")), source=HF_WAN13B_CONFIG_URL)
        except Exception:
            if wan_config == "hf":
                raise

    return WanArchConfig.wan21_t2v_13b(source="builtin Wan2.1-T2V-1.3B fallback")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except Exception:
        OmegaConf = None

    if OmegaConf is not None:
        cfg = OmegaConf.load(path)
        return dict(OmegaConf.to_container(cfg, resolve=True))

    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("reading minWM yaml requires omegaconf or pyyaml") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_chunk_shape_config(
    minwm_config: str,
    patch_size: tuple[int, int, int],
    frames_per_chunk_override: int | None = None,
    latent_h_override: int | None = None,
    latent_w_override: int | None = None,
    latent_channels_override: int | None = None,
) -> ChunkShapeConfig:
    # minWM 的 latent shape 是 [B, F, C, H, W]；dummy benchmark 只需要每个 chunk 的
    # token 数，即 frames_per_chunk * (H / patch_h) * (W / patch_w)。
    source = "builtin minWM Wan defaults"
    data: dict[str, Any] = {}
    path = Path(minwm_config) if minwm_config else None
    if path and path.exists():
        data = _load_yaml(path)
        source = str(path)

    shape = data.get("image_or_video_shape", [1, 20, 16, 60, 104])
    if len(shape) != 5:
        raise ValueError(f"image_or_video_shape must be [B,F,C,H,W], got {shape}")

    frames_per_chunk = int(frames_per_chunk_override or data.get("num_frame_per_block", 4))
    latent_channels = int(latent_channels_override or shape[2])
    latent_h = int(latent_h_override or shape[3])
    latent_w = int(latent_w_override or shape[4])

    return ChunkShapeConfig(
        frames_per_chunk=frames_per_chunk,
        latent_h=latent_h,
        latent_w=latent_w,
        latent_channels=latent_channels,
        patch_size=patch_size,
        source=source,
    )


def parse_layers(value: str, arch: WanArchConfig) -> int:
    if value == "from_config":
        return arch.num_layers
    layers = int(value)
    if layers <= 0:
        raise ValueError(f"layers must be positive, got {layers}")
    return layers


def parse_patch_size(value: str | None, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if not value:
        return default
    parts = tuple(int(x) for x in value.replace("x", ",").split(","))
    if len(parts) != 3:
        raise ValueError(f"patch size must have three integers, got {value!r}")
    return parts


def maybe_override_arch(
    arch: WanArchConfig,
    d_model: int | None,
    heads: int | None,
    ffn_dim: int | None,
    patch_size: tuple[int, int, int] | None,
) -> WanArchConfig:
    return WanArchConfig(
        d_model=int(d_model or arch.d_model),
        ffn_dim=int(ffn_dim or arch.ffn_dim),
        num_heads=int(heads or arch.num_heads),
        num_layers=arch.num_layers,
        in_dim=arch.in_dim,
        out_dim=arch.out_dim,
        text_len=arch.text_len,
        eps=arch.eps,
        patch_size=patch_size or arch.patch_size,
        source=arch.source,
    )


def human_bytes(num_bytes: float) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = min(int(math.log(num_bytes, 1024)), len(units) - 1)
    return f"{num_bytes / (1024 ** idx):.2f} {units[idx]}"

#!/usr/bin/env python3
"""Wan DMD pipe-forcing inference entrypoint.

Run with torchrun and exactly four ranks. Rank i owns denoising step i, keeps
its own causal history cache, and streams chunk latents to rank i + 1.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

try:
    from .comm import recv_tensor, send_tensor
except ImportError:
    from comm import recv_tensor, send_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAN_ROOT = PROJECT_ROOT / "Wan21"


def _add_wan_to_path() -> None:
    wan_root = str(WAN_ROOT)
    if wan_root not in sys.path:
        sys.path.insert(0, wan_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan DMD four-stage pipe-forcing inference")
    parser.add_argument("--config_path", type=str, default="Wan21/configs/causal_forcing_dmd_camera.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="./ckpts/Wan21/Action2V/dmd/model.pt")
    parser.add_argument("--data_path", type=str, default="Wan21/prompts/demos.txt")
    parser.add_argument("--output_folder", type=str, default="output/pipe_forcing/wan_dmd_camera")
    parser.add_argument("--num_output_frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory", type=str, default="w*19")
    parser.add_argument("--trajectory_path", type=str, default=None)
    parser.add_argument("--max_prompts", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("pipe_forcing requires torchrun with --nproc_per_node=4")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size != 4:
        raise RuntimeError(f"pipe_forcing Wan DMD requires WORLD_SIZE=4, got {world_size}")
    return rank, world_size, local_rank, device


def load_config(config_path: str):
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load(str(WAN_ROOT / "configs/default_config.yaml"))
    return OmegaConf.merge(default_config, config)


def validate_config(config, world_size: int) -> None:
    if not hasattr(config, "denoising_step_list"):
        raise ValueError("pipe_forcing only supports Wan few-step DMD configs with denoising_step_list")
    if len(config.denoising_step_list) != world_size:
        raise ValueError(
            f"denoising_step_list must have {world_size} steps, got {len(config.denoising_step_list)}"
        )
    if int(getattr(config, "num_frame_per_block", 1)) != 4:
        raise ValueError("V1 pipe_forcing expects num_frame_per_block=4")
    if not bool(getattr(config, "causal", True)):
        raise ValueError("pipe_forcing only supports causal=true")
    if bool(getattr(config, "independent_first_frame", False)):
        raise ValueError("V1 pipe_forcing supports T2V only; independent_first_frame must be false")


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip() for line in f]


def build_camera_tensors(
    traj_str: Optional[str],
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not traj_str:
        return None, None

    from Wan21.wan_utils.camera_trajectory import parse_trajectory

    viewmats_np = parse_trajectory(traj_str)
    if len(viewmats_np) < num_frames:
        raise ValueError(
            f"trajectory '{traj_str}' has {len(viewmats_np)} frames but num_output_frames={num_frames}"
        )

    fx, fy, cx, cy = 0.5, 0.5, 0.5, 0.5
    Ks_np = np.array(
        [[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]] * len(viewmats_np),
        dtype=np.float32,
    )
    viewmats = torch.from_numpy(viewmats_np).unsqueeze(0).to(device=device, dtype=dtype)
    Ks = torch.from_numpy(Ks_np).unsqueeze(0).to(device=device, dtype=dtype)
    return viewmats, Ks


def chunk_camera(
    viewmats: Optional[torch.Tensor],
    Ks: Optional[torch.Tensor],
    start: int,
    num_frames: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    vm_chunk = viewmats[:, start:start + num_frames] if viewmats is not None else None
    ks_chunk = Ks[:, start:start + num_frames] if Ks is not None else None
    return vm_chunk, ks_chunk


def sanitize_filename(prompt: str, traj_str: Optional[str]) -> str:
    base = prompt[:100].strip() or "prompt"
    base = re.sub(r"[\\/:\0]+", "_", base)
    traj_suffix = ""
    if traj_str:
        traj_suffix = "_" + traj_str.replace("*", "").replace(",", "")
    return f"{base}{traj_suffix}.mp4"


def make_initial_noise(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    prompt_index: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 1_000_003 * int(prompt_index))
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def broadcast_prompt_embeds(prompt: str, text_encoder, rank: int, device: torch.device, dtype: torch.dtype) -> dict:
    shape = (1, 512, 4096)
    if rank == 0:
        prompt_embeds = text_encoder(text_prompts=[prompt])["prompt_embeds"].to(
            device=device,
            dtype=dtype,
        ).contiguous()
        if tuple(prompt_embeds.shape) != shape:
            raise ValueError(f"unexpected prompt embedding shape: {tuple(prompt_embeds.shape)}")
    else:
        prompt_embeds = torch.empty(shape, device=device, dtype=dtype)
    dist.broadcast(prompt_embeds, src=0)
    return {"prompt_embeds": prompt_embeds}


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = init_distributed()
    _add_wan_to_path()

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed + rank)

    # Import Wan modules after distributed init, matching the existing inference entrypoint.
    from Wan21.wan_utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
    try:
        from .wan_dmd_stage import WanDMDStage
    except ImportError:
        from wan_dmd_stage import WanDMDStage

    config = load_config(args.config_path)
    validate_config(config, world_size)

    dtype = torch.bfloat16
    stage = WanDMDStage(config, stage_index=rank, device=device, dtype=dtype, base_seed=args.seed)
    stage.load_checkpoint(args.checkpoint_path)
    stage.to_device()

    text_encoder = None
    if rank == 0:
        text_encoder = WanTextEncoder().eval().requires_grad_(False)
        text_encoder.to(device=device, dtype=dtype)

    vae = None
    if rank == world_size - 1:
        vae = WanVAEWrapper().eval().requires_grad_(False)
        vae.to(device=device, dtype=dtype)

    prompts = read_lines(args.data_path)
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
    trajectory_list = read_lines(args.trajectory_path) if args.trajectory_path else None
    if trajectory_list is not None and len(trajectory_list) < len(prompts):
        raise ValueError(
            f"trajectory_path has {len(trajectory_list)} lines but data_path has {len(prompts)} prompts"
        )

    if rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
        print("=== Pipe Forcing: Wan DMD 4-stage causal camera inference ===", flush=True)
        print(f"  Config:     {args.config_path}", flush=True)
        print(f"  Checkpoint: {args.checkpoint_path}", flush=True)
        print(f"  Output:     {args.output_folder}", flush=True)
        print(f"  Frames:     {args.num_output_frames}", flush=True)
    dist.barrier()

    block_size = int(config.num_frame_per_block)
    if args.num_output_frames % block_size != 0:
        raise ValueError(f"num_output_frames must be divisible by {block_size}")

    latent_shape_cfg = list(getattr(config, "image_or_video_shape", [1, 20, 16, 60, 104]))
    latent_channels, latent_height, latent_width = map(int, latent_shape_cfg[2:])
    chunk_shape = (1, block_size, latent_channels, latent_height, latent_width)
    full_shape = (1, args.num_output_frames, latent_channels, latent_height, latent_width)
    num_chunks = args.num_output_frames // block_size

    for prompt_index, prompt in enumerate(prompts):
        traj_str = trajectory_list[prompt_index] if trajectory_list else args.trajectory
        output_path = Path(args.output_folder) / sanitize_filename(prompt, traj_str)
        if output_path.exists() and not args.overwrite:
            if rank == 0:
                print(f"[skip] {output_path} already exists", flush=True)
            dist.barrier()
            continue

        conditional_dict = broadcast_prompt_embeds(prompt, text_encoder, rank, device, dtype)
        viewmats, Ks = build_camera_tensors(traj_str, args.num_output_frames, device, dtype)
        stage.reset_caches(batch_size=1, use_camera=(viewmats is not None))

        if rank == 0:
            sampled_noise = make_initial_noise(full_shape, device, dtype, args.seed, prompt_index)
        else:
            sampled_noise = None

        if rank == world_size - 1:
            output_latents = torch.empty(full_shape, device=device, dtype=dtype)
        else:
            output_latents = None

        dist.barrier()
        torch.cuda.synchronize(device)
        latent_t0 = time.perf_counter()

        for chunk_index in range(num_chunks):
            start_frame = chunk_index * block_size
            vm_chunk, ks_chunk = chunk_camera(viewmats, Ks, start_frame, block_size)

            if rank == 0:
                assert sampled_noise is not None
                chunk_latent = sampled_noise[:, start_frame:start_frame + block_size].contiguous()
            else:
                chunk_latent = recv_tensor(chunk_shape, dtype=dtype, device=device, src=rank - 1)

            next_latent, denoised_pred = stage.denoise_chunk(
                chunk_latent=chunk_latent,
                conditional_dict=conditional_dict,
                current_start_frame=start_frame,
                prompt_index=prompt_index,
                chunk_index=chunk_index,
                viewmats=vm_chunk,
                Ks=ks_chunk,
            )

            if rank < world_size - 1:
                send_tensor(next_latent, dst=rank + 1)
            else:
                assert output_latents is not None
                output_latents[:, start_frame:start_frame + block_size] = next_latent

            stage.append_context(
                denoised_pred=denoised_pred,
                conditional_dict=conditional_dict,
                current_start_frame=start_frame,
                viewmats=vm_chunk,
                Ks=ks_chunk,
            )

        torch.cuda.synchronize(device)
        latent_seconds = time.perf_counter() - latent_t0

        if rank == world_size - 1:
            assert output_latents is not None and vae is not None
            decode_t0 = time.perf_counter()
            video = vae.decode_to_pixel(output_latents, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            torch.cuda.synchronize(device)
            decode_seconds = time.perf_counter() - decode_t0

            current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
            video_u8 = (255.0 * current_video[0]).clamp(0, 255).to(torch.uint8)
            write_video(str(output_path), video_u8, fps=16)
            if hasattr(vae.model, "clear_cache"):
                vae.model.clear_cache()
            print(
                f"[pipe_forcing] prompt={prompt_index} latent={latent_seconds:.3f}s "
                f"decode={decode_seconds:.3f}s output={output_path}",
                flush=True,
            )

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

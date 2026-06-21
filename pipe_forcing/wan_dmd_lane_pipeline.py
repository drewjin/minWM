#!/usr/bin/env python3
"""Wan DMD chunk-lane wavefront pipeline.

This implements the user's chunk-owned schedule:

    owner(chunk c) = c % 4

Each rank/GPU owns a lane of chunks and runs all four denoising steps for those
chunks. After x[c, s] is produced, the latent is broadcast to every rank. Each
rank then writes that latent into its local static cache for step s via a
context forward. This preserves Wan's sink/sliding-window eviction logic while
using latent broadcast instead of KV-cache broadcast.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

try:
    from .wan_dmd_baselines import (
        WanDMDLocalRunner,
        build_camera_tensors,
        decode_and_write,
        load_config,
        read_lines,
        sanitize_filename,
        validate_config,
    )
except ImportError:
    from wan_dmd_baselines import (
        WanDMDLocalRunner,
        build_camera_tensors,
        decode_and_write,
        load_config,
        read_lines,
        sanitize_filename,
        validate_config,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAN_ROOT = PROJECT_ROOT / "Wan21"


def _add_wan_to_path() -> None:
    wan_root = str(WAN_ROOT)
    if wan_root not in sys.path:
        sys.path.insert(0, wan_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan DMD chunk-lane wavefront pipeline")
    parser.add_argument("--config_path", type=str, default="Wan21/configs/causal_forcing_dmd_camera.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="./ckpts/Wan21/Action2V/dmd/model.pt")
    parser.add_argument("--data_path", type=str, default="Wan21/prompts/demos.txt")
    parser.add_argument("--output_folder", type=str, default="output/pipe_forcing/wan_dmd_lane")
    parser.add_argument("--num_output_frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory", type=str, default="w*19")
    parser.add_argument("--trajectory_path", type=str, default=None)
    parser.add_argument("--max_prompts", type=int, default=-1)
    parser.add_argument("--no_decode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    os.environ["SP_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("lane pipeline requires torchrun with --nproc_per_node=4")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError(f"lane pipeline requires WORLD_SIZE=4, got {world_size}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


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


def make_and_broadcast_noise(
    full_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    prompt_index: int,
    rank: int,
) -> torch.Tensor:
    if rank == 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + 1_000_003 * int(prompt_index))
        noise = torch.randn(full_shape, device=device, dtype=dtype, generator=generator)
    else:
        noise = torch.empty(full_shape, device=device, dtype=dtype)
    dist.broadcast(noise, src=0)
    return noise


def diagonal_cells(tick: int, num_chunks: int, num_steps: int) -> list[tuple[int, int]]:
    cells = []
    for step_index in range(num_steps):
        chunk_index = tick - step_index
        if 0 <= chunk_index < num_chunks:
            cells.append((chunk_index, step_index))
    return cells


def main() -> None:
    args = parse_args()
    rank, world_size, _, device = init_distributed()
    _add_wan_to_path()
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed + rank)

    from Wan21.wan_utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

    config = load_config(args.config_path)
    validate_config(config)
    dtype = torch.bfloat16

    runner = WanDMDLocalRunner(config, device, dtype, args.seed)
    runner.load_checkpoint(args.checkpoint_path)
    runner.to_device()

    text_encoder = None
    if rank == 0:
        text_encoder = WanTextEncoder().eval().requires_grad_(False)
        text_encoder.to(device=device, dtype=dtype)

    vae = None
    if rank == 0 and not args.no_decode:
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
        print("=== Pipe Forcing: Wan DMD chunk-lane wavefront ===", flush=True)
        print(f"  Config:     {args.config_path}", flush=True)
        print(f"  Checkpoint: {args.checkpoint_path}", flush=True)
        print(f"  Output:     {args.output_folder}", flush=True)
        print(f"  Frames:     {args.num_output_frames}", flush=True)
    dist.barrier()

    block_size = int(config.num_frame_per_block)
    num_steps = len(config.denoising_step_list)
    if num_steps != world_size:
        raise ValueError(f"expected {world_size} denoising steps, got {num_steps}")
    if args.num_output_frames % block_size != 0:
        raise ValueError(f"num_output_frames must be divisible by {block_size}")

    latent_shape_cfg = list(getattr(config, "image_or_video_shape", [1, 20, 16, 60, 104]))
    latent_channels, latent_height, latent_width = map(int, latent_shape_cfg[2:])
    full_shape = (1, args.num_output_frames, latent_channels, latent_height, latent_width)
    chunk_shape = (1, block_size, latent_channels, latent_height, latent_width)
    num_chunks = args.num_output_frames // block_size

    for prompt_index, prompt in enumerate(prompts):
        traj_str = trajectory_list[prompt_index] if trajectory_list else args.trajectory
        output_path = Path(args.output_folder) / sanitize_filename(prompt, traj_str, "lane")
        if output_path.exists() and not args.overwrite and not args.no_decode:
            if rank == 0:
                print(f"[skip] {output_path} already exists", flush=True)
            dist.barrier()
            continue

        conditional_dict = broadcast_prompt_embeds(prompt, text_encoder, rank, device, dtype)
        viewmats, Ks = build_camera_tensors(traj_str, args.num_output_frames, device, dtype)
        sampled_noise = make_and_broadcast_noise(full_shape, device, dtype, args.seed, prompt_index, rank)

        runner.reset_cache_bank(
            num_stages=num_steps,
            batch_size=1,
            use_camera=(viewmats is not None),
        )

        local_next_latents: dict[int, torch.Tensor] = {}
        output_latents = torch.empty(full_shape, device=device, dtype=dtype) if rank == 0 else None

        dist.barrier()
        torch.cuda.synchronize(device)
        latent_t0 = time.perf_counter()

        for tick in range(num_chunks + num_steps - 1):
            produced: dict[tuple[int, int], torch.Tensor] = {}
            cells = diagonal_cells(tick, num_chunks, num_steps)

            for chunk_index, step_index in cells:
                if chunk_index % world_size != rank:
                    continue

                start_frame = chunk_index * block_size
                if step_index == 0:
                    chunk_latent = sampled_noise[:, start_frame:start_frame + block_size].contiguous()
                else:
                    chunk_latent = local_next_latents.pop(chunk_index)

                vm_chunk = viewmats[:, start_frame:start_frame + block_size] if viewmats is not None else None
                ks_chunk = Ks[:, start_frame:start_frame + block_size] if Ks is not None else None
                next_latent, denoised_pred = runner.denoise_step(
                    step_index=step_index,
                    chunk_latent=chunk_latent,
                    conditional_dict=conditional_dict,
                    current_start_frame=start_frame,
                    prompt_index=prompt_index,
                    chunk_index=chunk_index,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                )
                produced[(chunk_index, step_index)] = denoised_pred.contiguous()
                if step_index < num_steps - 1:
                    local_next_latents[chunk_index] = next_latent.detach()

            for chunk_index, step_index in cells:
                owner = chunk_index % world_size
                if rank == owner:
                    denoised_pred = produced[(chunk_index, step_index)]
                else:
                    denoised_pred = torch.empty(chunk_shape, device=device, dtype=dtype)
                dist.broadcast(denoised_pred, src=owner)

                start_frame = chunk_index * block_size
                vm_chunk = viewmats[:, start_frame:start_frame + block_size] if viewmats is not None else None
                ks_chunk = Ks[:, start_frame:start_frame + block_size] if Ks is not None else None
                runner.append_step_context(
                    step_index=step_index,
                    denoised_pred=denoised_pred,
                    conditional_dict=conditional_dict,
                    current_start_frame=start_frame,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                )

                if rank == 0 and step_index == num_steps - 1:
                    assert output_latents is not None
                    output_latents[:, start_frame:start_frame + block_size] = denoised_pred

        torch.cuda.synchronize(device)
        latent_seconds = time.perf_counter() - latent_t0

        decode_seconds = 0.0
        if rank == 0 and not args.no_decode:
            assert vae is not None and output_latents is not None
            decode_seconds = decode_and_write(vae, output_latents, output_path, device)

        if rank == 0:
            output_msg = "no_decode" if args.no_decode else str(output_path)
            print(
                f"[lane_pipeline] prompt={prompt_index} "
                f"latent={latent_seconds:.3f}s decode={decode_seconds:.3f}s "
                f"output={output_msg}",
                flush=True,
            )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

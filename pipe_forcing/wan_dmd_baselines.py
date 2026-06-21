#!/usr/bin/env python3
"""Wan DMD baselines for pipe-forcing experiments.

Modes:
- append: run the same heuristic step-cache computation as pipe forcing, but
  locally and sequentially. There is no inter-rank latent transfer.
- single_chunk: run plain 4-step generation for one chunk only.

For the "tp4" comparison requested by the experiment, this file uses the
existing Wan sequence-parallel implementation with sp_size=4. The repo does
not currently expose a separate Wan tensor-parallel inference path.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAN_ROOT = PROJECT_ROOT / "Wan21"


def _add_wan_to_path() -> None:
    wan_root = str(WAN_ROOT)
    if wan_root not in sys.path:
        sys.path.insert(0, wan_root)


@dataclass
class Runtime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    main: bool
    distributed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan DMD pipe-forcing baselines")
    parser.add_argument("--mode", choices=["append", "single_chunk"], required=True)
    parser.add_argument("--config_path", type=str, default="Wan21/configs/causal_forcing_dmd_camera.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="./ckpts/Wan21/Action2V/dmd/model.pt")
    parser.add_argument("--data_path", type=str, default="Wan21/prompts/demos.txt")
    parser.add_argument("--output_folder", type=str, default="output/pipe_forcing/baselines")
    parser.add_argument("--num_output_frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory", type=str, default="w*19")
    parser.add_argument("--trajectory_path", type=str, default=None)
    parser.add_argument("--max_prompts", type=int, default=-1)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--no_decode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def init_runtime(sp_size: int) -> Runtime:
    os.environ["SP_SIZE"] = str(sp_size)
    _add_wan_to_path()

    if sp_size > 1:
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("sp_size > 1 requires torchrun")
        world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size_env < sp_size:
            raise RuntimeError(f"sp_size={sp_size} requires WORLD_SIZE >= {sp_size}")
        from Wan21.wan_utils.distributed import launch_distributed_job

        launch_distributed_job(backend="nccl", sp_size=sp_size)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        return Runtime(rank, world_size, local_rank, device, rank == 0, True)

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return Runtime(rank, world_size, local_rank, device, rank == 0, True)

    return Runtime(0, 1, 0, torch.device("cuda"), True, False)


def load_config(config_path: str):
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load(str(WAN_ROOT / "configs/default_config.yaml"))
    return OmegaConf.merge(default_config, config)


def validate_config(config) -> None:
    if not hasattr(config, "denoising_step_list"):
        raise ValueError("baselines only support Wan few-step DMD configs")
    if len(config.denoising_step_list) != 4:
        raise ValueError(f"expected 4 denoising steps, got {len(config.denoising_step_list)}")
    if int(getattr(config, "num_frame_per_block", 1)) != 4:
        raise ValueError("V1 baselines expect num_frame_per_block=4")
    if not bool(getattr(config, "causal", True)):
        raise ValueError("baselines only support causal=true")
    if bool(getattr(config, "independent_first_frame", False)):
        raise ValueError("V1 baselines support T2V only; independent_first_frame must be false")


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
            f"trajectory '{traj_str}' has {len(viewmats_np)} frames but need {num_frames}"
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


def sanitize_filename(prompt: str, traj_str: Optional[str], suffix: str) -> str:
    base = prompt[:100].strip() or "prompt"
    base = re.sub(r"[\\/:\0]+", "_", base)
    traj_suffix = ""
    if traj_str:
        traj_suffix = "_" + traj_str.replace("*", "").replace(",", "")
    return f"{base}{traj_suffix}_{suffix}.mp4"


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


def broadcast_prompt_embeds(prompt: str, text_encoder, runtime: Runtime, dtype: torch.dtype) -> dict:
    shape = (1, 512, 4096)
    if runtime.main:
        prompt_embeds = text_encoder(text_prompts=[prompt])["prompt_embeds"].to(
            device=runtime.device,
            dtype=dtype,
        ).contiguous()
        if tuple(prompt_embeds.shape) != shape:
            raise ValueError(f"unexpected prompt embedding shape: {tuple(prompt_embeds.shape)}")
    else:
        prompt_embeds = torch.empty(shape, device=runtime.device, dtype=dtype)

    if runtime.distributed:
        dist.broadcast(prompt_embeds, src=0)
    return {"prompt_embeds": prompt_embeds}


class WanDMDLocalRunner:
    frame_seq_length = 1560

    def __init__(
        self,
        config,
        device: torch.device,
        dtype: torch.dtype,
        base_seed: int,
    ) -> None:
        from Wan21.wan_utils.wan_wrapper import WanDiffusionWrapper

        self.config = config
        self.device = device
        self.dtype = dtype
        self.base_seed = int(base_seed)
        self.generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}),
            is_causal=True,
        )
        self.generator.eval().requires_grad_(False)
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = self._resolve_denoising_steps(config)
        self.num_frame_per_block = int(getattr(config, "num_frame_per_block", 1))
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.local_attn_size = self.generator.model.local_attn_size
        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.cache_bank = []

    def _resolve_denoising_steps(self, config) -> torch.Tensor:
        denoising_steps = torch.tensor(config.denoising_step_list, dtype=torch.long)
        if getattr(config, "warp_denoising_step", False):
            timesteps = torch.cat(
                (
                    self.scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            denoising_steps = timesteps[1000 - denoising_steps]
        return denoising_steps

    def load_checkpoint(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "generator_ema" in state_dict:
            gen_sd = state_dict["generator_ema"]
        elif "generator" in state_dict:
            gen_sd = state_dict["generator"]
        else:
            gen_sd = state_dict

        try:
            self.generator.load_state_dict(gen_sd)
        except RuntimeError:
            fixed = {}
            for key, value in gen_sd.items():
                if key.startswith("model._fsdp_wrapped_module."):
                    key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
                fixed[key] = value
            self.generator.load_state_dict(fixed, strict=False)

    def to_device(self) -> None:
        self.generator.to(device=self.device, dtype=self.dtype)
        self.denoising_step_list = self.denoising_step_list.to(self.device)

    def reset_cache_bank(self, num_stages: int, batch_size: int, use_camera: bool) -> None:
        self.cache_bank = [self._new_cache_set(batch_size, use_camera) for _ in range(num_stages)]

    @torch.inference_mode()
    def generate_append(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        prompt_index: int,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, num_frames, channels, height, width = noise.shape
        block = self.num_frame_per_block
        if num_frames % block != 0:
            raise ValueError(f"num_frames={num_frames} must be divisible by block={block}")

        output = torch.empty(
            [1, num_frames, channels, height, width],
            device=self.device,
            dtype=self.dtype,
        )
        self.reset_cache_bank(
            num_stages=len(self.denoising_step_list),
            batch_size=noise.shape[0],
            use_camera=(viewmats is not None),
        )

        for chunk_index, start_frame in enumerate(range(0, num_frames, block)):
            vm_chunk, ks_chunk = chunk_camera(viewmats, Ks, start_frame, block)
            chunk_latent = noise[:, start_frame:start_frame + block].contiguous()

            for stage_index in range(len(self.denoising_step_list)):
                next_latent, denoised_pred = self._denoise_stage(
                    stage_index=stage_index,
                    chunk_latent=chunk_latent,
                    conditional_dict=conditional_dict,
                    current_start_frame=start_frame,
                    prompt_index=prompt_index,
                    chunk_index=chunk_index,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                )
                self._append_context(
                    stage_index=stage_index,
                    denoised_pred=denoised_pred,
                    conditional_dict=conditional_dict,
                    current_start_frame=start_frame,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                )
                chunk_latent = next_latent

            output[:, start_frame:start_frame + block] = chunk_latent

        return output

    @torch.inference_mode()
    def generate_single_chunk(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        prompt_index: int,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        block = self.num_frame_per_block
        chunk_latent = noise[:, :block].contiguous()
        self.reset_cache_bank(num_stages=1, batch_size=noise.shape[0], use_camera=(viewmats is not None))
        vm_chunk, ks_chunk = chunk_camera(viewmats, Ks, 0, block)

        for stage_index in range(len(self.denoising_step_list)):
            next_latent, _ = self._denoise_with_cache(
                cache_index=0,
                stage_index=stage_index,
                chunk_latent=chunk_latent,
                conditional_dict=conditional_dict,
                current_start_frame=0,
                prompt_index=prompt_index,
                chunk_index=0,
                viewmats=vm_chunk,
                Ks=ks_chunk,
            )
            chunk_latent = next_latent

        return chunk_latent

    def _denoise_stage(
        self,
        stage_index: int,
        chunk_latent: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
        prompt_index: int,
        chunk_index: int,
        viewmats: Optional[torch.Tensor],
        Ks: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._denoise_with_cache(
            cache_index=stage_index,
            stage_index=stage_index,
            chunk_latent=chunk_latent,
            conditional_dict=conditional_dict,
            current_start_frame=current_start_frame,
            prompt_index=prompt_index,
            chunk_index=chunk_index,
            viewmats=viewmats,
            Ks=Ks,
        )

    def denoise_step(
        self,
        step_index: int,
        chunk_latent: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
        prompt_index: int,
        chunk_index: int,
        viewmats: Optional[torch.Tensor],
        Ks: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._denoise_with_cache(
            cache_index=step_index,
            stage_index=step_index,
            chunk_latent=chunk_latent,
            conditional_dict=conditional_dict,
            current_start_frame=current_start_frame,
            prompt_index=prompt_index,
            chunk_index=chunk_index,
            viewmats=viewmats,
            Ks=Ks,
        )

    def append_step_context(
        self,
        step_index: int,
        denoised_pred: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
        viewmats: Optional[torch.Tensor],
        Ks: Optional[torch.Tensor],
    ) -> None:
        self._append_context(
            stage_index=step_index,
            denoised_pred=denoised_pred,
            conditional_dict=conditional_dict,
            current_start_frame=current_start_frame,
            viewmats=viewmats,
            Ks=Ks,
        )

    def _denoise_with_cache(
        self,
        cache_index: int,
        stage_index: int,
        chunk_latent: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
        prompt_index: int,
        chunk_index: int,
        viewmats: Optional[torch.Tensor],
        Ks: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_set = self.cache_bank[cache_index]
        timestep = torch.ones(
            [chunk_latent.shape[0], chunk_latent.shape[1]],
            device=self.device,
            dtype=torch.int64,
        ) * self.denoising_step_list[stage_index]

        _, denoised_pred = self.generator(
            noisy_image_or_video=chunk_latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=cache_set["kv"],
            crossattn_cache=cache_set["cross"],
            current_start=current_start_frame * self.frame_seq_length,
            viewmats=viewmats,
            Ks=Ks,
            prope_kv_cache=cache_set["prope"],
        )

        if stage_index == len(self.denoising_step_list) - 1:
            return denoised_pred.contiguous(), denoised_pred

        next_timestep = self.denoising_step_list[stage_index + 1] * torch.ones(
            [chunk_latent.shape[0] * chunk_latent.shape[1]],
            device=self.device,
            dtype=torch.long,
        )
        renoise = self._deterministic_noise_like(
            denoised_pred,
            prompt_index=prompt_index,
            stage_index=stage_index,
            chunk_index=chunk_index,
        )
        next_latent = self.scheduler.add_noise(
            denoised_pred.flatten(0, 1),
            renoise.flatten(0, 1),
            next_timestep,
        ).unflatten(0, denoised_pred.shape[:2])
        return next_latent.contiguous(), denoised_pred

    def _append_context(
        self,
        stage_index: int,
        denoised_pred: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
        viewmats: Optional[torch.Tensor],
        Ks: Optional[torch.Tensor],
    ) -> None:
        cache_set = self.cache_bank[stage_index]
        timestep = torch.ones(
            [denoised_pred.shape[0], denoised_pred.shape[1]],
            device=self.device,
            dtype=torch.int64,
        ) * getattr(self.config, "context_noise", 0)

        self.generator(
            noisy_image_or_video=denoised_pred.detach(),
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=cache_set["kv"],
            crossattn_cache=cache_set["cross"],
            current_start=current_start_frame * self.frame_seq_length,
            viewmats=viewmats,
            Ks=Ks,
            prope_kv_cache=cache_set["prope"],
        )

    def _new_cache_set(self, batch_size: int, use_camera: bool) -> dict[str, object]:
        return {
            "kv": self._new_kv_cache(batch_size),
            "cross": self._new_crossattn_cache(batch_size),
            "prope": self._new_kv_cache(batch_size) if use_camera else None,
        }

    def _new_kv_cache(self, batch_size: int) -> list[dict[str, torch.Tensor]]:
        num_heads = self._get_sp_num_heads(12)
        kv_cache_size = (
            self.local_attn_size * self.frame_seq_length
            if self.local_attn_size != -1
            else 31200
        )
        return [
            {
                "k": torch.zeros(
                    [batch_size, kv_cache_size, num_heads, 128],
                    dtype=self.dtype,
                    device=self.device,
                ),
                "v": torch.zeros(
                    [batch_size, kv_cache_size, num_heads, 128],
                    dtype=self.dtype,
                    device=self.device,
                ),
                "global_end_index": torch.zeros([1], dtype=torch.long, device=self.device),
                "local_end_index": torch.zeros([1], dtype=torch.long, device=self.device),
            }
            for _ in range(self.num_transformer_blocks)
        ]

    def _new_crossattn_cache(self, batch_size: int) -> list[dict[str, object]]:
        return [
            {
                "k": torch.zeros(
                    [batch_size, 512, 12, 128],
                    dtype=self.dtype,
                    device=self.device,
                ),
                "v": torch.zeros(
                    [batch_size, 512, 12, 128],
                    dtype=self.dtype,
                    device=self.device,
                ),
                "is_init": False,
            }
            for _ in range(self.num_transformer_blocks)
        ]

    def _deterministic_noise_like(
        self,
        tensor: torch.Tensor,
        prompt_index: int,
        stage_index: int,
        chunk_index: int,
    ) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        seed = (
            self.base_seed
            + 1_000_003 * int(prompt_index)
            + 10_007 * int(stage_index)
            + 97 * int(chunk_index)
        )
        generator.manual_seed(seed)
        return torch.randn(
            tensor.shape,
            device=self.device,
            dtype=tensor.dtype,
            generator=generator,
        )

    @staticmethod
    def _get_sp_num_heads(full_num_heads: int) -> int:
        try:
            from shared.sp.parallel_states import get_parallel_state

            ps = get_parallel_state()
            if ps.sp_enabled:
                return full_num_heads // ps.sp
        except (ImportError, AttributeError):
            pass
        return full_num_heads


def decode_and_write(
    vae,
    latents: torch.Tensor,
    output_path: Path,
    device: torch.device,
) -> float:
    decode_t0 = time.perf_counter()
    video = vae.decode_to_pixel(latents, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_t0

    current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
    video_u8 = (255.0 * current_video[0]).clamp(0, 255).to(torch.uint8)
    write_video(str(output_path), video_u8, fps=16)
    if hasattr(vae.model, "clear_cache"):
        vae.model.clear_cache()
    return decode_seconds


def main() -> None:
    args = parse_args()
    runtime = init_runtime(args.sp_size)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed + runtime.rank)

    from Wan21.wan_utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

    config = load_config(args.config_path)
    validate_config(config)
    dtype = torch.bfloat16

    runner = WanDMDLocalRunner(config, runtime.device, dtype, args.seed)
    runner.load_checkpoint(args.checkpoint_path)
    runner.to_device()

    text_encoder = None
    if runtime.main:
        text_encoder = WanTextEncoder().eval().requires_grad_(False)
        text_encoder.to(device=runtime.device, dtype=dtype)

    vae = None
    if runtime.main and not args.no_decode:
        vae = WanVAEWrapper().eval().requires_grad_(False)
        vae.to(device=runtime.device, dtype=dtype)

    prompts = read_lines(args.data_path)
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
    trajectory_list = read_lines(args.trajectory_path) if args.trajectory_path else None
    if trajectory_list is not None and len(trajectory_list) < len(prompts):
        raise ValueError(
            f"trajectory_path has {len(trajectory_list)} lines but data_path has {len(prompts)} prompts"
        )

    if runtime.main:
        os.makedirs(args.output_folder, exist_ok=True)
        print("=== Pipe Forcing Baseline: Wan DMD ===", flush=True)
        print(f"  Mode:       {args.mode}", flush=True)
        print(f"  SP size:    {args.sp_size}", flush=True)
        print(f"  Config:     {args.config_path}", flush=True)
        print(f"  Checkpoint: {args.checkpoint_path}", flush=True)
        print(f"  Output:     {args.output_folder}", flush=True)
    if runtime.distributed:
        dist.barrier()

    block_size = int(config.num_frame_per_block)
    effective_frames = block_size if args.mode == "single_chunk" else args.num_output_frames
    if effective_frames % block_size != 0:
        raise ValueError(f"num_output_frames must be divisible by {block_size}")

    latent_shape_cfg = list(getattr(config, "image_or_video_shape", [1, 20, 16, 60, 104]))
    latent_channels, latent_height, latent_width = map(int, latent_shape_cfg[2:])
    full_shape = (1, effective_frames, latent_channels, latent_height, latent_width)

    for prompt_index, prompt in enumerate(prompts):
        traj_str = trajectory_list[prompt_index] if trajectory_list else args.trajectory
        suffix = f"{args.mode}_sp{args.sp_size}"
        output_path = Path(args.output_folder) / sanitize_filename(prompt, traj_str, suffix)
        if output_path.exists() and not args.overwrite and not args.no_decode:
            if runtime.main:
                print(f"[skip] {output_path} already exists", flush=True)
            if runtime.distributed:
                dist.barrier()
            continue

        conditional_dict = broadcast_prompt_embeds(prompt, text_encoder, runtime, dtype)
        viewmats, Ks = build_camera_tensors(traj_str, effective_frames, runtime.device, dtype)
        sampled_noise = make_initial_noise(full_shape, runtime.device, dtype, args.seed, prompt_index)

        if runtime.distributed:
            dist.barrier()
        torch.cuda.synchronize(runtime.device)
        latent_t0 = time.perf_counter()
        if args.mode == "append":
            latents = runner.generate_append(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                prompt_index=prompt_index,
                viewmats=viewmats,
                Ks=Ks,
            )
        else:
            latents = runner.generate_single_chunk(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                prompt_index=prompt_index,
                viewmats=viewmats,
                Ks=Ks,
            )
        torch.cuda.synchronize(runtime.device)
        latent_seconds = time.perf_counter() - latent_t0

        decode_seconds = 0.0
        if runtime.main and not args.no_decode:
            assert vae is not None
            decode_seconds = decode_and_write(vae, latents, output_path, runtime.device)

        if runtime.main:
            output_msg = "no_decode" if args.no_decode else str(output_path)
            print(
                f"[baseline:{args.mode}] prompt={prompt_index} "
                f"latent={latent_seconds:.3f}s decode={decode_seconds:.3f}s "
                f"output={output_msg}",
                flush=True,
            )

        if runtime.distributed:
            dist.barrier()

    if runtime.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

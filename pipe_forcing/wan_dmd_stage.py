"""Single-stage Wan DMD pipe-forcing worker.

Each distributed rank owns one denoising step and keeps its own causal KV
cache. The historical cache is updated with the early x0 prediction produced
by that same stage, which is the heuristic part of pipe forcing.
"""

from __future__ import annotations

from typing import Optional

import torch

from Wan21.wan_utils.wan_wrapper import WanDiffusionWrapper


class WanDMDStage:
    """One GPU stage for a Wan causal few-step DMD pipeline."""

    frame_seq_length = 1560

    def __init__(
        self,
        config,
        stage_index: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        base_seed: int = 0,
    ) -> None:
        self.config = config
        self.stage_index = stage_index
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
        if len(self.denoising_step_list) <= stage_index:
            raise ValueError(
                f"stage_index={stage_index} but denoising_step_list has "
                f"{len(self.denoising_step_list)} steps"
            )

        self.step_timestep = self.denoising_step_list[stage_index]
        self.next_timestep = (
            self.denoising_step_list[stage_index + 1]
            if stage_index + 1 < len(self.denoising_step_list)
            else None
        )

        self.num_frame_per_block = int(getattr(config, "num_frame_per_block", 1))
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.local_attn_size = self.generator.model.local_attn_size
        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.kv_cache = None
        self.crossattn_cache = None
        self.prope_kv_cache = None

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
        self.step_timestep = self.step_timestep.to(self.device)
        if self.next_timestep is not None:
            self.next_timestep = self.next_timestep.to(self.device)

    def reset_caches(self, batch_size: int, use_camera: bool) -> None:
        if self.kv_cache is None:
            self._initialize_kv_cache(batch_size=batch_size)
            self._initialize_crossattn_cache(batch_size=batch_size)
            if use_camera:
                self._initialize_prope_kv_cache(batch_size=batch_size)
            return

        for block_cache in self.kv_cache:
            block_cache["global_end_index"].zero_()
            block_cache["local_end_index"].zero_()

        for block_cache in self.crossattn_cache:
            block_cache["is_init"] = False

        if use_camera and self.prope_kv_cache is None:
            self._initialize_prope_kv_cache(batch_size=batch_size)
        if self.prope_kv_cache is not None:
            for block_cache in self.prope_kv_cache:
                block_cache["global_end_index"].zero_()
                block_cache["local_end_index"].zero_()

    @torch.inference_mode()
    def denoise_chunk(
        self,
        chunk_latent: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
        prompt_index: int,
        chunk_index: int,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_num_frames = chunk_latent.shape[1]
        timestep = self._make_timestep(self.step_timestep, chunk_latent.shape[0], current_num_frames)

        _, denoised_pred = self.generator(
            noisy_image_or_video=chunk_latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start_frame * self.frame_seq_length,
            viewmats=viewmats,
            Ks=Ks,
            prope_kv_cache=self.prope_kv_cache,
        )

        if self.next_timestep is None:
            return denoised_pred.contiguous(), denoised_pred

        next_timestep = self.next_timestep * torch.ones(
            [chunk_latent.shape[0] * current_num_frames],
            device=self.device,
            dtype=torch.long,
        )
        renoise = self._deterministic_noise_like(
            denoised_pred,
            prompt_index=prompt_index,
            chunk_index=chunk_index,
        )
        next_latent = self.scheduler.add_noise(
            denoised_pred.flatten(0, 1),
            renoise.flatten(0, 1),
            next_timestep,
        ).unflatten(0, denoised_pred.shape[:2])
        return next_latent.contiguous(), denoised_pred

    @torch.inference_mode()
    def append_context(
        self,
        denoised_pred: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> None:
        timestep = torch.ones(
            [denoised_pred.shape[0], denoised_pred.shape[1]],
            device=self.device,
            dtype=torch.int64,
        ) * getattr(self.config, "context_noise", 0)

        self.generator(
            noisy_image_or_video=denoised_pred.detach(),
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start_frame * self.frame_seq_length,
            viewmats=viewmats,
            Ks=Ks,
            prope_kv_cache=self.prope_kv_cache,
        )

    def _make_timestep(self, value: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        return torch.ones(
            [batch_size, num_frames],
            device=self.device,
            dtype=torch.int64,
        ) * value

    def _deterministic_noise_like(
        self,
        tensor: torch.Tensor,
        prompt_index: int,
        chunk_index: int,
    ) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        seed = (
            self.base_seed
            + 1_000_003 * int(prompt_index)
            + 10_007 * int(self.stage_index)
            + 97 * int(chunk_index)
        )
        generator.manual_seed(seed)
        return torch.randn(
            tensor.shape,
            device=self.device,
            dtype=tensor.dtype,
            generator=generator,
        )

    def _initialize_kv_cache(self, batch_size: int) -> None:
        num_heads = self._get_sp_num_heads(12)
        kv_cache_size = (
            self.local_attn_size * self.frame_seq_length
            if self.local_attn_size != -1
            else 31200
        )
        self.kv_cache = [
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

    def _initialize_crossattn_cache(self, batch_size: int) -> None:
        self.crossattn_cache = [
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

    def _initialize_prope_kv_cache(self, batch_size: int) -> None:
        num_heads = self._get_sp_num_heads(12)
        kv_cache_size = (
            self.local_attn_size * self.frame_seq_length
            if self.local_attn_size != -1
            else 31200
        )
        self.prope_kv_cache = [
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

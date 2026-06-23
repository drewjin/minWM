from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from wavefront_test.block import build_blocks
from wavefront_test.config import (
    ChunkShapeConfig,
    human_bytes,
    load_chunk_shape_config,
    load_wan_arch_config,
    maybe_override_arch,
    parse_layers,
    parse_patch_size,
)
from wavefront_test.pf_comm import FrontierKVComm
from wavefront_test.profiler import BenchMetrics, gather_summaries
from wavefront_test.sp_comm import head_parallel_to_seq, seq_to_head_parallel


MODES = ("correctness", "pipe_blocking_kv", "pipe_kv_prefetch", "sp_dense", "sp_causal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan-like PipeForcing vs SP dummy benchmark")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--wan-config", default="auto", help="auto, hf, builtin, or a config.json path")
    parser.add_argument("--checkpoint-dir", default=None, help="future weight path; v1 reads config.json only")
    parser.add_argument("--minwm-config", default="Wan21/configs/causal_forcing_dmd_camera.yaml")

    parser.add_argument("--layers", default="from_config", help="'from_config' or an integer")
    parser.add_argument("--tokens-per-chunk", type=int, default=None)
    parser.add_argument("--frames-per-chunk", type=int, default=None)
    parser.add_argument("--latent-h", type=int, default=None)
    parser.add_argument("--latent-w", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--patch-size", default=None, help="e.g. 1,2,2")

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefetch-hit-threshold-ms", type=float, default=1.0)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, torch.device]:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("run with torchrun, for example: torchrun --nproc_per_node=4 -m wavefront_test.run_dummy_pf")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    try:
        dist.init_process_group(backend="nccl", device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, device


def parse_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def make_local_hidden(n_tokens: int, d_model: int, dtype: torch.dtype, device: torch.device, seed: int, rank: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1009 * rank)
    return torch.randn((n_tokens, d_model), generator=gen, device=device, dtype=dtype)


def make_all_hidden(
    world_size: int,
    n_tokens: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn((world_size, n_tokens, d_model), generator=gen, dtype=torch.float32)
    return x.to(device=device, dtype=dtype)


def chronological_kv(
    rank: int,
    world_size: int,
    all_k: list[torch.Tensor],
    all_v: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    # 单卡 reference 使用同一套 frontier 规则：query rank r attend [oldest ... r]。
    srcs = list(range(world_size - 1, rank - 1, -1))
    return [all_k[src] for src in srcs], [all_v[src] for src in srcs]


@torch.no_grad()
def run_reference(blocks, x_chunks: torch.Tensor, world_size: int) -> torch.Tensor:
    xs = [x_chunks[r].clone() for r in range(world_size)]
    for block in blocks:
        qkv = [block.compute_qkv(x) for x in xs]
        next_xs = []
        all_q = [item[0] for item in qkv]
        all_k = [item[1] for item in qkv]
        all_v = [item[2] for item in qkv]
        for rank in range(world_size):
            k_list, v_list = chronological_kv(rank, world_size, all_k, all_v)
            attn = block.self_attn(all_q[rank], k_list, v_list)
            next_xs.append(block.finish_block(xs[rank], attn))
        xs = next_xs
    return torch.stack(xs, dim=0)


@torch.no_grad()
def run_pipe_blocking(blocks, x: torch.Tensor, metrics: BenchMetrics, args: argparse.Namespace) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    comm = FrontierKVComm(rank, world_size, x.shape[0], blocks[0].num_heads, blocks[0].head_dim, x.dtype, x.device)
    for layer_id, block in enumerate(blocks):
        layer_t0 = time.perf_counter()
        metrics.start_cuda("qkv")
        q, k, v = block.compute_qkv(x)
        metrics.stop_cuda("qkv")

        send_t0 = time.perf_counter()
        remote, bytes_sent, bytes_recv, wait_sec = comm.exchange_blocking(layer_id, k, v)
        metrics.add("send_post_sec", max(0.0, time.perf_counter() - send_t0 - wait_sec))
        metrics.add("bytes_sent", bytes_sent)
        metrics.add("remote_kv_wait_sec", wait_sec)
        metrics.add("bytes_recv", bytes_recv)
        k_list, v_list = comm.assemble_chronological(remote, k, v)
        metrics.start_cuda("attn")
        attn = block.self_attn(q, k_list, v_list)
        metrics.stop_cuda("attn")

        metrics.start_cuda("mlp")
        x = block.finish_block(x, attn)
        metrics.stop_cuda("mlp")
        metrics.add("layer_total_sec", time.perf_counter() - layer_t0)

    comm.wait_sends()
    return x


@torch.no_grad()
def run_pipe_prefetch(
    blocks,
    x: torch.Tensor,
    metrics: BenchMetrics,
    args: argparse.Namespace,
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    comm = FrontierKVComm(
        rank,
        world_size,
        x.shape[0],
        blocks[0].num_heads,
        blocks[0].head_dim,
        x.dtype,
        x.device,
        direct_async=True,
    )
    hit_threshold = args.prefetch_hit_threshold_ms / 1000.0
    qkv_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    comm.post_recvs(0)
    for layer_id, block in enumerate(blocks):
        layer_t0 = time.perf_counter()
        comm.post_recvs(layer_id)

        if layer_id in qkv_cache:
            # 上一层已经提前算过当前层 Q/K/V，直接复用，避免重复 projection。
            q, k, v = qkv_cache.pop(layer_id)
        else:
            metrics.start_cuda("qkv")
            q, k, v = block.compute_qkv(x)
            metrics.stop_cuda("qkv")

        send_t0 = time.perf_counter()
        bytes_sent = comm.send_kv(layer_id, k, v)
        metrics.add("send_post_sec", time.perf_counter() - send_t0)
        metrics.add("bytes_sent", bytes_sent)

        remote, bytes_recv, wait_sec = comm.wait_recvs(layer_id)
        metrics.add("remote_kv_wait_sec", wait_sec)
        metrics.add("bytes_recv", bytes_recv)
        if comm.older_sources():
            metrics.inc("prefetch_layers")
            if wait_sec < hit_threshold:
                metrics.inc("prefetch_hits")

        k_list, v_list = comm.assemble_chronological(remote, k, v)
        metrics.start_cuda("attn")
        attn = block.self_attn(q, k_list, v_list)
        metrics.stop_cuda("attn")

        metrics.start_cuda("mlp")
        x = block.finish_block(x, attn)
        metrics.stop_cuda("mlp")

        next_layer = layer_id + 1
        if next_layer < len(blocks):
            # max_ahead=1：当前层 finish 后，立刻为下一层贴 recv、算本地 Q/K/V 并发 K/V。
            comm.post_recvs(next_layer)
            metrics.start_cuda("qkv_prefetch")
            q_next, k_next, v_next = blocks[next_layer].compute_qkv(x)
            metrics.stop_cuda("qkv_prefetch")
            qkv_cache[next_layer] = (q_next, k_next, v_next)
            send_t0 = time.perf_counter()
            bytes_sent = comm.send_kv(next_layer, k_next, v_next)
            metrics.add("send_post_sec", time.perf_counter() - send_t0)
            metrics.add("bytes_sent", bytes_sent)

        metrics.add("layer_total_sec", time.perf_counter() - layer_t0)

    comm.wait_sends()
    return x


def sdpa_from_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_sdpa = q.permute(1, 0, 2).unsqueeze(0)
    k_sdpa = k.permute(1, 0, 2).unsqueeze(0)
    v_sdpa = v.permute(1, 0, 2).unsqueeze(0)
    return F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False).squeeze(0).permute(1, 0, 2).contiguous()


@torch.no_grad()
def run_sp(blocks, x: torch.Tensor, metrics: BenchMetrics, causal: bool) -> torch.Tensor:
    world_size = dist.get_world_size()
    n_tokens = x.shape[0]
    for block in blocks:
        layer_t0 = time.perf_counter()
        metrics.start_cuda("qkv")
        q, k, v = block.compute_qkv(x)
        metrics.stop_cuda("qkv")

        metrics.start_cuda("sp_a2a_qkv")
        q_hp = seq_to_head_parallel(q)
        k_hp = seq_to_head_parallel(k)
        v_hp = seq_to_head_parallel(v)
        metrics.stop_cuda("sp_a2a_qkv")

        metrics.start_cuda("attn")
        if causal:
            # causal-aware SP 不构造 [4N, 4N] 大 mask，按 query chunk 分块调用 SDPA。
            out_chunks = []
            q_chunks = q_hp.split(n_tokens, dim=0)
            k_chunks = k_hp.split(n_tokens, dim=0)
            v_chunks = v_hp.split(n_tokens, dim=0)
            for chunk_id in range(world_size):
                k_allowed = torch.cat(k_chunks[chunk_id:world_size], dim=0)
                v_allowed = torch.cat(v_chunks[chunk_id:world_size], dim=0)
                out_chunks.append(sdpa_from_heads(q_chunks[chunk_id], k_allowed, v_allowed))
            out_hp = torch.cat(out_chunks, dim=0)
        else:
            out_hp = sdpa_from_heads(q_hp, k_hp, v_hp)
        metrics.stop_cuda("attn")

        metrics.start_cuda("sp_a2a_out")
        out_heads = head_parallel_to_seq(out_hp)
        metrics.stop_cuda("sp_a2a_out")

        metrics.start_cuda("mlp")
        attn = block.project_attention(out_heads)
        x = block.finish_block(x, attn)
        metrics.stop_cuda("mlp")

        remote_fraction = (world_size - 1) / world_size
        qkv_bytes = 3 * q.numel() * q.element_size() * remote_fraction
        out_bytes = out_heads.numel() * out_heads.element_size() * remote_fraction
        metrics.add("bytes_sent", qkv_bytes + out_bytes)
        metrics.add("bytes_recv", qkv_bytes + out_bytes)
        metrics.add("layer_total_sec", time.perf_counter() - layer_t0)
    return x


def run_one_iteration(
    mode: str,
    blocks,
    x: torch.Tensor,
    args: argparse.Namespace,
    metrics: BenchMetrics,
) -> torch.Tensor:
    if mode == "pipe_blocking_kv":
        return run_pipe_blocking(blocks, x, metrics, args)
    if mode == "pipe_kv_prefetch":
        return run_pipe_prefetch(blocks, x, metrics, args)
    if mode == "sp_dense":
        return run_sp(blocks, x, metrics, causal=False)
    if mode == "sp_causal":
        return run_sp(blocks, x, metrics, causal=True)
    raise ValueError(f"unsupported benchmark mode: {mode}")


def print_summary(
    args: argparse.Namespace,
    summaries: list[dict[str, float | int | str]],
    arch_source: str,
    chunk_cfg: ChunkShapeConfig,
    layers: int,
    n_tokens: int,
) -> None:
    summaries = sorted(summaries, key=lambda x: int(x["rank"]))
    print(
        f"\n=== wavefront_test summary mode={args.mode} layers={layers} "
        f"tokens_per_chunk={n_tokens} arch_source={arch_source} chunk_source={chunk_cfg.source} ===",
        flush=True,
    )
    for s in summaries:
        rank = int(s["rank"])
        iter_mean = float(s.get("iter_total_sec", 0.0)) / max(float(s.get("timed_iters", 1.0)), 1.0)
        layer_total = float(s.get("layer_total_sec", 0.0))
        wait = float(s.get("remote_kv_wait_sec", 0.0))
        wait_ratio = wait / layer_total if layer_total > 0 else 0.0
        hits = float(s.get("prefetch_hits", 0.0))
        hit_layers = float(s.get("prefetch_layers", 0.0))
        hit_rate = hits / hit_layers if hit_layers > 0 else 0.0
        print(
            "rank "
            f"{rank}: iter_mean={iter_mean:.4f}s layer_total={layer_total:.4f}s "
            f"qkv={float(s.get('qkv_sec', 0.0)):.4f}s qkv_prefetch={float(s.get('qkv_prefetch_sec', 0.0)):.4f}s "
            f"attn={float(s.get('attn_sec', 0.0)):.4f}s mlp={float(s.get('mlp_sec', 0.0)):.4f}s "
            f"sp_a2a={float(s.get('sp_a2a_qkv_sec', 0.0)) + float(s.get('sp_a2a_out_sec', 0.0)):.4f}s "
            f"remote_wait={wait:.4f}s wait_ratio={wait_ratio:.2%} hit_rate={hit_rate:.2%} "
            f"sent={human_bytes(float(s.get('bytes_sent', 0.0)))} recv={human_bytes(float(s.get('bytes_recv', 0.0)))}",
            flush=True,
        )
    rank0 = summaries[0]
    rank0_wait = float(rank0.get("remote_kv_wait_sec", 0.0))
    rank0_total = float(rank0.get("layer_total_sec", 0.0))
    print(f"rank0_remote_wait_ratio={rank0_wait / rank0_total if rank0_total > 0 else 0.0:.2%}", flush=True)


def write_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


@torch.no_grad()
def run_correctness(blocks, args: argparse.Namespace, n_tokens: int, d_model: int, dtype: torch.dtype, device: torch.device) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    x_all = make_all_hidden(world_size, n_tokens, d_model, dtype, device, args.seed + 17)
    ref = run_reference(blocks, x_all, world_size)

    pipe_metrics = BenchMetrics(rank=rank, mode="correctness_pipe")
    dist.barrier()
    x_pipe = run_pipe_blocking(blocks, x_all[rank].clone(), pipe_metrics, args)
    pipe_err = (x_pipe.float() - ref[rank].float()).abs().max()

    sp_metrics = BenchMetrics(rank=rank, mode="correctness_sp")
    dist.barrier()
    x_sp = run_sp(blocks, x_all[rank].clone(), sp_metrics, causal=True)
    sp_err = (x_sp.float() - ref[rank].float()).abs().max()

    errs = torch.stack([pipe_err, sp_err])
    gathered = [torch.empty_like(errs) for _ in range(world_size)]
    dist.all_gather(gathered, errs)
    if rank == 0:
        all_errs = torch.stack(gathered)
        pipe_max = all_errs[:, 0].max().item()
        sp_max = all_errs[:, 1].max().item()
        print(f"correctness: pipe_blocking_kv max_abs_error={pipe_max:.6g}", flush=True)
        print(f"correctness: sp_causal max_abs_error={sp_max:.6g}", flush=True)
        if pipe_max >= 1e-4 or sp_max >= 1e-4:
            raise RuntimeError(f"correctness failed: pipe={pipe_max}, sp={sp_max}")


def validate(world_size: int, d_model: int, heads: int, n_tokens: int, mode: str) -> None:
    if world_size != 4:
        raise RuntimeError(f"this dummy benchmark expects WORLD_SIZE=4, got {world_size}")
    if d_model % heads != 0:
        raise RuntimeError(f"d_model={d_model} must be divisible by heads={heads}")
    if heads % world_size != 0 and mode.startswith("sp"):
        raise RuntimeError(f"heads={heads} must be divisible by world_size={world_size} for SP")
    if n_tokens <= 0:
        raise RuntimeError(f"tokens_per_chunk must be positive, got {n_tokens}")


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_distributed()
    dtype = parse_dtype(args.dtype)
    torch.set_grad_enabled(False)

    arch = load_wan_arch_config(args.wan_config, checkpoint_dir=args.checkpoint_dir)
    patch_size = parse_patch_size(args.patch_size, arch.patch_size)
    arch = maybe_override_arch(arch, args.d_model, args.heads, args.ffn_dim, patch_size)
    layers = parse_layers(args.layers, arch)
    chunk_cfg = load_chunk_shape_config(
        args.minwm_config,
        patch_size=arch.patch_size,
        frames_per_chunk_override=args.frames_per_chunk,
        latent_h_override=args.latent_h,
        latent_w_override=args.latent_w,
        latent_channels_override=args.latent_channels,
    )
    n_tokens = int(args.tokens_per_chunk or chunk_cfg.tokens_per_chunk)
    validate(world_size, arch.d_model, arch.num_heads, n_tokens, args.mode)

    torch.manual_seed(args.seed)
    blocks = build_blocks(layers, arch.d_model, arch.num_heads, arch.ffn_dim, arch.eps, device, dtype)

    if rank == 0:
        print(
            f"config: d_model={arch.d_model} heads={arch.num_heads} head_dim={arch.head_dim} "
            f"ffn_dim={arch.ffn_dim} layers={layers} dtype={args.dtype} "
            f"tokens_per_chunk={n_tokens} patch={arch.patch_size}",
            flush=True,
        )

    if args.mode == "correctness":
        run_correctness(blocks, args, n_tokens, arch.d_model, dtype, device)
        dist.destroy_process_group()
        return

    if args.mode == "pipe_kv_prefetch":
        # 不把 P2P warmup 计入 benchmark；它只用来稳定 NCCL 首轮 communicator 初始化。
        warmup_comm = FrontierKVComm(
            rank,
            world_size,
            1,
            1,
            1,
            dtype,
            device,
            direct_async=True,
        )
        dist.barrier()
        warmup_comm.warmup_p2p()
        dist.barrier()

    metrics = BenchMetrics(rank=rank, mode=args.mode)
    total_iters = args.warmup + args.iters
    for iter_id in range(total_iters):
        x = make_local_hidden(n_tokens, arch.d_model, dtype, device, args.seed + 7919 * iter_id, rank)
        dist.barrier()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        x = run_one_iteration(
            args.mode,
            blocks,
            x,
            args,
            metrics if iter_id >= args.warmup else BenchMetrics(rank, args.mode),
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        if iter_id >= args.warmup:
            metrics.add("iter_total_sec", elapsed)
            metrics.inc("timed_iters")
        if not torch.isfinite(x).all():
            raise RuntimeError(f"rank {rank} produced non-finite output")

    local_summary = metrics.summary()
    summaries = gather_summaries(local_summary)
    if rank == 0:
        print_summary(args, summaries, arch.source, chunk_cfg, layers, n_tokens)
        if args.json_out:
            write_json(
                args.json_out,
                {
                    "args": vars(args),
                    "arch_source": arch.source,
                    "chunk_source": chunk_cfg.source,
                    "summaries": summaries,
                },
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

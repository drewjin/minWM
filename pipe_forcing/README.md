# Pipe Forcing

This directory contains the isolated Wan DMD pipe-forcing experiment. It does
not modify the existing naive Wan inference entrypoint.

## Run

```bash
bash pipe_forcing/run_wan_dmd_pipeline.sh
```

Useful overrides:

```bash
MAX_PROMPTS=1 NUM_OUTPUT_FRAMES=20 bash pipe_forcing/run_wan_dmd_pipeline.sh
NUM_OUTPUT_FRAMES=80 OUTPUT_FOLDER=output/pipe_forcing/wan_dmd_camera_80 bash pipe_forcing/run_wan_dmd_pipeline.sh
OVERWRITE=1 MAX_PROMPTS=1 bash pipe_forcing/run_wan_dmd_pipeline.sh
```

Chunk-lane wavefront version matching the original idea:

```bash
MAX_PROMPTS=1 NUM_OUTPUT_FRAMES=20 bash pipe_forcing/run_wan_dmd_lane_pipeline.sh
```

This assigns `chunk_id % 4` to each GPU. Each rank owns full denoising for its
chunk lane, broadcasts every produced `x[c,s]` latent, and every rank writes the
received latent into its local static step cache through Wan's original
sliding-window KV update path.

## Baselines

Append baseline, single GPU, no inter-rank latent transfer:

```bash
MAX_PROMPTS=1 NUM_OUTPUT_FRAMES=20 bash pipe_forcing/run_wan_dmd_append_baseline.sh
```

Append baseline with four-GPU hardware control:

```bash
MAX_PROMPTS=1 NUM_OUTPUT_FRAMES=20 bash pipe_forcing/run_wan_dmd_append_tp4_baseline.sh
```

Note: Wan in this repo exposes sequence parallelism for this inference path.
The `tp4` script name matches the experiment label, but internally it uses
`sp_size=4`.

Naive single-chunk generation:

```bash
MAX_PROMPTS=1 bash pipe_forcing/run_wan_dmd_single_chunk_baseline.sh
```

## Design

- `rank0 / gpu0`: DMD step 1
- `rank1 / gpu1`: DMD step 2
- `rank2 / gpu2`: DMD step 3
- `rank3 / gpu3`: DMD step 4 and VAE decode

Every rank loads one Wan generator and keeps its own historical causal cache.
Only the current chunk latent is transferred between ranks. The cache is
updated from that rank's early x0 prediction, so this is a heuristic throughput
experiment rather than an exact reproduction of naive serial inference.

The lane pipeline uses a different schedule: each GPU owns chunks instead of
steps, while all ranks maintain four static step caches locally.

## Limits

V1 intentionally supports only Wan causal few-step DMD with four denoising
steps, `num_frame_per_block=4`, and T2V camera inference. HY15, I2V,
bidirectional inference, 48/50-step diffusion inference, and CFG branches are
out of scope for these entrypoints. The four-rank pipeline itself is `sp_size=1`;
the append baseline also has an `sp_size=4` hardware-control script.

## Baseline

Compare against the existing naive Wan command:

```bash
bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

Use the same seed, prompt file, trajectory, checkpoint, and frame count when
comparing latency or output quality.

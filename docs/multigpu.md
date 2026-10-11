# Multi-GPU training: DDP, FSDP2 and activation checkpointing

`train.py` stays a single file. A few environment variables, read only when set, turn it into a
`torchrun` program; with none of them set it behaves exactly as before.

| Variable | Effect |
|---|---|
| `AR_DIST=ddp` | DistributedDataParallel: a full model replica per rank, gradients all-reduced during backward (bucketed, overlapped with compute). Scales throughput, not model size. |
| `AR_DIST=fsdp` | FSDP2 (`fully_shard`): parameters, gradients and optimizer state sharded across ranks, one group per transformer Block plus the root; each Block's weights are all-gathered just before it runs and freed after; gradients reduce-scattered. Memory per rank falls toward 1/N at the cost of communication. Mixed precision: bf16 parameters, fp32 reductions. |
| `AR_ACT_CKPT=1` | Non-reentrant activation checkpointing on every Block: activations are recomputed in backward instead of stored. Composes with both parallelisms and with single-GPU runs. |
| `AR_COMPILE=0/1` | Force `torch.compile` off/on. Default: on for CUDA, off for the FSDP arm (FSDP2 + compile is a follow-up). |
| `AR_TIME_BUDGET`, `AR_DEPTH`, `AR_DEVICE_BATCH`, `AR_TOTAL_BATCH`, `AR_WARMUP_STEPS` | Overrides for the benchmark grid and smoke runs. |
| `AR_SMOKE=1` | Hermetic run on random tokens with a fake tokenizer (no download), to exercise every distributed path on a laptop CPU (gloo) or a short GPU job (NCCL). |
| `AR_TAG`, `AR_RESULTS_DIR` | Name and folder of the results JSON written by rank 0. |
| `AR_ATTN=sdpa` | Skip the FlashAttention-3 kernel load and use PyTorch SDPA (the Explorer grid runs this way; the JSON records the backend actually used). |

Design points worth being able to explain:

- **Strong scaling.** The global batch stays at `TOTAL_BATCH_SIZE` tokens per optimizer step; the
  per-rank micro-batch halves until `ranks x micro-batch` divides it, so 1 GPU runs 128 x 2 (accumulation),
  2 GPUs 128 x 1, 4 GPUs 64 x 1. Tokens/s is therefore directly comparable across rows.

On 80 GB A100s the eager, non-checkpointed rows OOM at the default 128 x 2048 micro-batch (first grid,
2026-10-11: 72.7 GB in use when an 8 GiB allocation failed; the compiled single-GPU run peaked at 44 GB and
the checkpointed 2-GPU rows at 38 GB), so `grid.sbatch` sets `AR_DEVICE_BATCH=64`: 1 GPU 64 x 4, 2 GPUs 64 x 2,
same 524K global batch. The three 128-batch rows that did run are kept in `results/multigpu_b128_partial/`.
- **Data sharding.** `prepare.make_dataloader(..., rank, world_size)` strides parquet row groups by rank,
  so data-parallel ranks read disjoint documents. Validation is not sharded: every rank evaluates the same
  stream, which the FSDP forward needs anyway (it is a collective).
- **Gradient accumulation under data parallelism.** The gradient all-reduce (DDP `no_sync`) or
  reduce-scatter (FSDP2 `set_requires_gradient_sync`) runs only on the last micro-step.
- **Stopping together.** The time budget is checked per rank; the stop and fail flags are combined with an
  `all_reduce(MAX)` every step so no rank leaves a collective early.
- **Muon and FSDP do not mix out of the box.** Muon's Newton-Schulz orthogonalisation needs each whole
  gradient matrix, and FSDP shards it. The FSDP arm therefore trains every group with AdamW (fused on CUDA),
  so its `val_bpb` is not comparable with the Muon arms; tokens/s and memory are. Gathering Muon's matrices
  per step (or a Muon-aware sharding) is the natural follow-up.
- **Peak memory** is reported for rank 0 and as the max over ranks (`all_reduce(MAX)`).

## Running on Explorer

```bash
# laptop: CPU smoke of every path with 2 processes (gloo)
AR_SMOKE=1 AR_DIST=fsdp AR_ACT_CKPT=1 AR_TIME_BUDGET=2 AR_WARMUP_STEPS=1 AR_DEPTH=2 AR_SEQ_LEN=128 \
  AR_DEVICE_BATCH=2 AR_TOTAL_BATCH=1024 AR_COMPILE=0 torchrun --standalone --nproc_per_node=2 train.py

# cluster: env -> data -> 2-GPU NCCL smoke -> 2-GPU grid on gpu-short (dependencies handled)
bash scripts/multigpu/submit_chain.sh
```

The grid (`scripts/multigpu/grid.sbatch`) runs six 2-minute configurations on one node: single GPU with and
without compile, then DDP, DDP + checkpointing, FSDP2 and FSDP2 + checkpointing on 2 GPUs, and writes
`docs/multigpu_results.md`. On Explorer the `gpu` partition's QOS caps every job at one GPU and the `multigpu`
partition is closed to this account, so the grid runs on `gpu-short` with two A100s (about 25 GPU-minutes).
Set `AR_MAX_WS=4` on a cluster that gives one job four GPUs: the ws4 rows (DDP 4, DDP 4 + checkpointing,
FSDP2 4, FSDP2 4 + checkpointing) then replace the 2-GPU checkpointing rows, seven runs in all.

## Results

See `docs/multigpu_results.md` once the grid has run. Raw JSON per run lives in `results/multigpu/`.

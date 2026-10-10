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

Design points worth being able to explain:

- **Strong scaling.** The global batch stays at `TOTAL_BATCH_SIZE` tokens per optimizer step; the
  per-rank micro-batch halves until `ranks x micro-batch` divides it, so 1 GPU runs 128 x 2 (accumulation),
  2 GPUs 128 x 1, 4 GPUs 64 x 1. Tokens/s is therefore directly comparable across rows.
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

# cluster: env -> data -> 2-GPU NCCL smoke -> 4-GPU grid (dependencies handled)
bash scripts/multigpu/submit_chain.sh
```

The grid (`scripts/multigpu/grid.sbatch`) runs seven 2-minute configurations on one node with four H200s:
single GPU with and without compile, DDP on 2 and 4 GPUs, DDP + checkpointing, FSDP2, FSDP2 + checkpointing,
then writes `docs/multigpu_results.md`. Expect about 1.5 H200-hours in total.

## Results

See `docs/multigpu_results.md` once the grid has run. Raw JSON per run lives in `results/multigpu/`.

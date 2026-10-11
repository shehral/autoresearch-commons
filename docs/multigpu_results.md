# Multi-GPU results (autoresearch-commons, strong scaling at a fixed global batch)

GPU: NVIDIA A100-SXM4-80GB | torch 2.11.0+cu128 | depth 8 (50.3M params) | global batch 524,288 tokens | budget 120 s per run

| run | GPUs | parallelism | act. ckpt | compile | per-rank batch x accum | tokens/s | speedup vs 1 GPU (no compile) | peak GB (max rank) | val_bpb |
|---|---|---|---|---|---|---|---|---|---|
| ddp_ws2 | 2 | ddp | no | no | 64 x 2 | 482,424 | 1.97x | 44.7 | 1.3046 |
| ddp_ws2_ckpt | 2 | ddp | yes | no | 64 x 2 | 392,881 | 1.61x | 19.6 | 1.3942 |
| fsdp_ws2 | 2 | fsdp | no | no | 64 x 2 | 487,801 | 2.00x | 44.4 | 1.6543 |
| fsdp_ws2_ckpt | 2 | fsdp | yes | no | 64 x 2 | 395,565 | 1.62x | 19.3 | 1.7476 |
| single_compile0 | 1 | none | no | no | 64 x 4 | 244,424 | 1.00x | 44.6 | 1.5910 |
| single_compile1 | 1 | none | no | yes | 64 x 4 | 389,410 | 1.59x | 22.3 | 1.3723 |

val_bpb is comparable only between runs with the same optimizer: the FSDP arm uses AdamW for every group (Muon needs whole gradient matrices, which FSDP shards), the others use Muon + AdamW. Tokens/s and memory are comparable across all rows.

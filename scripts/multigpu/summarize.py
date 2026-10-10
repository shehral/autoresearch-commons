"""Turn results/multigpu/*.json into a markdown table (docs/multigpu_results.md)."""
import glob, json, os, sys

src = sys.argv[1] if len(sys.argv) > 1 else "results/multigpu"
dst = sys.argv[2] if len(sys.argv) > 2 else "docs/multigpu_results.md"
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(src, "*.json")))]
if not rows:
    sys.exit(f"no results in {src}")
base = next((r for r in rows if r["dist"] == "none" and not r["compile"]), rows[0])
lines = ["# Multi-GPU results (autoresearch-commons, strong scaling at a fixed global batch)", "",
         f"GPU: {base.get('gpu')} | torch {base.get('torch')} | depth {base['depth']} ({base['num_params']/1e6:.1f}M params) | "
         f"global batch {base['total_batch_tokens']:,} tokens | budget {base['time_budget_s']:.0f} s per run", "",
         "| run | GPUs | parallelism | act. ckpt | compile | per-rank batch x accum | tokens/s | speedup vs 1 GPU (no compile) | peak GB (max rank) | val_bpb |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    sp = r["steady_tok_per_sec"] / base["steady_tok_per_sec"] if base["steady_tok_per_sec"] else float("nan")
    lines.append(f"| {r['tag']} | {r['world_size']} | {r['dist']} | {'yes' if r['act_ckpt'] else 'no'} | {'yes' if r['compile'] else 'no'} | "
                 f"{r['device_batch']} x {r['grad_accum_steps']} | {r['steady_tok_per_sec']:,.0f} | {sp:.2f}x | {r['peak_vram_mb_max']/1024:.1f} | {r['val_bpb']:.4f} |")
lines += ["", "val_bpb is comparable only between runs with the same optimizer: the FSDP arm uses AdamW for every group "
          "(Muon needs whole gradient matrices, which FSDP shards), the others use Muon + AdamW. Tokens/s and memory are comparable across all rows.", ""]
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
open(dst, "w").write("\n".join(lines))
print(f"wrote {dst} ({len(rows)} runs)")

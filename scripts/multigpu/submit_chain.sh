#!/bin/bash
# Submit env -> data -> smoke -> grid with afterok dependencies. Run on the Explorer login node:
#   bash scripts/multigpu/submit_chain.sh            # full chain
#   bash scripts/multigpu/submit_chain.sh rerun      # smoke -> grid (env + data already there)
#   bash scripts/multigpu/submit_chain.sh race       # same, but each job queued on A100s AND H200s; first to start wins
#   bash scripts/multigpu/submit_chain.sh grid       # just the grid
# GPU jobs exclude the node in AR_EXCLUDE (default d4055, flagged sick on 2026-10-07).
set -eo pipefail
EXC="--exclude=${AR_EXCLUDE:-d4055}"
cd "$(dirname "$0")/../.." && REPO=$(pwd)
S=scripts/multigpu
case "${1:-all}" in
  all)
    E=$(sbatch --parsable "$S/env_build.sbatch"); echo "env     $E"
    P=$(sbatch --parsable --dependency=afterok:$E "$S/prepare_data.sbatch"); echo "prepare $P"
    K=$(sbatch --parsable $EXC --dependency=afterok:$E "$S/smoke.sbatch"); echo "smoke   $K"
    G=$(sbatch --parsable $EXC --dependency=afterok:$P:$K "$S/grid.sbatch"); echo "grid    $G" ;;
  rerun)
    K=$(sbatch --parsable $EXC "$S/smoke.sbatch"); echo "smoke   $K"
    G=$(sbatch --parsable $EXC --dependency=afterok:$K "$S/grid.sbatch"); echo "grid    $G" ;;
  race)
    # Queue each GPU job twice, on A100s and on H200s; the first to start wins (lock in the scripts).
    LK=$(date +%s)
    S1=$(sbatch --parsable --gres=gpu:a100:2 --export=ALL,AR_LOCK=smoke-$LK "$S/smoke.sbatch"); echo "smoke a100 $S1"
    S2=$(sbatch --parsable --gres=gpu:h200:2 $EXC --export=ALL,AR_LOCK=smoke-$LK "$S/smoke.sbatch"); echo "smoke h200 $S2"
    DEP="--dependency=afterok:$S1?afterok:$S2"
    G1=$(sbatch --parsable --gres=gpu:a100:4 --cpus-per-task=16 --mem=120G $DEP --export=ALL,AR_LOCK=grid-$LK "$S/grid.sbatch"); echo "grid  a100 $G1"
    G2=$(sbatch --parsable --gres=gpu:h200:4 $EXC $DEP --export=ALL,AR_LOCK=grid-$LK "$S/grid.sbatch"); echo "grid  h200 $G2" ;;
  smoke) sbatch $EXC "$S/smoke.sbatch" ;;
  grid)  sbatch $EXC "$S/grid.sbatch" ;;
esac
squeue -u "$USER" -o "%.10i %.11j %.9T %.10M %.6D %R"

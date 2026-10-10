#!/bin/bash
# Submit env -> data -> smoke -> grid with afterok dependencies. Run on the Explorer login node:
#   bash scripts/multigpu/submit_chain.sh            # full chain
#   bash scripts/multigpu/submit_chain.sh grid       # just the grid (env + data already there)
set -eo pipefail
cd "$(dirname "$0")/../.." && REPO=$(pwd)
S=scripts/multigpu
case "${1:-all}" in
  all)
    E=$(sbatch --parsable "$S/env_build.sbatch"); echo "env     $E"
    P=$(sbatch --parsable --dependency=afterok:$E "$S/prepare_data.sbatch"); echo "prepare $P"
    K=$(sbatch --parsable --dependency=afterok:$E "$S/smoke.sbatch"); echo "smoke   $K"
    G=$(sbatch --parsable --dependency=afterok:$P:$K "$S/grid.sbatch"); echo "grid    $G" ;;
  smoke) sbatch "$S/smoke.sbatch" ;;
  grid)  sbatch "$S/grid.sbatch" ;;
esac
squeue -u "$USER" -o "%.10i %.11j %.9T %.10M %.6D %R"

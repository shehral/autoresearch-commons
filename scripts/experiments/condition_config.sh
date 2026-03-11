#!/usr/bin/env bash
# Shared condition configuration for experiment scripts.
# Bash 3.2 compatible (no associative arrays).
#
# Usage: source this file, then call get_config <condition>
# Sets: NUM_AGENTS, NUM_EXPERIMENTS, IS_SHARED, HAS_DIRECTOR

get_config() {
    local cond="$1"
    case "$cond" in
        condA)           NUM_AGENTS=1; NUM_EXPERIMENTS=100; IS_SHARED=false; HAS_DIRECTOR=false ;;
        condB)           NUM_AGENTS=1; NUM_EXPERIMENTS=100; IS_SHARED=false; HAS_DIRECTOR=false ;;
        condC)           NUM_AGENTS=2; NUM_EXPERIMENTS=50;  IS_SHARED=false; HAS_DIRECTOR=false ;;
        condD)           NUM_AGENTS=2; NUM_EXPERIMENTS=50;  IS_SHARED=true;  HAS_DIRECTOR=false ;;
        condE)           NUM_AGENTS=2; NUM_EXPERIMENTS=50;  IS_SHARED=true;  HAS_DIRECTOR=true  ;;
        condF)           NUM_AGENTS=4; NUM_EXPERIMENTS=25;  IS_SHARED=false; HAS_DIRECTOR=false ;;
        condG)           NUM_AGENTS=4; NUM_EXPERIMENTS=25;  IS_SHARED=true;  HAS_DIRECTOR=true  ;;
        ablation_G-cov)  NUM_AGENTS=4; NUM_EXPERIMENTS=25;  IS_SHARED=true;  HAS_DIRECTOR=true  ;;
        ablation_G-lin)  NUM_AGENTS=4; NUM_EXPERIMENTS=25;  IS_SHARED=true;  HAS_DIRECTOR=true  ;;
        ablation_G-dir)  NUM_AGENTS=4; NUM_EXPERIMENTS=25;  IS_SHARED=true;  HAS_DIRECTOR=false ;;
        ablation_G-syn)  NUM_AGENTS=4; NUM_EXPERIMENTS=25;  IS_SHARED=true;  HAS_DIRECTOR=true  ;;
        ablation_G-struct) NUM_AGENTS=4; NUM_EXPERIMENTS=25; IS_SHARED=true; HAS_DIRECTOR=true  ;;
        *)
            echo "ERROR: Unknown condition '$cond'" >&2
            echo "Valid: condA-G, ablation_G-{cov,lin,dir,syn,struct}" >&2
            return 1
            ;;
    esac
}

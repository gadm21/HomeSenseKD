#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Publication experiment runner
#
# Phase 1: all 4 datasets x 7 algorithms (uniform heterogeneity)
# Phase 2: home_occupancy x 7 algorithms x 3 heterogeneity distributions
#
# Usage:
#   bash run_experiments.sh               # both phases, 4 parallel jobs
#   bash run_experiments.sh --jobs 8      # more parallelism
#   bash run_experiments.sh --phase 1     # main comparison only
#   bash run_experiments.sh --phase 2     # heterogeneity ablation only
#   bash run_experiments.sh --dry-run     # print commands without running
# =============================================================================

set -euo pipefail

JOBS=4
PHASE="all"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)    JOBS="$2";  shift 2 ;;
        --phase)   PHASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "[warn] Unknown arg: $1"; shift ;;
    esac
done

# ── Locate Python in venv ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv"
LOG_DIR="$SCRIPT_DIR/logs"
RUNNER="$SCRIPT_DIR/run_fedkd.py"

if   [[ -f "$VENV/Scripts/python.exe" ]]; then PYTHON="$VENV/Scripts/python.exe"
elif [[ -f "$VENV/Scripts/python"     ]]; then PYTHON="$VENV/Scripts/python"
elif [[ -f "$VENV/bin/python"         ]]; then PYTHON="$VENV/bin/python"
else echo "[error] venv not found at $VENV"; exit 1; fi

export PYTHONIOENCODING=utf-8

# ── Phase-scoped directories (set by run_phase1/run_phase2 before run_pool) ────
# These globals are read by run_job so the pool interface stays unchanged.
PHASE_LOG_DIR=""        # per-job logs land here
PHASE_MASTER=""         # phase-level master log
PHASE_RESULTS_DIR=""    # --override experiment.results_dir
PHASE_FIG_DIR=""        # --override experiment.fig_dir

RUN_TS="$(date '+%Y%m%d_%H%M%S')"

# Top-level master log (spans all phases)
GLOBAL_MASTER="$LOG_DIR/master_${RUN_TS}.log"
mkdir -p "$LOG_DIR"

# ── Logging helpers ────────────────────────────────────────────────────────────

# Prepend [YYYY-MM-DD HH:MM:SS] to every line; flush immediately
log_ts() {
    awk '{ printf "[%s] %s\n", strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }'
}

# ── System info header ─────────────────────────────────────────────────────────
print_sys_info() {
    local log="$1" label="$2"
    {
        echo "============================================================"
        echo "  $label"
        echo "  Run started : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Host        : $(hostname 2>/dev/null || echo N/A)"
        echo "  OS          : $(uname -a 2>/dev/null || echo N/A)"
        if   command -v nproc   &>/dev/null; then _cpu="$(nproc) logical cores"
        elif command -v sysctl  &>/dev/null; then _cpu="$(sysctl -n hw.logicalcpu 2>/dev/null) logical cores"
        else _cpu="N/A"; fi
        echo "  CPU         : $_cpu"
        if   command -v free    &>/dev/null; then _ram="$(free -h | awk '/^Mem:/{print $2}')"
        elif command -v vm_stat &>/dev/null; then _ram="$(vm_stat | awk '/Pages free:/{printf "%.1f GB\n", $3*4096/1073741824}')"
        else _ram="N/A"; fi
        echo "  RAM         : $_ram"
        if command -v nvidia-smi &>/dev/null; then
            _gpu="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
        else
            _gpu="N/A (nvidia-smi not found)"
        fi
        echo "  GPU         : $_gpu"
        echo "  Python      : $("$PYTHON" --version 2>&1)"
        echo "  Phase       : $PHASE   Jobs: $JOBS   Dry-run: $DRY_RUN"
        echo "  Log dir     : $(dirname "$log")"
        echo "  Results dir : $PHASE_RESULTS_DIR"
        echo "  Master log  : $log"
        echo "============================================================"
    } | log_ts | tee -a "$log"
}

# ── Job runner ─────────────────────────────────────────────────────────────────
# Reads phase-scoped globals: PHASE_LOG_DIR, PHASE_MASTER, PHASE_RESULTS_DIR, PHASE_FIG_DIR
run_job() {
    local dataset="$1" algo="$2" hetero="$3"
    local tag="${dataset}_${algo}_${hetero}"
    local log="$PHASE_LOG_DIR/${tag}.log"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] $algo  dataset=$dataset  hetero=$hetero  results=$PHASE_RESULTS_DIR" \
            | log_ts | tee -a "$PHASE_MASTER" | tee -a "$GLOBAL_MASTER"
        return
    fi

    {
        echo "============================================================"
        echo "  JOB START   : $tag"
        echo "  PID         : $BASHPID"
        echo "  Log         : $log"
        echo "  Results dir : $PHASE_RESULTS_DIR"
        echo "  Fig dir     : $PHASE_FIG_DIR"
        echo "  Cmd         : $PYTHON $RUNNER \\"
        echo "                  --config $SCRIPT_DIR/config/${dataset}.yaml \\"
        echo "                  --algorithm $algo --heterogeneity $hetero \\"
        echo "                  --override experiment.results_dir=$PHASE_RESULTS_DIR \\"
        echo "                             experiment.fig_dir=$PHASE_FIG_DIR"
        echo "============================================================"
    } | log_ts | tee -a "$log" | tee -a "$PHASE_MASTER" | tee -a "$GLOBAL_MASTER"

    local exit_code=0
    "$PYTHON" "$RUNNER" \
        --config "$SCRIPT_DIR/config/${dataset}.yaml" \
        --algorithm "$algo" \
        --heterogeneity "$hetero" \
        --override \
            "experiment.results_dir=$PHASE_RESULTS_DIR" \
            "experiment.fig_dir=$PHASE_FIG_DIR" \
        2>&1 \
      | log_ts \
      | tee -a "$log" \
      | tee -a "$PHASE_MASTER" \
      | tee -a "$GLOBAL_MASTER" \
      || exit_code=$?

    {
        echo "============================================================"
        if [[ $exit_code -eq 0 ]]; then
            echo "  JOB DONE    : $tag  (exit 0)"
        else
            echo "  JOB FAILED  : $tag  (exit $exit_code)"
        fi
        echo "  Full log    : $log"
        echo "============================================================"
    } | log_ts | tee -a "$log" | tee -a "$PHASE_MASTER" | tee -a "$GLOBAL_MASTER"
}

# ── Pool scheduler ─────────────────────────────────────────────────────────────
run_pool() {
    local running=0
    for args in "$@"; do
        IFS=',' read -r ds algo hetero <<< "$args"
        run_job "$ds" "$algo" "$hetero" &
        running=$((running + 1))
        if [[ $running -ge $JOBS ]]; then
            wait -n 2>/dev/null || wait
            running=$((running - 1))
        fi
    done
    wait
}

DATASETS=(home_occupancy home_har mnist cifar10)
ALGORITHMS=(fedmd fedakd mks fedavg fedprox local central)
HETEROS=(all_small uniform skewed)

run_phase1() {
    PHASE_LOG_DIR="$LOG_DIR/phase1"
    PHASE_MASTER="$PHASE_LOG_DIR/master_${RUN_TS}.log"
    PHASE_RESULTS_DIR="$SCRIPT_DIR/results/phase1"
    PHASE_FIG_DIR="$SCRIPT_DIR/results/phase1/figures"
    mkdir -p "$PHASE_LOG_DIR" "$PHASE_RESULTS_DIR" "$PHASE_FIG_DIR"

    print_sys_info "$PHASE_MASTER" "Phase 1 — Main comparison (heterogeneity=uniform)"
    {
        echo "  Scope : ${#DATASETS[@]} datasets x ${#ALGORITHMS[@]} algorithms"
    } | log_ts | tee -a "$PHASE_MASTER" | tee -a "$GLOBAL_MASTER"

    local jobs=()
    for ds in "${DATASETS[@]}"; do
        for algo in "${ALGORITHMS[@]}"; do
            jobs+=("$ds,$algo,uniform")
        done
    done
    run_pool "${jobs[@]}"

    {
        echo "============================================================"
        echo "  Phase 1 complete."
        echo "  Logs    -> $PHASE_LOG_DIR"
        echo "  Results -> $PHASE_RESULTS_DIR"
        echo "  Figures -> $PHASE_FIG_DIR"
        echo "  Master  -> $PHASE_MASTER"
        echo "============================================================"
    } | log_ts | tee -a "$PHASE_MASTER" | tee -a "$GLOBAL_MASTER"
}

run_phase2() {
    PHASE_LOG_DIR="$LOG_DIR/phase2"
    PHASE_MASTER="$PHASE_LOG_DIR/master_${RUN_TS}.log"
    PHASE_RESULTS_DIR="$SCRIPT_DIR/results/phase2"
    PHASE_FIG_DIR="$SCRIPT_DIR/results/phase2/figures"
    mkdir -p "$PHASE_LOG_DIR" "$PHASE_RESULTS_DIR" "$PHASE_FIG_DIR"

    print_sys_info "$PHASE_MASTER" "Phase 2 — Heterogeneity ablation (dataset=home_occupancy)"
    {
        echo "  Scope : ${#ALGORITHMS[@]} algorithms x ${#HETEROS[@]} heterogeneity distributions"
    } | log_ts | tee -a "$PHASE_MASTER" | tee -a "$GLOBAL_MASTER"

    local jobs=()
    for algo in "${ALGORITHMS[@]}"; do
        for hetero in "${HETEROS[@]}"; do
            jobs+=("home_occupancy,$algo,$hetero")
        done
    done
    run_pool "${jobs[@]}"

    {
        echo "============================================================"
        echo "  Phase 2 complete."
        echo "  Logs    -> $PHASE_LOG_DIR"
        echo "  Results -> $PHASE_RESULTS_DIR"
        echo "  Figures -> $PHASE_FIG_DIR"
        echo "  Master  -> $PHASE_MASTER"
        echo "============================================================"
    } | log_ts | tee -a "$PHASE_MASTER" | tee -a "$GLOBAL_MASTER"
}

print_sys_info "$GLOBAL_MASTER" "Global run — all phases"

case "$PHASE" in
    1|phase1) run_phase1 ;;
    2|phase2) run_phase2 ;;
    all)      run_phase1; run_phase2 ;;
    *) echo "[error] Unknown --phase '$PHASE'  (use 1, 2, or all)"; exit 1 ;;
esac

{
    echo "============================================================"
    echo "  All experiments complete."
    echo "  Phase 1 logs    -> $LOG_DIR/phase1/"
    echo "  Phase 2 logs    -> $LOG_DIR/phase2/"
    echo "  Phase 1 results -> $SCRIPT_DIR/results/phase1/"
    echo "  Phase 2 results -> $SCRIPT_DIR/results/phase2/"
    echo "  Global master   -> $GLOBAL_MASTER"
    echo "============================================================"
} | log_ts | tee -a "$GLOBAL_MASTER"

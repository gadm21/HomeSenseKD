#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Publication experiment runner
#
# Phase 1: all 4 datasets x 7 algorithms (uniform heterogeneity)
# Phase 2: home_occupancy x 7 algorithms x 3 heterogeneity distributions
# Phase 3: FedMKS vs FedDF on all datasets (accuracy + compute + communication)
#
# Usage:
#   bash run_experiments.sh               # all phases, 4 parallel jobs
#   bash run_experiments.sh --jobs 8      # more parallelism
#   bash run_experiments.sh --phase 1     # main comparison only
#   bash run_experiments.sh --phase 2     # heterogeneity ablation only
#   bash run_experiments.sh --phase 3     # FedMKS vs FedDF efficiency comparison
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
VENV="$SCRIPT_DIR/myenv"
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
PHASE_EXTRA_OVERRIDES=()  # extra KEY=VAL pairs appended to --override

RUN_TS="$(date '+%Y%m%d_%H%M%S')"

# Top-level master log (spans all phases)
GLOBAL_MASTER="$LOG_DIR/master_${RUN_TS}.log"
mkdir -p "$LOG_DIR" 2>/dev/null || true

# Track which phases failed (for parallel 'all' and final exit code)
PHASE_FAILURES=()

# Early fallback if we cannot write the intended global master (e.g. root-owned logs dir)
if ! ( : >> "$GLOBAL_MASTER" ) 2>/dev/null; then
    FALLBACK_GLOBAL="/tmp/homesensekd_master_${RUN_TS}.log"
    printf "[%s] WARNING: cannot write to %s, falling back to %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$GLOBAL_MASTER" "$FALLBACK_GLOBAL" >&2
    GLOBAL_MASTER="$FALLBACK_GLOBAL"
    : > "$GLOBAL_MASTER" 2>/dev/null || true
fi

# Helper to log events even during traps (before/after full setup)
log_event() {
    local msg="$1"
    # Always emit to stderr so user sees it even if file write fails
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" >&2
    if [[ -n "${GLOBAL_MASTER:-}" ]]; then
        printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" >> "$GLOBAL_MASTER" 2>/dev/null || true
    fi
}

# Capture errors and signals that stop the script
on_err() {
    local code=$?
    local cmd=${BASH_COMMAND:-?}
    log_event "ERROR: command failed (exit=$code): $cmd (line ${BASH_LINENO[0]:-?})"
}
trap on_err ERR

on_term() {
    log_event "TERMINATED by signal"
    exit 143
}
trap on_term INT TERM

cleanup_on_exit() {
    local code=$?
    if [[ -n "${GLOBAL_MASTER:-}" ]]; then
        printf "[%s] SCRIPT EXIT (code=%s)\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$code" >> "$GLOBAL_MASTER" 2>/dev/null || true
    fi
}
trap cleanup_on_exit EXIT

# ── Logging helpers ────────────────────────────────────────────────────────────

# Prepend [YYYY-MM-DD HH:MM:SS] to every line; flush immediately
log_ts() {
    awk '{ printf "[%s] %s\n", strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }'
}

# Tolerate write failures (e.g. permission on logs dir) by falling back to passthrough
safe_tee() {
    local f="$1"
    tee -a "$f" 2>/dev/null || cat
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
        _gpu="N/A"
        if command -v nvidia-smi &>/dev/null; then
            if _out=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1); then
                _gpu="$_out"
            else
                _gpu="N/A (nvidia-smi failed to query)"
            fi
        fi
        echo "  GPU         : $_gpu"
        py_ver="$("$PYTHON" --version 2>&1 || echo 'unavailable')"
        echo "  Python      : $py_ver"
        echo "  Phase       : $PHASE   Jobs: $JOBS   Dry-run: $DRY_RUN"
        echo "  Log dir     : $(dirname "$log")"
        echo "  Results dir : $PHASE_RESULTS_DIR"
        echo "  Master log  : $log"
        echo "============================================================"
    } 2>&1 | log_ts | safe_tee "$log"
}

# ── Job runner ─────────────────────────────────────────────────────────────────
# Reads phase-scoped globals: PHASE_LOG_DIR, PHASE_MASTER, PHASE_RESULTS_DIR, PHASE_FIG_DIR
run_job() {
    local dataset="$1" algo="$2" hetero="$3"
    local tag="${dataset}_${algo}_${hetero}"
    local log="$PHASE_LOG_DIR/${tag}.log"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] $algo  dataset=$dataset  hetero=$hetero  results=$PHASE_RESULTS_DIR" \
            | log_ts | safe_tee "$PHASE_MASTER" | safe_tee "$GLOBAL_MASTER"
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
    } | log_ts | safe_tee "$log" | safe_tee "$PHASE_MASTER" | safe_tee "$GLOBAL_MASTER"

    local exit_code=0
    "$PYTHON" "$RUNNER" \
        --config "$SCRIPT_DIR/config/${dataset}.yaml" \
        --algorithm "$algo" \
        --heterogeneity "$hetero" \
        --override \
            "experiment.results_dir=$PHASE_RESULTS_DIR" \
            "experiment.fig_dir=$PHASE_FIG_DIR" \
            "${PHASE_EXTRA_OVERRIDES[@]}" \
        2>&1 \
      | log_ts \
      | safe_tee "$log" \
      | safe_tee "$PHASE_MASTER" \
      | safe_tee "$GLOBAL_MASTER" || true
    exit_code=${PIPESTATUS[0]:-0}

    {
        echo "============================================================"
        if [[ $exit_code -eq 0 ]]; then
            echo "  JOB DONE    : $tag  (exit 0)"
        else
            echo "  JOB FAILED  : $tag  (exit $exit_code)"
        fi
        echo "  Full log    : $log"
        echo "============================================================"
    } | log_ts | safe_tee "$log" | safe_tee "$PHASE_MASTER" | safe_tee "$GLOBAL_MASTER"
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

# ── Phase failure detection ────────────────────────────────────────────────────
# Returns 0 (true) if the phase master log contains any "JOB FAILED" entries.
phase_had_job_failures() {
    local master="$1"
    [[ -f "$master" ]] && grep -q "JOB FAILED" "$master"
}

# Make phase log/master dirs resilient. Sets up fallbacks for PHASE_MASTER and PHASE_LOG_DIR.
setup_phase_logging() {
    mkdir -p "$PHASE_LOG_DIR" "$PHASE_RESULTS_DIR" "$PHASE_FIG_DIR" 2>/dev/null || true

    # Fallback for PHASE_MASTER if not writable
    if ! ( : >> "$PHASE_MASTER" ) 2>/dev/null; then
        local fb="/tmp/homesensekd_$(basename "$PHASE_LOG_DIR")_master_${RUN_TS}.log"
        log_event "WARNING: cannot write phase master $PHASE_MASTER, falling back to $fb"
        PHASE_MASTER="$fb"
        : > "$PHASE_MASTER" 2>/dev/null || true
    fi

    # Fallback for PHASE_LOG_DIR (per-job logs) if not writable
    local testf="$PHASE_LOG_DIR/.wtest_${RUN_TS}"
    if ! ( : > "$testf" ) 2>/dev/null; then
        local fbdir="/tmp/homesensekd_$(basename "$PHASE_LOG_DIR")_${RUN_TS}"
        log_event "WARNING: cannot write phase logs to $PHASE_LOG_DIR, using $fbdir"
        mkdir -p "$fbdir" 2>/dev/null || true
        PHASE_LOG_DIR="$fbdir"
    else
        rm -f "$testf" 2>/dev/null || true
    fi
}

# DATASETS=(home_occupancy home_har mnist cifar10)
DATASETS=(home_har mnist cifar10)
ALGORITHMS=(fedmd fedakd mks fedavg fedprox local central)
HETEROS=(all_small uniform skewed)

run_phase1() {
    PHASE_EXTRA_OVERRIDES=()
    PHASE_LOG_DIR="$LOG_DIR/phase1"
    PHASE_MASTER="$PHASE_LOG_DIR/master_${RUN_TS}.log"
    PHASE_RESULTS_DIR="$SCRIPT_DIR/results/phase1"
    PHASE_FIG_DIR="$SCRIPT_DIR/results/phase1/figures"
    setup_phase_logging

    print_sys_info "$PHASE_MASTER" "Phase 1 — Main comparison (heterogeneity=uniform)"
    {
        echo "  Scope : ${#DATASETS[@]} datasets x ${#ALGORITHMS[@]} algorithms"
    } | log_ts | safe_tee "$PHASE_MASTER" | safe_tee "$GLOBAL_MASTER"

    local jobs=()
    for ds in "${DATASETS[@]}"; do
        for algo in "${ALGORITHMS[@]}"; do
            jobs+=("$ds,$algo,uniform")
        done
    done
    run_pool "${jobs[@]}"

    local _phase_status=0
    if phase_had_job_failures "$PHASE_MASTER"; then
        _phase_status=1
    fi

    {
        echo "============================================================"
        if [[ $_phase_status -ne 0 ]]; then
            echo "  Phase 1 complete (with JOB FAILURES)."
        else
            echo "  Phase 1 complete."
        fi
        echo "  Logs    -> $PHASE_LOG_DIR"
        echo "  Results -> $PHASE_RESULTS_DIR"
        echo "  Figures -> $PHASE_FIG_DIR"
        echo "  Master  -> $PHASE_MASTER"
        echo "============================================================"
    } | log_ts | safe_tee "$PHASE_MASTER" | safe_tee "$GLOBAL_MASTER"

    return $_phase_status
}

run_phase2() {
    PHASE_EXTRA_OVERRIDES=()
    PHASE_LOG_DIR="$LOG_DIR/phase2"
    PHASE_MASTER="$PHASE_LOG_DIR/master_${RUN_TS}.log"
    PHASE_RESULTS_DIR="$SCRIPT_DIR/results/phase2"
    PHASE_FIG_DIR="$SCRIPT_DIR/results/phase2/figures"
    setup_phase_logging

    print_sys_info "$PHASE_MASTER" "Phase 2 — Heterogeneity ablation (dataset=home_occupancy)"
    {
        echo "  Scope : ${#ALGORITHMS[@]} algorithms x ${#HETEROS[@]} heterogeneity distributions"
    } | log_ts | safe_tee "$PHASE_MASTER" | safe_tee "$GLOBAL_MASTER"

    local jobs=()
    for algo in "${ALGORITHMS[@]}"; do
        for hetero in "${HETEROS[@]}"; do
            jobs+=("home_occupancy,$algo,$hetero")
        done
    done
    run_pool "${jobs[@]}"

    local _phase_status=0
    if phase_had_job_failures "$PHASE_MASTER"; then
        _phase_status=1
    fi

    {
        echo "============================================================"
        if [[ $_phase_status -ne 0 ]]; then
            echo "  Phase 2 complete (with JOB FAILURES)."
        else
            echo "  Phase 2 complete."
        fi
        echo "  Logs    -> $PHASE_LOG_DIR"
        echo "  Results -> $PHASE_RESULTS_DIR"
        echo "  Figures -> $PHASE_FIG_DIR"
        echo "  Master  -> $PHASE_MASTER"
        echo "============================================================"
    } | log_ts | safe_tee "$PHASE_MASTER" | safe_tee "$GLOBAL_MASTER"

    return $_phase_status
}

run_phase3() {
    # NOTE: algorithm key 'feddf' must be implemented in run_fedkd.py
    PHASE_EXTRA_OVERRIDES=("experiment.profile_compute=true")
    PHASE_LOG_DIR="$LOG_DIR/phase3"
    PHASE_MASTER="$PHASE_LOG_DIR/master_${RUN_TS}.log"
    PHASE_RESULTS_DIR="$SCRIPT_DIR/results/phase3"
    PHASE_FIG_DIR="$SCRIPT_DIR/results/phase3/figures"
    setup_phase_logging

    local P3_DATASETS=(home_occupancy home_har mnist cifar10)
    local P3_ALGORITHMS=(mks feddf)
    local P3_HETERO=uniform

    print_sys_info "$PHASE_MASTER" \
        "Phase 3 -- FedMKS vs FedDF (accuracy + server/client compute + communication)"
    {
        echo "  Scope : ${#P3_DATASETS[@]} datasets x ${#P3_ALGORITHMS[@]} algorithms (${P3_HETERO} heterogeneity)"
        echo "  Extra : ${PHASE_EXTRA_OVERRIDES[*]}"
    } | log_ts | safe_tee "$PHASE_MASTER" | safe_tee "$GLOBAL_MASTER"

    local jobs=()
    for ds in "${P3_DATASETS[@]}"; do
        for algo in "${P3_ALGORITHMS[@]}"; do
            jobs+=("$ds,$algo,$P3_HETERO")
        done
    done
    run_pool "${jobs[@]}"

    local _phase_status=0
    if phase_had_job_failures "$PHASE_MASTER"; then
        _phase_status=1
    fi

    {
        echo "============================================================"
        if [[ $_phase_status -ne 0 ]]; then
            echo "  Phase 3 complete (with JOB FAILURES)."
        else
            echo "  Phase 3 complete."
        fi
        echo "  Logs    -> $PHASE_LOG_DIR"
        echo "  Results -> $PHASE_RESULTS_DIR"
        echo "  Figures -> $PHASE_FIG_DIR"
        echo "  Master  -> $PHASE_MASTER"
        echo "============================================================"
    } | log_ts | safe_tee "$PHASE_MASTER" | safe_tee "$GLOBAL_MASTER"

    return $_phase_status
}

print_sys_info "$GLOBAL_MASTER" "Global run — all phases"

run_all_phases_parallel() {
    # Launch phases concurrently
    run_phase1 & p1=$!
    run_phase2 & p2=$!
    run_phase3 & p3=$!

    # Wait and capture exit statuses
    local s1=0 s2=0 s3=0
    wait $p1; s1=$? || true
    wait $p2; s2=$? || true
    wait $p3; s3=$? || true

    # Record failures from return codes
    if [[ $s1 -ne 0 ]]; then
        PHASE_FAILURES+=("phase1")
    fi
    if [[ $s2 -ne 0 ]]; then
        PHASE_FAILURES+=("phase2")
    fi
    if [[ $s3 -ne 0 ]]; then
        PHASE_FAILURES+=("phase3")
    fi

    # Also scan phase master logs for JOB FAILED (catches hard kills / internal failures)
    # Check both canonical and possible fallback locations (setup_phase_logging may redirect to /tmp on permission errors)
    local m1="$LOG_DIR/phase1/master_${RUN_TS}.log"
    local m2="$LOG_DIR/phase2/master_${RUN_TS}.log"
    local m3="$LOG_DIR/phase3/master_${RUN_TS}.log"
    local fb1="/tmp/homesensekd_phase1_master_${RUN_TS}.log"
    local fb2="/tmp/homesensekd_phase2_master_${RUN_TS}.log"
    local fb3="/tmp/homesensekd_phase3_master_${RUN_TS}.log"

    if phase_had_job_failures "$m1" || phase_had_job_failures "$fb1"; then
        if [[ " ${PHASE_FAILURES[*]} " != *" phase1 "* ]]; then
            PHASE_FAILURES+=("phase1")
        fi
    fi
    if phase_had_job_failures "$m2" || phase_had_job_failures "$fb2"; then
        if [[ " ${PHASE_FAILURES[*]} " != *" phase2 "* ]]; then
            PHASE_FAILURES+=("phase2")
        fi
    fi
    if phase_had_job_failures "$m3" || phase_had_job_failures "$fb3"; then
        if [[ " ${PHASE_FAILURES[*]} " != *" phase3 "* ]]; then
            PHASE_FAILURES+=("phase3")
        fi
    fi
}

case "$PHASE" in
    1|phase1) run_phase1 || PHASE_FAILURES+=("phase1") ;;
    2|phase2) run_phase2 || PHASE_FAILURES+=("phase2") ;;
    3|phase3) run_phase3 || PHASE_FAILURES+=("phase3") ;;
    all)      run_all_phases_parallel ;;
    *) echo "[error] Unknown --phase '$PHASE'  (use 1, 2, 3, or all)"; exit 1 ;;
esac

# Final summary (always runs; reports any captured failures)
{
    echo "============================================================"
    if ((${#PHASE_FAILURES[@]} > 0)); then
        echo "  All experiments complete (with FAILURES)."
        echo "  Failed phases: ${PHASE_FAILURES[*]}"
    else
        echo "  All experiments complete."
    fi
    echo "  Phase 1 logs    -> $LOG_DIR/phase1/"
    echo "  Phase 2 logs    -> $LOG_DIR/phase2/"
    echo "  Phase 3 logs    -> $LOG_DIR/phase3/"
    echo "  Phase 1 results -> $SCRIPT_DIR/results/phase1/"
    echo "  Phase 2 results -> $SCRIPT_DIR/results/phase2/"
    echo "  Phase 3 results -> $SCRIPT_DIR/results/phase3/"
    echo "  Global master   -> $GLOBAL_MASTER"
    echo "============================================================"
} | log_ts | safe_tee "$GLOBAL_MASTER"

if ((${#PHASE_FAILURES[@]} > 0)); then
    exit 1
fi

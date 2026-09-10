#!/bin/bash
# Phase 2 of the Turbo Heretic A/B: Donato Core-19 (kyuz0/terminal-bench-mini), two arms.
#
#   core19_campaign.sh smoke                 # heretic, smoke tier (git-leak-recovery), then restore production
#   core19_campaign.sh full                  # heretic full arm, then Turbo Q4_K_S full arm, back-to-back
#   core19_campaign.sh full-arm heretic|turbo
#
# Each arm = the live production launch line with only -m swapped, except that the test server binds
# 127.0.0.1:8085 (port split, the owner 2026-09-10): anything that targets production's 8080 fails loudly
# instead of sharing the GPU with an arm. Captured during the run (unrecoverable afterwards):
#   <arm>/llama-server.log      per-request timings, cache hits, MTP acceptance, any stray requests
#   <arm>/gpu_power_temp.jsonl  rocm-smi power + temperature for both GPUs every 30 s
# Everything else (jobs/, results/, ATIF transcripts) is written by the runner itself.
# Production (qwen38.service on 8080) is restored on exit, whatever happens.
set -uo pipefail

MODE="${1:?usage: core19_campaign.sh smoke|full|full-arm <heretic|turbo>}"
ONLY="${2:-}"

TB="$HOME/projects/terminal-bench-mini"
C19="$HOME/benchmarks/core19"
EM="$HOME/benchmarks/engine_matrix"
BIN="$HOME/projects/llama.cpp-master/build-vulkan/bin/llama-server"
MMPROJ=/mnt/models/llm/qwen3.8-27b-instruct-unsloth/mmproj-BF16.gguf
PORT=8085
EP="http://127.0.0.1:$PORT/v1"

HERETIC=/mnt/models/llm/qwen3.8-27b-heretic-trohrbaugh/qwen3.8-27b-heretic-trohrbaugh-q4_k_s.gguf
HERETIC_NAME="Qwen3.8-27B-Heretic-trohrbaugh"
TURBO=/mnt/models/llm/qwen3.8-27b-turbo-fable-cold-fusion-735-heretic-davidau/qwen3.8-27b-turbo-fable-cold-fusion-735-heretic-davidau-mtp-q4_k_s.gguf
TURBO_NAME="Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic"

ID=(--platform r9700
    --platform-name "AMD Radeon AI PRO R9700 32GB (single card, LACT Vram1 250 W cap)"
    --engine llama.cpp --engine-version "434ddbb+vision-patch"
    --backend vulkan --backend-version "mesa-26.2.2-radv"
    --inference-profile mtp-n2)

mkdir -p "$C19"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$C19/campaign.log"; }

gpu0_mib() {  # the 32 GB R9700, picked by size (tool numbering is inverted)
  for d in /sys/class/drm/card*/device; do
    t=$(cat "$d/mem_info_vram_total" 2>/dev/null) || continue
    [ "$t" -gt 32000000000 ] && echo $(( $(cat "$d/mem_info_vram_used") / 1048576 ))
  done
}

http_code() { curl -s -o /dev/null -w '%{http_code}' -m 4 "$1" 2>/dev/null || echo 000; }

stop_production() {
  for _ in $(seq 1 120); do
    busy=$(curl -s -m 5 http://127.0.0.1:8080/slots 2>/dev/null \
      | python3 -c 'import sys,json; print(sum(1 for s in json.load(sys.stdin) if s.get("is_processing")))' 2>/dev/null || echo 0)
    [ "$busy" = "0" ] && break
    log "production slot busy; waiting for the turn to finish"
    sleep 10
  done
  live=$(pgrep -af '[l]lama-server' | cut -d' ' -f2-)
  if [ -n "$live" ]; then
    r="$EM/restore/llamaserver_cmd_production_mtp_n2_$(date +%Y%m%d_%H%M).txt"
    echo "$live" > "$r"
    log "restore line saved -> $r"
  fi
  systemctl --user stop qwen38.service
  for _ in $(seq 1 60); do
    if [ -z "$(pgrep -x llama-server)" ] && [ "$(gpu0_mib)" -lt 1500 ]; then
      log "production stopped; GPU0 at $(gpu0_mib) MiB"
      return 0
    fi
    sleep 2
  done
  log "ABORT: GPU0 not freed ($(gpu0_mib) MiB; llama-server pids: $(pgrep -x llama-server | tr '\n' ' '))"
  return 1
}

stop_arm_processes() {
  tmux kill-session -t c19srv 2>/dev/null
  tmux kill-session -t c19pwr 2>/dev/null
  for _ in $(seq 1 30); do [ -z "$(pgrep -x llama-server)" ] && return 0; sleep 2; done
}

restore_production() {
  stop_arm_processes
  systemctl --user start qwen38.service
  for _ in $(seq 1 90); do
    if [ "$(http_code http://127.0.0.1:8080/health)" = "200" ]; then
      log "production restored: qwen38.service healthy on 8080"
      return 0
    fi
    sleep 5
  done
  log "WARNING: production did not come back healthy on 8080"
}

start_server() {
  local arm=$1 model=$2 dir="$C19/$1"
  mkdir -p "$dir"
  stop_arm_processes
  local cmd="$BIN -m $model --mmproj $MMPROJ --image-min-tokens 1024 -dev Vulkan0 -ngl 99 -c 262144 -ctk q8_0 -ctv q8_0 -fa on --ctx-checkpoints 2 -np 2 -kvu --jinja --reasoning-format deepseek --spec-type draft-mtp --spec-draft-n-max 2 -cram 12288 --host 127.0.0.1 --port $PORT --log-timestamps --log-prefix"
  echo "$cmd" > "$dir/launch.txt"
  tmux new-session -d -s c19srv "$cmd >> $dir/llama-server.log 2>&1"
  tmux new-session -d -s c19pwr "$C19/gpu_sampler.sh $dir/gpu_power_temp.jsonl"
  for i in $(seq 1 150); do
    if [ "$(http_code "http://127.0.0.1:$PORT/health")" = "200" ]; then
      log "$arm: server healthy on 127.0.0.1:$PORT after ~$((i * 4))s, GPU0 $(gpu0_mib) MiB"
      return 0
    fi
    if [ "$i" -gt 10 ] && [ -z "$(pgrep -x llama-server)" ]; then break; fi
    sleep 4
  done
  log "ABORT: $arm server failed to start (tail of $dir/llama-server.log follows)"
  tail -n 15 "$dir/llama-server.log" | tee -a "$C19/campaign.log"
  return 1
}

run_arm() {
  local arm=$1 model=$2 name=$3 quant=$4 tier=$5 dir="$C19/$1" rc
  log "================ $arm: $name $quant tier=$tier ================"
  start_server "$arm" "$model" || return 1
  cd "$TB" || return 1
  log "$arm: doctor"
  if ! ./terminal_bench.py doctor --tier "$tier" --endpoint "$EP" >> "$dir/doctor.log" 2>&1; then
    log "ABORT: $arm doctor failed (tail of $dir/doctor.log follows)"
    tail -n 25 "$dir/doctor.log" | tee -a "$C19/campaign.log"
    return 1
  fi
  log "$arm: runner starting"
  ./terminal_bench.py run --tier "$tier" --endpoint "$EP" \
    --model-name "$name" --quant "$quant" "${ID[@]}" 2>&1 | tee -a "$dir/runner.log"
  rc=${PIPESTATUS[0]}
  log "$arm: runner exited rc=$rc"
  stop_arm_processes
  return "$rc"
}

trap restore_production EXIT
stop_production || exit 1

case "$MODE" in
  smoke)
    run_arm smoke-heretic "$HERETIC" "$HERETIC_NAME" Q4_K_S smoke ;;
  full)
    run_arm full-heretic "$HERETIC" "$HERETIC_NAME" Q4_K_S full
    run_arm full-turbo "$TURBO" "$TURBO_NAME" Q4_K_S full ;;
  full-arm)
    case "$ONLY" in
      heretic) run_arm full-heretic "$HERETIC" "$HERETIC_NAME" Q4_K_S full ;;
      turbo)   run_arm full-turbo "$TURBO" "$TURBO_NAME" Q4_K_S full ;;
      *) log "usage: full-arm heretic|turbo"; exit 2 ;;
    esac ;;
  *) log "unknown mode: $MODE"; exit 2 ;;
esac
log "CORE19 CAMPAIGN DONE ($MODE)"

#!/bin/bash
# Core-19 against an ALREADY-RUNNING OpenAI-compatible endpoint (no server management).
#
# Written for the vLLM MXFP4 arms: switching models costs ~8-9 min of Triton/inductor JIT,
# so the campaign must not restart anything. It does NOT touch production and does NOT
# start or stop a server - the caller owns the endpoint's lifecycle.
#
#   core19_external.sh <arm-dir> <tier> <model-name> <quant> [task-ids...]
#
# Differences from core19_campaign.sh that affect comparability, stated up front:
#   - context is 65,536 (a TP=1 vLLM serve on one card), not the 262,144 the GGUF arms had.
#     Passed explicitly with --context-length so the agent's summarisation threshold is
#     correct rather than guessed; the runner will warn that it is below its recommended
#     262,144, which is expected and honest.
#   - reasoning effort is forced via TB_EXTRA_BODY (the template default is medium, this
#     model's WORST setting); the GGUF arms used the GGUF's own embedded template.
set -uo pipefail
ARM="${1:?usage: core19_external.sh <arm-dir> <tier> <model-name> <quant> [ids...]}"
TIER="${2:?}"; NAME="${3:?}"; QUANT="${4:?}"; shift 4
TB="$HOME/projects/terminal-bench-mini"
C19="$HOME/benchmarks/core19"
PORT="${PORT:-8085}"
EP="http://127.0.0.1:$PORT/v1"
CTX="${CTX:-65536}"
EFFORT="${EFFORT:-xhigh}"
dir="$C19/$ARM"; mkdir -p "$dir"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$C19/campaign.log"; }

ID=(--platform "${PLATFORM:-r9700-330w}"
    --platform-name "${PLATFORM_NAME:-AMD Radeon AI PRO R9700 32GB (single card, 330 W cap)}"
    --engine vllm --engine-version "0.27.1-radiance-0.9.3"
    --backend rocm --backend-version "rocm-7.14"
    --inference-profile "mxfp4-w4a8-dflash2-spec5")

code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/health" || echo 000)
[ "$code" = "200" ] || { log "ABORT: no healthy endpoint on $PORT (got $code)"; exit 1; }
log "$ARM: endpoint healthy, ctx=$CTX effort=$EFFORT model=$NAME"

export TB_EXTRA_BODY="{\"chat_template_kwargs\": {\"reasoning_effort\": \"$EFFORT\"}}"
tmux kill-session -t c19pwr 2>/dev/null
tmux new-session -d -s c19pwr "$C19/gpu_sampler.sh $dir/gpu_power_temp.jsonl"

cd "$TB" || exit 1
log "$ARM: doctor"
if ! ./terminal_bench.py doctor --tier "$TIER" --endpoint "$EP" --context-length "$CTX" >> "$dir/doctor.log" 2>&1; then
  log "ABORT: doctor failed"; tail -n 25 "$dir/doctor.log" | tee -a "$C19/campaign.log"
  tmux kill-session -t c19pwr 2>/dev/null; exit 1
fi
log "$ARM: runner starting"
./terminal_bench.py run --tier "$TIER" --endpoint "$EP" --context-length "$CTX" \
  --model-name "$NAME" --quant "$QUANT" "${ID[@]}" "$@" 2>&1 | tee -a "$dir/runner.log"
rc=${PIPESTATUS[0]}
log "$ARM: runner exited rc=$rc"
tmux kill-session -t c19pwr 2>/dev/null
exit "$rc"

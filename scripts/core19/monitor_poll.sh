#!/bin/bash
# monitor_poll.sh -- print only NEW Core-19 campaign events since the previous call.
# Called every few minutes by a Claude Code Monitor over ssh; every printed line becomes a notification,
# so it stays sparse: campaign.log lines, one line per finished trial, and liveness alerts.
C19="$HOME/benchmarks/core19"
TB="$HOME/projects/terminal-bench-mini"
S="$C19/.monitor_state"
mkdir -p "$S"
touch "$S/seen_trials"

# 1. new campaign.log lines (offset initialised at deploy to where the full campaign began)
n=$(wc -l < "$C19/campaign.log")
last=$(cat "$S/log_lines" 2>/dev/null || echo 0)
if [ "$n" -gt "$last" ]; then
  sed -n "$((last + 1)),${n}p" "$C19/campaign.log" | sed 's/^/LOG /'
fi
echo "$n" > "$S/log_lines"

# 2. newly finished trials in the full-campaign jobs (jobs/<job>/<task>__<id>/result.json)
find "$TB/jobs" -mindepth 3 -maxdepth 3 -name result.json -path '*core19-full*' 2>/dev/null | sort | while read -r f; do
  grep -qxF "$f" "$S/seen_trials" && continue
  echo "$f" >> "$S/seen_trials"
  python3 - "$f" <<'PY'
import json, os, sys
f = sys.argv[1]
trial = os.path.basename(os.path.dirname(f))
job = os.path.basename(os.path.dirname(os.path.dirname(f)))
arm = "turbo" if "turbo" in job.lower() else ("heretic" if "heretic" in job.lower() else "?")
attempt = "attempt2" if "attempt2" in job else "attempt1"
try:
    d = json.load(open(f))
except Exception as e:
    print(f"TRIAL {arm} {attempt} {trial} unreadable: {e}")
    raise SystemExit
vr = d.get("verifier_result") or {}
reward = (vr.get("rewards") or {}).get("reward", vr.get("reward", d.get("reward")))
exc = d.get("exception_info") or {}
exc_type = exc.get("exception_type") if isinstance(exc, dict) else exc
ar = d.get("agent_result") or {}
start, end = d.get("started_at"), d.get("finished_at")
dur = ""
try:
    from datetime import datetime
    dur = "%.0fm" % ((datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() / 60)
except Exception:
    pass
print(f"TRIAL {arm} {attempt} {trial.split('__')[0]} reward={reward} exc={exc_type} dur={dur} out_tok={ar.get('n_output_tokens')}")
PY
done

# 3. liveness alerts (each emitted once)
alert_once() {  # key, message
  [ -f "$S/alert_$1" ] && return
  touch "$S/alert_$1"
  echo "ALERT $2"
}
if ! tmux has-session -t c19 2>/dev/null; then
  grep -q "CORE19 CAMPAIGN DONE (full)" "$C19/campaign.log" || alert_once session_gone "c19 tmux session is gone but campaign.log has no 'CORE19 CAMPAIGN DONE (full)'"
else
  if tail -n 1 "$C19/campaign.log" | grep -q "runner starting" && ! ss -ltn 2>/dev/null | grep -q '127.0.0.1:8085'; then
    alert_once "no8085_$(date +%Y%m%d%H)" "an arm's runner is active but nothing is listening on 127.0.0.1:8085 (server crash?)"
  fi
fi

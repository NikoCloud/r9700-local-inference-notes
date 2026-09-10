#!/bin/bash
# gpu_sampler.sh OUTFILE -- one JSON line every 30 s: ISO timestamp + rocm-smi power/temperature for both GPUs.
# Captured during Core-19 arms so thermal or power behaviour can be ruled in or out afterwards.
OUT="${1:?usage: gpu_sampler.sh OUTFILE}"
while true; do
  s=$(rocm-smi --showpower --showtemp --json 2>/dev/null | tr -d '\n')
  printf '{"t":"%s","smi":%s}\n' "$(date -Is)" "${s:-null}" >> "$OUT"
  sleep 30
done

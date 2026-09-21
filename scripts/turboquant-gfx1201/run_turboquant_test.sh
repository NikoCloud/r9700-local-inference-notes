#!/bin/bash
# Dedicated test launcher for the TurboQuant KV-cache-dtype experiment (MTP drafter,
# radiance/paroquant stack). Forked from run_paroquant_nospec.sh rather than patched in
# place, so the production launcher stays untouched. Two deliberate differences from it:
#   - KV_DTYPE and ATTN_BACKEND are real first-class knobs (auto-picks TURBOQUANT backend
#     for turboquant_* dtypes; R4D otherwise -- R4D cannot read turboquant-quantized KV).
#   - compilation-config omits compile_sizes / max_cudagraph_capture_size entirely. Under
#     MTP speculative decoding, vLLM's cudagraph batch-size padding does not fix any point
#     in {1,2,4,8} (each gets padded to something else, e.g. 8->9, 1->3, depending on the
#     spec width), so any hardcoded compile_sizes list here is a ticking time bomb. Omitting
#     the key lets vLLM derive safe sizes itself instead of guessing values by trial and
#     costly error (each wrong guess is a 20-45 min cold rebuild on this box).
# MODE=eval / CHECKALL numerics-gate machinery from the original is dropped: this script only
# ever serves (mirrors MODE=prod).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PATCHES_DIR="$(realpath -m "${PATCHES:-$SCRIPT_DIR/..}")"
MODELS="$(realpath -m "${MODELS:-$HOME/models}")"
HF_CACHE="$(realpath -m "${HF_CACHE:-$HOME/.cache/huggingface}")"

RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v podman >/dev/null 2>&1; then RUNTIME=podman
  elif command -v docker >/dev/null 2>&1; then RUNTIME=docker
  else echo "no container runtime found: install podman (preferred) or docker" >&2; exit 1
  fi
fi
command -v "$RUNTIME" >/dev/null 2>&1 || { echo "RUNTIME=$RUNTIME is not on PATH" >&2; exit 1; }

RT_FLAGS=()
GROUP_FLAGS=()
if [ "$RUNTIME" = podman ]; then
  RT_FLAGS+=(--replace)
  GROUP_FLAGS+=(--group-add keep-groups)
else
  for g in render video; do
    gid=$(getent group "$g" 2>/dev/null | cut -d: -f3) || true
    if [ -n "$gid" ]; then GROUP_FLAGS+=(--group-add "$gid"); fi
  done
fi

PORT=${PORT:-8085}
NAME=${NAME:-vllmtest}
SERVED_NAMES=${SERVED_NAMES:-Qwen3.8-PARO}
SPEC_METHOD=${SPEC_METHOD:-mtp}
SPEC=${SPEC:-2}
DRAFTER=${DRAFTER:-Qwen3.8-27B-DFlash2-FP8}
PREFIX_CACHE=${PREFIX_CACHE:-1}
CAPTURE_DIR=${CAPTURE_DIR:-}
PROFILE=${PROFILE:-0}
ASYNC=${ASYNC:-0}
if [ "$ASYNC" = 1 ]; then ASYNC_FLAG="--async-scheduling"; UNPAD=false; else ASYNC_FLAG="--no-async-scheduling"; UNPAD=true; fi
CHUNK=${CHUNK:-8192}

MAXLEN=${MAXLEN:-262144}
MAXSEQS=${MAXSEQS:-8}
GPU_UTIL=${GPU_UTIL:-0.94}
# Explicit pin bypasses gpu_memory_utilization's auto-maximizing KV sizing entirely (vLLM
# ignores GPU_UTIL when this is set). Needed because vLLM always consumes the FULL util-implied
# budget for KV blocks regardless of chunk size, leaving no headroom for turboquant's
# continuation-prefill fallback burst no matter how low GPU_UTIL goes without also breaking
# the KV pool's ability to fit MAXLEN. Not part of the compile-graph hash, safe to change
# without clearing the compile cache.
KV_MEM_BYTES=${KV_MEM_BYTES:-}
KV_MEM_ARGS=(); [ -n "$KV_MEM_BYTES" ] && KV_MEM_ARGS=(--kv-cache-memory-bytes "$KV_MEM_BYTES")

# turboquant_attn.py fixes NUM_KV_SPLITS at this value for cudagraph shape-stability
# (vllm/config/attention.py default: 32). Per-split serial work is context_len/splits,
# so at a fixed split count the decode kernel's inner loop grows unboundedly with depth
# -- this is the likely source of the accelerating (not flat) decode-vs-depth falloff
# measured 2026-09-20. 32 CUs on this card are already oversubscribed at the default
# (24 heads x 32 splits = 768 program instances for one request), so raising splits has
# no real downside: bigger mid_o_buf/lse_buf workspace (KB-scale, negligible) and a bit
# more stage-2 reduction work (O(splits), cheap).
TQ_KV_SPLITS=${TQ_KV_SPLITS:-}
TQ_SPLITS_ARGS=(); [ -n "$TQ_KV_SPLITS" ] && TQ_SPLITS_ARGS=(--attention-config "{\"tq_max_kv_splits_for_cuda_graph\": $TQ_KV_SPLITS}")

# KV dtype + attention backend: the one thing this fork exists to make a first-class knob.
# TURBOQUANT is the only ROCm attention backend that can read turboquant-quantized KV; R4D
# (radiance's own custom kernel, the production default) explicitly rejects it at boot with
# "kv_cache_dtype not supported" (confirmed empirically 2026-09-19).
KV_DTYPE=${KV_DTYPE:-fp8}
case "$KV_DTYPE" in
  turboquant*) ATTN_BACKEND=${ATTN_BACKEND:-TURBOQUANT} ;;
  *)           ATTN_BACKEND=${ATTN_BACKEND:-R4D} ;;
esac

GDN_FUSED=${RADIANCE_GDN_FUSED_UPDATE:-1}
R4D_KEY=${R4D_KEY:-b9e42ab-rx6}
R4D_CACHE=${R4D_CACHE:-$HOME/.cache/radiance-libr4d}
[ -f "$R4D_CACHE/$R4D_KEY/r4d.so" ] || {
  echo "libr4d $R4D_KEY not built at $R4D_CACHE/$R4D_KEY/r4d.so" >&2
  exit 1
}
ROT_STREAM=${RADIANCE_PQ_ROT_STREAM:-1}
ROT_STREAM2=${RADIANCE_PQ_ROT_STREAM2:-1}
ROT_STREAM3=${RADIANCE_PQ_ROT_STREAM3:-0}
CACHE_SUF="-turboquant"; [ "$GDN_FUSED" = 1 ] && CACHE_SUF="$CACHE_SUF-fu"; [ "$ROT_STREAM" = 1 ] && CACHE_SUF="$CACHE_SUF-rs"
[ "$ROT_STREAM2" = 1 ] && CACHE_SUF="${CACHE_SUF}-rs2"
[ "$ROT_STREAM3" = 1 ] && CACHE_SUF="${CACHE_SUF}-rs3"
CACHE=${CACHE:-$HOME/.radiance-cache-paro-093$CACHE_SUF}
mkdir -p "$CACHE"

MODEL_DIR=${MODEL_DIR:-qwen3.8-27b-heretic-paro-mxfp4-mtp}
MODEL=/models/$MODEL_DIR
[ -d "$MODELS/$MODEL_DIR" ] || { echo "model missing at $MODELS/$MODEL_DIR" >&2; exit 1; }

TP=${TP:-1}
GPUS=${GPUS:-0}
HIP_IDX=$(seq -s, 0 $(( $(tr -cd , <<<"$GPUS" | wc -c) )))
MEM_LIMIT=${MEM_LIMIT:-}
MEM_ARGS=(); [ -n "$MEM_LIMIT" ] && MEM_ARGS=(--memory "$MEM_LIMIT")
CAPTURE_MOUNT=(); if [ -n "$CAPTURE_DIR" ]; then mkdir -p "$CAPTURE_DIR"; CAPTURE_MOUNT=(-v "$CAPTURE_DIR:/capture:z"); fi

PROF_ARGS=(); [ "$PROFILE" = 1 ] && PROF_ARGS=(--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/cache/prof --profiler-config.torch_profiler_with_stack=false) && mkdir -p "$CACHE/prof"
EXTRA_ARGS=("${PROF_ARGS[@]}" --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --max-num-batched-tokens "$CHUNK"
            $([ "$PREFIX_CACHE" = 1 ] && echo --enable-prefix-caching || echo --no-enable-prefix-caching)
            --compilation-config
            '{"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true},"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false,"combo_kernels":true,"benchmark_combo_kernel":true,"triton.cooperative_reductions":true}}')

if [ "${NOSPEC:-0}" = 1 ]; then
  SPEC_ARGS=()
elif [ "$SPEC_METHOD" = mtp ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC}}")
else
  SPEC_ARGS=(--speculative-config
    "{\"method\":\"dflash\",\"model\":\"/models/${DRAFTER}\",\"num_speculative_tokens\":${SPEC},\"attention_backend\":\"TRITON_ATTN\",\"disable_padded_drafter_batch\":${UNPAD},\"draft_sample_method\":\"greedy\"}")
fi

if [ "$RUNTIME" != podman ]; then "$RUNTIME" rm -f "$NAME" >/dev/null 2>&1 || true; fi

MM_ARGS=(); [ -n "${MM_KWARGS:-}" ] && MM_ARGS=(--mm-processor-kwargs "$MM_KWARGS")
CPUSET_ARGS=(); [ -n "${CPUSET:-}" ] && CPUSET_ARGS=(--cpuset-cpus "$CPUSET")

echo "### run_turboquant_test.sh: kv_dtype=$KV_DTYPE attn_backend=$ATTN_BACKEND maxlen=$MAXLEN model=$MODEL_DIR spec=$SPEC_METHOD/$SPEC ###" >&2

exec "$RUNTIME" run "${RT_FLAGS[@]}" "${CPUSET_ARGS[@]}" --name "$NAME" --privileged --ipc=host --network=host "${MEM_ARGS[@]}" \
  --device /dev/kfd --device /dev/dri "${GROUP_FLAGS[@]}" \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e ROCR_VISIBLE_DEVICES="$GPUS" -e HIP_VISIBLE_DEVICES="$HIP_IDX" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e RADIANCE_USE_R4D=1 -e RADIANCE_USE_R4D_AR=1 -e RADIANCE_USE_R4D_AR_QUANT=1 \
  -e RADIANCE_R4D_REPORT=1 -e RADIANCE_AR_MAX_KB=86016 \
  -e RADIANCE_STEP_TRACE="${RADIANCE_STEP_TRACE:-0}" \
  -e RADIANCE_DFLASH_CAPTURE_DIR="${CAPTURE_DIR:+/capture}" "${CAPTURE_MOUNT[@]}" \
  -e RADIANCE_PRESHUFFLE=1 -e RADIANCE_FUSE_RMS_QUANT=1 \
  -e R4D_ATTN_FP8=3 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -e RADIANCE_GDN_FUSED_UPDATE="$GDN_FUSED" -e RADIANCE_GDN_MERGE_INPROJ="${RADIANCE_GDN_MERGE_INPROJ:-0}" \
  -e RADIANCE_GDN_FUSED_MAX_ITEMS="${RADIANCE_GDN_FUSED_MAX_ITEMS:-32}" \
  -e RADIANCE_DYNAMIC_WIDTH=1 -e RADIANCE_DYNW_ALPHA=0.35 -e RADIANCE_DYNW_MARGIN=2 \
  -e RADIANCE_DYNW_MIN=2 -e RADIANCE_DYNW_MIN_BATCH=3 \
  -e RADIANCE_AR_QNB=96 -e RADIANCE_AR_QNT=1024 -e RADIANCE_AR_OVERLAP=0 \
  -e RADIANCE_DFLASH_SELECTOR_TOPK= \
  -e RADIANCE_PAROQUANT=1 \
  -e RADIANCE_MXFP4_W4A8="${RADIANCE_MXFP4_W4A8:-1}" -e RADIANCE_MXFP4_WPERM="${RADIANCE_MXFP4_WPERM:-1}" \
  -e RADIANCE_MXFP4_DECODE_NT="${RADIANCE_MXFP4_DECODE_NT:-1}" -e RADIANCE_MXFP4_DECODE_MAX_M="${RADIANCE_MXFP4_DECODE_MAX_M:-64}" \
  -e RADIANCE_MXFP4_EPIFAST="${RADIANCE_MXFP4_EPIFAST:-1}" -e RADIANCE_MXFP4_TN4_MIN_M="${RADIANCE_MXFP4_TN4_MIN_M:-2048}" \
  -e RADIANCE_MXFP4_A_TILED_MIN_M="${RADIANCE_MXFP4_A_TILED_MIN_M:-513}" \
  -e RADIANCE_PQ_WPERM="${RADIANCE_PQ_WPERM:-1}" -e RADIANCE_PQ_DECODE_NT="${RADIANCE_PQ_DECODE_NT:-1}" \
  -e RADIANCE_PQ_ATILED="${RADIANCE_PQ_ATILED:-1}" -e RADIANCE_PQ_AT_LBK="${RADIANCE_PQ_AT_LBK:-128}" \
  -e RADIANCE_PQ_AT_HOIST="${RADIANCE_PQ_AT_HOIST:-1}" -e RADIANCE_PQ_PTOK="${RADIANCE_PQ_PTOK:-1}" \
  -e RADIANCE_PQ_ROT_STREAM="$ROT_STREAM" -e RADIANCE_PQ_ROT_STREAM2="$ROT_STREAM2" \
  -e RADIANCE_PQ_ROT_STREAM3="$ROT_STREAM3" -e RADIANCE_PQ_AR_CHECK="${RADIANCE_PQ_AR_CHECK:-0}" \
  -e RADIANCE_PQ_AR_FALLBACK="${RADIANCE_PQ_AR_FALLBACK:-0}" \
  -e RADIANCE_PQ_ROT_V2="${RADIANCE_PQ_ROT_V2:-1}" \
  -e RADIANCE_PQ_SKIP="${RADIANCE_PQ_SKIP:-.*mtp.*}" \
  -e RADIANCE_FAST_DRAFT="${RADIANCE_FAST_DRAFT:-1}" -e RADIANCE_DRAFT_TAU=0.20 -e RADIANCE_DRAFT_RERANK=80 \
  -e RADIANCE_VERIFY_HEAD=1 -e RADIANCE_VERIFY_HEAD_MAX_M=32 \
  -e RADIANCE_TOPK_TRITON_MIN_ROWS=1 -e RADIANCE_SKINNY_GEMM="${RADIANCE_SKINNY_GEMM:-1}" \
  -e RADIANCE_GDN_PATHS=both \
  -e RADIANCE_KV_GROUP_OPT=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1 \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$MODELS":/models \
  -v "$CACHE":/cache \
  -v "$PATCHES_DIR":/patches:z \
  -v "$SCRIPT_DIR":/paro:z \
  -v "$R4D_CACHE/$R4D_KEY":/r4d:z \
  -e R4D_SO="$R4D_CACHE/$R4D_KEY" \
  --entrypoint bash "${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}" -lc '
    set -e
    SP=/opt/vllm/lib/python3.12/site-packages
    cd /patches
    python3 patch_quark_mxfp4.py
    python3 patch_ar_maxbytes.py
    python3 patch_topk_triton_rows.py
    python3 patch_dflash_calib.py
    python3 patch_dflash_mxfp4_kv.py
    python3 patch_rmsquant_fusion.py
    python3 patch_verify_head.py
    python3 patch_kv_group_size.py
    python3 patch_topk_composite.py
    python3 patch_gdn_shared_build.py
    python3 patch_async_dynwidth.py
    python3 patch_step_trace.py
    python3 patch_skinny_gemm.py
    python3 patch_dflash_selector_topk.py
    python3 patch_dynwidth.py
    python3 patch_ar_geometry.py
    python3 patch_gdn_merge_inproj.py
    python3 patch_qwen3_thinkoff.py \
      || echo "[radiance] WARNING: thinkoff patch did not apply"
    python3 patch_turboquant_flash_attn.py
    cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/
    # RADIANCE_FA_ANY_ARCH follow-up: AITER Triton MHA needs a per-arch autotune config
    # and ships none for gfx1201 (only gfx1151/gfx1250/gfx942/gfx950). gfx1151 (Strix
    # Halo) is the closest architectural relative -- same RDNA4-family Triton kernels,
    # modest block sizes (BLOCK_M=64/BLOCK_N=32) well inside the 64KB LDS budget on gfx1201.
    # Correctness risk is low: a wrong config causes an obvious kernel-launch failure,
    # not silent bad output.
    if [ ! -f "$SP"/aiter/ops/triton/configs/gfx1201-MHA-DEFAULT.json ]; then
      cp "$SP"/aiter/ops/triton/configs/gfx1151-MHA-DEFAULT.json \
         "$SP"/aiter/ops/triton/configs/gfx1201-MHA-DEFAULT.json
      echo "[radiance] gfx1201-MHA-DEFAULT.json missing from AITER -- stood in the gfx1151 config"
    fi
    cp radiance_mxfp4.py radiance_gemm.py radiance_gdn.py radiance_gdnmerge.py radiance_rmsquant.py \
       radiance_drafthead.py radiance_verifyhead.py radiance_aroverlap.py radiance_topk.py \
       radiance_arnq.py "$SP"/
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so
    if [ -n "${R4D_SO:-}" ] && [ -f /r4d/r4d.so ]; then
      cp /r4d/r4d.so "$SP"/r4d.so
      echo "[radiance] using patched r4d.so from $R4D_SO"
    fi
    cd /paro
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_paroquant.hip -o "$SP"/radiance_paroquant_kernel.so
    cp radiance_paroquant.py radiance_paroquant_mxfp4.py "$SP"/
    cp /patches/radiance_dflash_capture.py "$SP"/ 2>/dev/null || cp ../radiance_dflash_capture.py "$SP"/
    printf "%s\n" \
      "try:" \
      "    import radiance_paroquant" \
      "    import radiance_paroquant_mxfp4" \
      "    import radiance_dflash_capture" \
      "except Exception as e:" \
      "    import sys" \
      "    sys.stderr.write(\"[radiance.paroquant] registration failed: %r\\n\" % (e,))" \
      >> /usr/lib/python3.12/sitecustomize.py
    cd /
    exec /opt/radiance_entrypoint.sh "$@"' \
  _ \
  "$MODEL" \
  --served-model-name $SERVED_NAMES \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype "$KV_DTYPE" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" \
  "${KV_MEM_ARGS[@]}" \
  "${TQ_SPLITS_ARGS[@]}" \
  --attention-backend "$ATTN_BACKEND" \
  $ASYNC_FLAG \
  --mamba-cache-mode align \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20}' \
  --chat-template /root/.cache/huggingface/qwen-fixed-v22.3.jinja \
  "${SPEC_ARGS[@]}" \
  "${MM_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

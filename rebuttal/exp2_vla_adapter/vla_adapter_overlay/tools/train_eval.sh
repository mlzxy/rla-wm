#!/usr/bin/env bash
# Train + eval VLA-Adapter entirely inside the pristine upstream clone
# on the authors' own pinned stack:
# torch 2.2.0 / transformers 4.40.1 / timm 0.9.10 / numpy 1.26.4 / flash-attn 2.5.6,
# with upstream's robosuite 1.4.1 / mujoco 3.3.0 / LIBERO 8f1084e3 for eval.
#
#   tools/train_eval.sh <spatial|object|goal|long>
#   tools/train_eval.sh spatial                     # defaults below
#   ACCUM=1 tools/train_eval.sh object              # grad accumulation 2 -> 1
#   GPUS="0 1 2 3" tools/train_eval.sh goal         # multi-GPU DDP
#   EVAL_AFTER=0 tools/train_eval.sh long           # train only
#
# Env:
#   GPUS         GPUs for DDP (default: all on the machine)
#   BATCH        per-GPU batch size                       (default 8)
#   ACCUM        gradient accumulation steps              (default 1)
#   STEPS        override the per-suite step count
#   SAVE_FREQ    checkpoint interval (default STEPS/10, so 10 checkpoints)
#   LR           learning rate                            (default 2e-4)
#   LORA_RANK    LoRA rank                                (default 64)
#   SHUFFLE_BUFFER  RLDS frame shuffle buffer, per rank   (default 80000)
#                NOTE: this is the one deliberate deviation from upstream, which uses 100000.
#                It is held per rank, so 4-GPU DDP keeps 4 copies (~3.6 GB each at 100000).
#                Job 59120608 was OOM-killed by the SLURM cgroup at 194.1/193.8 GiB; 80000
#                buys back ~2.9 GB across 4 ranks. tf.data's shuffle is a sliding window,
#                not a global permutation, so this is a *sampling* knob, not a speed knob --
#                but every suite still keeps >=0.79 epochs resident, so batch composition is
#                statistically unchanged. Set SHUFFLE_BUFFER=100000 for a bit-faithful rerun.
#   EVAL_AFTER   1 = evaluate after training              (default 1)
#   EVAL_TRIALS  trials/task for the final eval           (default 50 -> 500 episodes)
#   SWEEP_TRIALS trials/task when ranking checkpoints     (default 2 -> 20 episodes)
#   SWEEP        1 = rank every saved checkpoint and full-eval the best
#                0 = full-eval the final checkpoint only  (default 1)
#   SWEEP_ORDER  asc (default, oldest first) or desc -- order the sweep walks checkpoints
#   RUN_TAG      suffix for the run id, so a rerun does not overwrite older checkpoints
#   WANDB_MODE   online|offline|disabled                  (default offline)
set -euo pipefail

REF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (this file lives in tools/)
PY="$REF/.venv/bin/python"
TORCHRUN="$REF/.venv/bin/torchrun"

SUITE="${1:?usage: $0 <spatial|object|goal|long>}"

# suite -> "<dataset> <Suite> <task_suite_name> <steps>"
# Step counts are upstream's reported best-checkpoint points (github issue #35).
case "$SUITE" in
    spatial) SPEC="libero_spatial_no_noops Spatial libero_spatial 20000" ;;
    object)  SPEC="libero_object_no_noops  Object  libero_object  20000" ;;
    goal)    SPEC="libero_goal_no_noops    Goal    libero_goal    50000" ;;
    long)    SPEC="libero_10_no_noops      Long    libero_10      60000" ;;
    *) echo "unknown suite: $SUITE (expected spatial|object|goal|long)" >&2; exit 2 ;;
esac
read -r DATASET Suite TASK_SUITE DEFAULT_STEPS <<<"$SPEC"

: "${BATCH:=8}" "${ACCUM:=2}" "${LR:=2e-4}" "${LORA_RANK:=64}" "${SHUFFLE_BUFFER:=80000}"
: "${STEPS:=$DEFAULT_STEPS}" "${EVAL_AFTER:=1}" "${EVAL_TRIALS:=50}" "${SWEEP_TRIALS:=2}" "${SWEEP:=1}"
: "${WANDB_MODE:=online}"; export WANDB_MODE
SAVE_FREQ="${SAVE_FREQ:-10000}"

if [ -z "${GPUS:-}" ]; then
    n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    [ "${n:-0}" -ge 1 ] 2>/dev/null || n=1
    GPUS="$(seq 0 $((n - 1)) | tr '\n' ' ')"
fi
read -r -a gpu_arr <<<"$GPUS"
NGPU="${#gpu_arr[@]}"
GPU_CSV="$(IFS=,; echo "${gpu_arr[*]}")"
EFF_BATCH=$(( BATCH * NGPU * ACCUM ))

RUN_ID="REF-${Suite}${RUN_TAG:+-$RUN_TAG}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REF/train_logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SUITE}--${STAMP}.log"

# Everything below the env vars is the upstream tree's own code, unmodified.
export LIBERO_CONFIG_PATH="$REF/.libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTHONPATH="$REF:$REF/LIBERO"

# Cap glibc malloc arenas. The default on 64-bit is 8 x ncores (240 here), and freed blocks in
# an arena are effectively never returned to the OS, so a thread-heavy process (tf.data pipeline
# + TF thread pools + torch + NCCL) accumulates fragmented arenas and RSS only ever climbs. The
# OOM dump for job 59120608 showed 191.4 GiB inactive_anon against 1.9 GiB active_anon -- i.e.
# ~99% of the resident memory was allocated-but-cold, the signature of arena fragmentation
# rather than a live working set. Purely an allocator setting; cannot change training results.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

cat <<EOF
==========================================================================
 upstream reference run: $SUITE
   run_id      $RUN_ID
   GPUs        $GPU_CSV  (x$NGPU)
   batch       $BATCH/GPU  x  accum $ACCUM   ->  effective batch $EFF_BATCH
   steps       $STEPS      (checkpoint every $SAVE_FREQ)
   lr $LR   lora_rank $LORA_RANK   shuffle_buffer $SHUFFLE_BUFFER$([ "$SHUFFLE_BUFFER" != 100000 ] && echo "  (upstream: 100000)")
   log         $LOG
==========================================================================
EOF

cd "$REF"
CUDA_VISIBLE_DEVICES="$GPU_CSV" "$TORCHRUN" --standalone --nnodes 1 --nproc-per-node "$NGPU" \
    vla-scripts/finetune.py \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --config_file_path pretrained_models/configs \
    --data_root_dir data/libero \
    --dataset_name "$DATASET" \
    --run_root_dir outputs \
    --run_id_override "$RUN_ID" \
    --use_minivlm True \
    --use_pro_version True \
    --use_l1_regression True \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --use_lora True \
    --lora_rank "$LORA_RANK" \
    --shuffle_buffer_size "$SHUFFLE_BUFFER" \
    --use_fz False \
    --image_aug True \
    --learning_rate "$LR" \
    --batch_size "$BATCH" \
    --grad_accumulation_steps "$ACCUM" \
    --num_steps_before_decay "$STEPS" \
    --max_steps "$STEPS" \
    --save_freq "$SAVE_FREQ" \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --wandb_project "vla-adapter-ref" \
    --run_id_note "$SUITE" \
    2>&1 | tee "$LOG"

# --- loss curve ----------------------------------------------------------------------------
# Always plotted, even with EVAL_AFTER=0: a rising tail means the run diverged, which is worth
# seeing before spending an hour of rollouts on it.
CURVE="$LOG_DIR/${SUITE}--${STAMP}-loss.png"
"$PY" "$REF/tools/plot_loss.py" \
    --run-id "$RUN_ID" --out "$CURVE" --log "$LOG" --steps "$STEPS" \
    --title "$RUN_ID  ($SUITE, ${STEPS} steps, eff. batch ${EFF_BATCH})" || true

[ "$EVAL_AFTER" = "1" ] || { echo "training done (EVAL_AFTER=0)"; exit 0; }

# --- eval ---------------------------------------------------------------------------------
EVAL_OUT="$REF/eval_out"; mkdir -p "$EVAL_OUT"

# Every result is appended to this file the moment it is produced, so a lost terminal or a
# reclaimed node never costs you the numbers. Self-describing: the run config is written first.
RESULTS="$EVAL_OUT/${RUN_ID}--${STAMP}-results.tsv"
{
    echo "# run_id=$RUN_ID suite=$SUITE steps=$STEPS batch=$BATCH accum=$ACCUM eff_batch=$EFF_BATCH"
    echo "# lr=$LR lora_rank=$LORA_RANK shuffle_buffer=$SHUFFLE_BUFFER gpus=$GPU_CSV host=$(hostname -s)"
    echo "# train_log=$LOG"
    echo "# loss_curve=$CURVE"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' when checkpoint kind trials success_pct eval_log
} > "$RESULTS"
echo "results file: $RESULTS"

run_eval() {   # <ckpt-dir-name> <trials> <label>
    local ckpt="$1" trials="$2" label="$3"
    local log="$EVAL_OUT/${SUITE}--${ckpt}--${label}.log"
    CUDA_VISIBLE_DEVICES="${gpu_arr[0]}" "$PY" experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint "outputs/$ckpt" \
        --task_suite_name "$TASK_SUITE" \
        --use_pro_version True \
        --use_proprio True \
        --num_images_in_input 2 \
        --use_film False \
        --num_trials_per_task "$trials" \
        --local_log_dir "$EVAL_OUT" \
        --run_id_note "${label}-${ckpt}" > "$log" 2>&1 || true
    local r pct
    r=$(tr '\r' '\n' < "$log" | grep -ao 'successes: [0-9]* ([0-9.]*%)' | tail -1)
    pct=$(sed -E 's/.*\(([0-9.]+)%\)/\1/' <<<"${r:-0 (0%)}")
    # append + fsync-ish: one line, one write, immediately after the eval returns
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%H:%M:%S)" "$ckpt" "$label" "$trials" "${pct:-0}" "$log" >> "$RESULTS"
    echo "$r"
}

# Ascending by step. The sweep walks this order, so you watch the curve rise, peak and (if it
# happens) collapse -- more legible than starting at the end and reading a run of zeros.
# SWEEP_ORDER=desc walks newest-first instead.
SORT_FLAGS="-n"; [ "${SWEEP_ORDER:-asc}" = "desc" ] && SORT_FLAGS="-rn"
mapfile -t CKPTS < <(ls -d "$REF/outputs/${RUN_ID}--"*_chkpt 2>/dev/null \
    | sed -E 's/.*--([0-9]+)_chkpt$/\1 &/' | sort $SORT_FLAGS | cut -d' ' -f2- | xargs -r -n1 basename)
[ "${#CKPTS[@]}" -gt 0 ] || { echo "no checkpoints found for $RUN_ID" >&2; exit 1; }
echo "saved checkpoints: ${CKPTS[*]}"

pct_of() { sed -E 's/.*\(([0-9.]+)%\)/\1/' <<<"${1:-0 (0%)}"; }

# 1) the final checkpoint, straight away -- this is the number you asked the run for.
#    Highest step wins, independent of SWEEP_ORDER.
FINAL=$(printf '%s\n' "${CKPTS[@]}" | sed -E 's/.*--([0-9]+)_chkpt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
echo
echo "==> eval FINAL checkpoint $FINAL (${EVAL_TRIALS} trials/task)"
FINAL_R=$(run_eval "$FINAL" "$EVAL_TRIALS" final)
FINAL_PCT=$(pct_of "$FINAL_R")
echo "    $FINAL  ->  ${FINAL_PCT}%"

# 2) then check whether an earlier checkpoint is better. Worth doing: success can collapse long
#    before the loss does, so the last checkpoint is not reliably the best one.
BEST="$FINAL"; BEST_PCT="$FINAL_PCT"
if [ "$SWEEP" = "1" ] && [ "${#CKPTS[@]}" -gt 1 ]; then
    echo
    echo "==> ranking the other $(( ${#CKPTS[@]} - 1 )) checkpoints at ${SWEEP_TRIALS} trials/task"
    best_t=$(awk -v p="${FINAL_PCT:-0}" 'BEGIN{printf "%d", p*10 + 0.5}')
    for c in "${CKPTS[@]}"; do
        [ "$c" = "$FINAL" ] && { printf '    %-42s %6s%%   (final, %s trials)\n' "$c" "$FINAL_PCT" "$EVAL_TRIALS"; continue; }
        pct=$(pct_of "$(run_eval "$c" "$SWEEP_TRIALS" sweep)")
        printf '    %-42s %6s%%\n' "$c" "${pct:-0}"
        t=$(awk -v p="${pct:-0}" 'BEGIN{printf "%d", p*10 + 0.5}')
        [ "$t" -gt "$best_t" ] && { best_t=$t; BEST="$c"; BEST_PCT="$pct"; }
    done
    if [ "$BEST" != "$FINAL" ]; then
        echo
        echo "==> $BEST beat the final checkpoint on the short eval; full eval on it (${EVAL_TRIALS} trials/task)"
        BEST_PCT=$(pct_of "$(run_eval "$BEST" "$EVAL_TRIALS" final)")
    fi
fi

{
    echo "#"
    printf '# SUMMARY  final\t%s\t%s%%\n' "$FINAL" "$FINAL_PCT"
    [ "$BEST" != "$FINAL" ] && printf '# SUMMARY  best\t%s\t%s%%\n' "$BEST" "$BEST_PCT"
    echo "# done $(date '+%F %T')"
} >> "$RESULTS"

echo
echo "=========================================================================="
printf ' %s  final %s -> %s%%\n' "$SUITE" "$FINAL" "$FINAL_PCT"
[ "$BEST" != "$FINAL" ] && printf ' %s  best  %s -> %s%%\n' "$SUITE" "$BEST" "$BEST_PCT"
echo " loss curve: $CURVE"
echo " results:    $RESULTS"
echo "=========================================================================="

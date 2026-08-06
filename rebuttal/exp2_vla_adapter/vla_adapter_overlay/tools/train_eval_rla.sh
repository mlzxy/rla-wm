#!/usr/bin/env bash
# Two-stage RLA training on top of the pristine upstream clone.
#
#   stage 1   RLA-only pretraining on all four LIBERO suites (no action labels used)
#   stage 2   suite-specific action + RLA co-training, warm-started from stage 1
#
# Upstream code is never modified: rla/train.py loads vla-scripts/finetune.py and patches its
# module globals. tools/train_eval.sh remains the untouched vanilla arm.
#
#   tools/train_eval_rla.sh pretrain                      # stage 1, all suites, ~20k steps
#   RLA_INIT_FROM=outputs/RLA-P1--20000_chkpt \
#     tools/train_eval_rla.sh long                        # stage 2 on LIBERO-Long
#
# Env (in addition to everything tools/train_eval.sh accepts):
#   RLA_SIDECAR    sidecar dir       (default data/libero_rla/16x64-a8-step040000-snapshot)
#   RLA_STEPS      RLA token groups; must equal NUM_ACTIONS_CHUNK          (default 8)
#   RLA_QUERIES    Q per group, asserted against the sidecar manifest      (default 16)
#   RLA_DIM        D per query, likewise                                   (default 64)
#   LAMBDA_ACT     weight on the action L1     (stage 1 forces 0, stage 2 default 1.0)
#   LAMBDA_RLA     weight on the RLA L1: a float, or auto:R to hold the RLA term at a fixed
#                  fraction R of the action term      (stage 1 forces 1.0, stage 2 default auto:0.25)
#   RLA_ACTION_TOKENS  stage 1 only. 1 (default) keeps the 8 action tokens in the policy sequence,
#                  unsupervised, so both stages share one RoPE geometry and one state dict.
#                  0 pretrains on the 128 RLA tokens alone -- see rla/README.md section 3.
#   RLA_INIT_FROM  stage-1 checkpoint dir; required for stage 2
#   SKIP_PREFLIGHT 1 skips the sidecar<->RLDS bijection check (not recommended)
#   RSS_WATCH      1 (default) samples per-rank host RSS into <log>-rss.log every 60s
#
#   GPUS BATCH ACCUM STEPS SAVE_FREQ LR LORA_RANK SHUFFLE_BUFFER EVAL_AFTER EVAL_TRIALS SWEEP
#   SWEEP_TRIALS SWEEP_ORDER RUN_TAG WANDB_MODE   -- same meaning as in tools/train_eval.sh
#
# Host RAM is the binding constraint on this cluster, not GPU memory -- see tools/OOM_DEBUGGING.md
# and the STEP_WALL guard below. SHUFFLE_BUFFER defaults to 80000 here, matching the baseline arm.
set -euo pipefail

REF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REF/.venv/bin/python"
TORCHRUN="$REF/.venv/bin/torchrun"

STAGE="${1:?usage: $0 <pretrain|spatial|object|goal|long>}"

# suite -> "<dataset> <Suite> <task_suite_name> <steps>"
case "$STAGE" in
    # 20000 steps ~= 2.4 epochs over the 261.6k usable frames at effective batch 32. The default is
    # deliberately *not* higher: host RAM grows with steps, not with corpus size, and past ~45k steps
    # at 4 ranks a run walks into the SLURM cgroup limit (tools/OOM_DEBUGGING.md; see the guard below).
    pretrain) SPEC="libero_4_task_suites_no_noops Pretrain -      100000" ;;
    spatial)  SPEC="libero_spatial_no_noops       Spatial  libero_spatial 20000" ;;
    object)   SPEC="libero_object_no_noops        Object   libero_object  20000" ;;
    goal)     SPEC="libero_goal_no_noops          Goal     libero_goal    50000" ;;
    long)     SPEC="libero_10_no_noops            Long     libero_10      60000" ;;
    *) echo "unknown stage: $STAGE (expected pretrain|spatial|object|goal|long)" >&2; exit 2 ;;
esac
read -r DATASET Suite TASK_SUITE DEFAULT_STEPS <<<"$SPEC"

# --- RLA configuration ----------------------------------------------------------------------
export RLA_SIDECAR="${RLA_SIDECAR:-$REF/data/libero_rla/16x64-a8-step040000-snapshot}"
export RLA_STEPS="${RLA_STEPS:-8}"
export RLA_QUERIES="${RLA_QUERIES:-16}"
export RLA_DIM="${RLA_DIM:-64}"

if [ "$STAGE" = "pretrain" ]; then
    # Stage 1 is RLA-only by definition: the action labels are deliberately unused, so that this
    # stage is a legal pretraining task on actionless video (the LAPA/UniVLA protocol).
    export LAMBDA_ACT="${LAMBDA_ACT:-0.0}"
    export LAMBDA_RLA="${LAMBDA_RLA:-1.0}"
    # The action tokens stay in the sequence by default, unsupervised. No action labels are used
    # either way -- this is purely about whether the 8 action positions exist during pretraining.
    # Keeping them means stage 1 and stage 2 have identical RoPE geometry and an identical state
    # dict, and fc1 (896x6272, the largest matrix in the head) trains. RLA_ACTION_TOKENS=0 gives
    # the RLA-only sequence instead; rla/warmstart.py splices the gates, but the RLA tokens' RoPE
    # positions then shift by 8 between the stages.
    export RLA_ACTION_TOKENS="${RLA_ACTION_TOKENS:-1}"
    : "${EVAL_AFTER:=0}"        # nothing to roll out: this checkpoint predicts z, not actions
else
    export LAMBDA_ACT="${LAMBDA_ACT:-1.0}"
    # A fixed weight drifts: the action L1 falls ~10x over a run while the RLA L1 does not. auto:R
    # recomputes the weight every step so the RLA term stays at fraction R of the action term.
    export LAMBDA_RLA="${LAMBDA_RLA:-auto:0.25}"
    export RLA_ACTION_TOKENS=1          # stage 2 always has both groups; that is the whole point
    : "${EVAL_AFTER:=1}"
fi
export RLA_INIT_FROM="${RLA_INIT_FROM:-}"

[ -f "$RLA_SIDECAR/manifest.json" ] || {
    echo "RLA_SIDECAR=$RLA_SIDECAR is not a sidecar directory (no manifest.json)" >&2; exit 2; }

# Resolve the stage-1 checkpoint up front. rla/config.py checks the same things inside every rank,
# but that is after the two-minute preflight and after torchrun has spun up; a typo in a path should
# cost a second. Printing the resolved step in the banner below is also how you confirm that the
# checkpoint being loaded is the one you meant -- the step lives in the file name, not the dir name.
INIT_STEP=""
if [ -n "$RLA_INIT_FROM" ]; then
    [ -d "$RLA_INIT_FROM" ] || {
        echo "RLA_INIT_FROM=$RLA_INIT_FROM is not a directory. It must be a checkpoint dir written" >&2
        echo "  by stage 1: outputs/<run_id>--<step>_chkpt   (ls -d outputs/RLA-P1*--*_chkpt)" >&2
        exit 2; }
    mapfile -t HEADS < <(ls "$RLA_INIT_FROM"/action_head--*_checkpoint.pt 2>/dev/null || true)
    [ "${#HEADS[@]}" -eq 1 ] || {
        echo "expected exactly 1 action_head--*_checkpoint.pt in $RLA_INIT_FROM, found ${#HEADS[@]}" >&2
        exit 2; }
    INIT_STEP="$(basename "${HEADS[0]}" | sed -E 's/^action_head--(.*)_checkpoint\.pt$/\1/')"
    ls "$RLA_INIT_FROM"/*.safetensors >/dev/null 2>&1 || {
        echo "no merged VLM (*.safetensors) in $RLA_INIT_FROM -- only the action head would be" >&2
        echo "  warm-started. Stage 1 must run with --merge_lora_during_training True." >&2
        exit 2; }
fi

if [ "$STAGE" != "pretrain" ] && [ -z "$RLA_INIT_FROM" ]; then
    echo "warning: RLA_INIT_FROM is unset -- stage 2 will train from scratch, with no stage-1" >&2
    echo "         warm start. That is a valid arm (RLA as a pure auxiliary target), but if you" >&2
    echo "         meant the two-stage run, point it at outputs/RLA-P1--<step>_chkpt." >&2
fi

: "${BATCH:=8}" "${ACCUM:=1}" "${LR:=2e-4}" "${LORA_RANK:=64}" "${SHUFFLE_BUFFER:=80000}"
: "${STEPS:=$DEFAULT_STEPS}" "${EVAL_TRIALS:=50}" "${SWEEP_TRIALS:=2}" "${SWEEP:=1}"
: "${WANDB_MODE:=online}"; export WANDB_MODE
SAVE_FREQ="${SAVE_FREQ:-10000}"

# SHUFFLE_BUFFER=80000 is the value tools/train_eval.sh settled on after job 59120608 was
# OOM-killed, and it is inherited here for two independent reasons. Memory: the buffer is per rank
# and holds JPEG bytes (~36 KB/frame), so upstream's 100000 is ~3.6 GB/rank and 80000 buys back
# ~2.9 GB across 4 ranks. Comparability: every REF baseline in eval_out/ ran at 80000, and tf.data's
# shuffle is a *sampling* knob rather than a speed knob -- a stage-2 run at 100000 would not be
# sampling-comparable to the baseline it is being measured against.
#
# Cap glibc malloc arenas, as train_eval.sh does: the default is 8 x ncores, freed blocks in a
# non-main arena are effectively never returned to the OS, and the OOM dump showed 99% of resident
# memory was inactive_anon -- fragmentation, not a live working set. This matters *more* for stage 1
# than for a single-suite run: the 4-dataset mixture gets traj_read_threads = traj_transform_threads
# = 4 (logged as "Threads per Dataset: [1 1 1 1]") instead of 1, so there are more threads competing
# for arenas. Purely an allocator setting; cannot change training results.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

if [ -z "${GPUS:-}" ]; then
    n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    [ "${n:-0}" -ge 1 ] 2>/dev/null || n=1
    GPUS="$(seq 0 $((n - 1)) | tr '\n' ' ')"
fi
read -r -a gpu_arr <<<"$GPUS"
NGPU="${#gpu_arr[@]}"
GPU_CSV="$(IFS=,; echo "${gpu_arr[*]}")"
EFF_BATCH=$(( BATCH * NGPU * ACCUM ))

RUN_ID="RLA-$([ "$STAGE" = "pretrain" ] && echo P1 || echo "$Suite")${RUN_TAG:+-$RUN_TAG}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REF/train_logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/rla-${STAGE}--${STAMP}.log"
RSS_LOG="$LOG_DIR/rla-${STAGE}--${STAMP}-rss.log"

# --- host-RAM headroom ------------------------------------------------------------------------
# Host RAM, not GPU: job 59120608 was SIGKILLed by the SLURM cgroup at 194.1/193.8 GiB while GPU
# memory was fine (tools/OOM_DEBUGGING.md). Growth is driven by **steps**, not by corpus size,
# because `.repeat()` sits before `.shuffle()` so the frame buffer fills to SHUFFLE_BUFFER whatever
# the dataset, and there is one such buffer for the whole mixture (dataset.py:569, after
# sample_from_datasets). Measured here at 4 ranks, step 300, cgroup usage:
#
#                             B=100000, arenas default   B=80000, MALLOC_ARENA_MAX=2
#     4-suite mix (stage 1)      80.4 GiB (19.2/rank)        65.7 GiB (15.5/rank)
#     single suite (stage 2)     73.6 GiB (17.5/rank)        --
#
# Two things fall out. The mixture costs ~1.7 GiB/rank over a single suite -- three extra concurrent
# readers ("Threads per Dataset: [1 1 1 1]"), not 4x the buffer -- so **the corpus size is not the
# risk**. And the two settings above buy 14.7 GiB, which is worth more than the entire margin by
# which the last job overshot (0.2%).
#
# The slope comes from the incident itself: 73 -> 194 GiB over 52,857 steps is ~2.3 MiB/step summed
# across 4 ranks. Applied to the 65.7 GiB baseline that puts the wall near 57k steps, and it
# reproduces the observed death at 52.8k from the un-tuned baseline. Warn before the run, not at
# hour nine. The projection uses the mixture's baseline, so it is conservative for a single suite.
STEP_WALL=40000
RSS_BASE_GIB=66            # measured above, at the defaults this script sets
if [ "$STEPS" -gt "$STEP_WALL" ] && [ "$NGPU" -ge 4 ]; then
    cat >&2 <<WARN
--------------------------------------------------------------------------
 WARNING: $STEPS steps at $NGPU ranks is in host-RAM OOM territory.
   Measured baseline ~$RSS_BASE_GIB GiB cgroup at step 300, growing ~2.3 MiB/step across ranks
   => ~$(( (RSS_BASE_GIB * 1024 + STEPS * 23 / 10) / 1024 )) GiB projected at step $STEPS, against a 193.8 GiB cgroup limit.
   Job 59120608 died this way at step 52,857 of 60,000 (9h24m in, exitcode -9).
   Options: lower STEPS, SHUFFLE_BUFFER=50000, or GPUS="0 1 2" (fewer ranks = fewer copies).
   $RSS_LOG tells you within an hour.
--------------------------------------------------------------------------
WARN
fi

export LIBERO_CONFIG_PATH="$REF/.libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTHONPATH="$REF:$REF/LIBERO"

cd "$REF"

# The banner comes before the preflight: the resolved configuration -- above all which stage-1
# checkpoint is about to be loaded, and at which step -- is what you read before letting a run go,
# and it should not sit behind two minutes of join checking.
cat <<EOF
==========================================================================
 RLA $STAGE
   run_id      $RUN_ID
   dataset     $DATASET
   GPUs        $GPU_CSV  (x$NGPU)
   batch       $BATCH/GPU  x  accum $ACCUM   ->  effective batch $EFF_BATCH
   steps       $STEPS      (checkpoint every $SAVE_FREQ)
   lr $LR   lora_rank $LORA_RANK   shuffle_buffer $SHUFFLE_BUFFER$([ "$SHUFFLE_BUFFER" != 80000 ] && echo "  (baseline arm: 80000)")
   rla         ${RLA_STEPS}x${RLA_QUERIES}x${RLA_DIM} tokens
   lambda      act=$LAMBDA_ACT  rla=$LAMBDA_RLA
   sidecar     $RLA_SIDECAR
   warm start  ${RLA_INIT_FROM:-<none>}${INIT_STEP:+  @ step $INIT_STEP}
   log         $LOG
   rss         $RSS_LOG
==========================================================================
EOF

# --- host-RAM watch ---------------------------------------------------------------------------
# tools/OOM_DEBUGGING.md closes by recommending exactly this, but its snippet greps `[f]inetune.py`
# and so matches nothing here: the RLA entry point is rla/train.py. Sample it ourselves, next to the
# training log. One hour of this extrapolates to whether the run clears STEPS -- the alternative is
# finding out at hour nine, which is what happened to job 59120608.
if [ "${RSS_WATCH:-1}" = "1" ]; then
    (
        cg="/sys/fs/cgroup/memory/slurm_$(hostname -s)/uid_$(id -u)/job_${SLURM_JOB_ID:-none}"
        while sleep 60; do
            pgrep -f "rla/train[.]py" >/dev/null 2>&1 || break
            printf '%s  %s  cgroup=%s\n' "$(date '+%F %T')" \
                "$(ps -eo rss,pid,cmd | grep '[r]la/train.py' \
                    | awk '{printf "%d:%.1fG ", $2, $1/1048576; s+=$1} END {printf "| sum=%.1fG", s/1048576}')" \
                "$(awk '{printf "%.1fG", $1/1073741824}' "$cg/memory.usage_in_bytes" 2>/dev/null || echo "n/a")"
        done
    ) > "$RSS_LOG" 2>/dev/null &
    RSS_PID=$!
    trap 'kill "${RSS_PID:-}" 2>/dev/null || true' EXIT
fi

# --- preflight ------------------------------------------------------------------------------
# Once, in one process, before torchrun: prove the sidecar and data/libero are a clean bijection.
# Cheaper here than inside every rank, and a failure costs seconds instead of an hour. The in-run
# guard still stands: a join miss raises inside RlaBatchTransform rather than yielding zeros.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    echo "==> preflight: sidecar <-> data/libero join"
    RLA_PREFLIGHT=1 "$PY" -m rla.tests --gates join --rlds-root data/libero \
        || { echo "preflight FAILED -- refusing to train against a partial join" >&2; exit 3; }
fi

RLA_PREFLIGHT=0 CUDA_VISIBLE_DEVICES="$GPU_CSV" \
    "$TORCHRUN" --standalone --nnodes 1 --nproc-per-node "$NGPU" \
    rla/train.py \
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
    --wandb_project "vla-adapter-rla" \
    --run_id_note "rla-$STAGE" \
    2>&1 | tee "$LOG"

CURVE="$LOG_DIR/rla-${STAGE}--${STAMP}-loss.png"
"$PY" "$REF/tools/plot_loss.py" \
    --run-id "$RUN_ID" --out "$CURVE" --log "$LOG" --steps "$STEPS" \
    --title "$RUN_ID  ($STAGE, ${STEPS} steps, eff. batch ${EFF_BATCH})" || true

if [ "$STAGE" = "pretrain" ]; then
    echo
    echo "=========================================================================="
    echo " stage 1 done. Feed it to stage 2 with:"
    echo "   RLA_INIT_FROM=outputs/${RUN_ID}--${STEPS}_chkpt tools/train_eval_rla.sh long"
    echo " loss curve: $CURVE"
    echo "=========================================================================="
    exit 0
fi

[ "$EVAL_AFTER" = "1" ] || { echo "training done (EVAL_AFTER=0)"; exit 0; }

# --- eval -------------------------------------------------------------------------------------
# Same shape as tools/train_eval.sh: full-eval the final checkpoint, then rank the rest on a short
# eval and full-eval the winner. Success can collapse long before the loss does, so the last
# checkpoint is not reliably the best one.
EVAL_OUT="$REF/eval_out"; mkdir -p "$EVAL_OUT"
RESULTS="$EVAL_OUT/${RUN_ID}--${STAMP}-results.tsv"
{
    echo "# run_id=$RUN_ID stage=$STAGE steps=$STEPS batch=$BATCH accum=$ACCUM eff_batch=$EFF_BATCH"
    echo "# lr=$LR lora_rank=$LORA_RANK shuffle_buffer=$SHUFFLE_BUFFER gpus=$GPU_CSV host=$(hostname -s)"
    echo "# rla=${RLA_STEPS}x${RLA_QUERIES}x${RLA_DIM} lambda_act=$LAMBDA_ACT lambda_rla=$LAMBDA_RLA"
    echo "# sidecar=$RLA_SIDECAR"
    echo "# init_from=${RLA_INIT_FROM:-<none>}"
    echo "# train_log=$LOG"
    echo "# loss_curve=$CURVE"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' when checkpoint kind trials success_pct eval_log
} > "$RESULTS"
echo "results file: $RESULTS"

run_eval() {   # <ckpt-dir-name> <trials> <label>
    local ckpt="$1" trials="$2" label="$3"
    local log="$EVAL_OUT/rla-${STAGE}--${ckpt}--${label}.log"
    # rla/eval.py, not run_libero_eval.py: the checkpoint's action head carries the RLA tokens and
    # get_action_head loads strict, so the same RLA_* must be in scope here as during training.
    CUDA_VISIBLE_DEVICES="${gpu_arr[0]}" "$PY" rla/eval.py \
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%H:%M:%S)" "$ckpt" "$label" "$trials" "${pct:-0}" "$log" >> "$RESULTS"
    echo "$r"
}

SORT_FLAGS="-n"; [ "${SWEEP_ORDER:-asc}" = "desc" ] && SORT_FLAGS="-rn"
mapfile -t CKPTS < <(ls -d "$REF/outputs/${RUN_ID}--"*_chkpt 2>/dev/null \
    | sed -E 's/.*--([0-9]+)_chkpt$/\1 &/' | sort $SORT_FLAGS | cut -d' ' -f2- | xargs -r -n1 basename)
[ "${#CKPTS[@]}" -gt 0 ] || { echo "no checkpoints found for $RUN_ID" >&2; exit 1; }
echo "saved checkpoints: ${CKPTS[*]}"

pct_of() { sed -E 's/.*\(([0-9.]+)%\)/\1/' <<<"${1:-0 (0%)}"; }

FINAL=$(printf '%s\n' "${CKPTS[@]}" | sed -E 's/.*--([0-9]+)_chkpt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
echo
echo "==> eval FINAL checkpoint $FINAL (${EVAL_TRIALS} trials/task)"
FINAL_R=$(run_eval "$FINAL" "$EVAL_TRIALS" final)
FINAL_PCT=$(pct_of "$FINAL_R")
echo "    $FINAL  ->  ${FINAL_PCT}%"

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
printf ' %s  final %s -> %s%%\n' "$STAGE" "$FINAL" "$FINAL_PCT"
[ "$BEST" != "$FINAL" ] && printf ' %s  best  %s -> %s%%\n' "$STAGE" "$BEST" "$BEST_PCT"
echo " loss curve: $CURVE"
echo " results:    $RESULTS"
echo "=========================================================================="

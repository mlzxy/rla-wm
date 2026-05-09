import sys
from argparse import Namespace
import random
import os
import time
from rich import print
import json
import shutil
import subprocess
import glob
from dataclasses import dataclass, field
from typing import Optional
import torch
import optuna
import tyro
from datalib.ppo_cpu_able import Args

VALID_CONFIGS = [
    # ("PushT-v2", "panda_stick"),  # TESTING
    ("PokeCube-v2", "xarm6_robotiq"),
    ("PullCube-v2", "xarm6_robotiq"),
    # ("StackCube-v1", "xarm6_robotiq"),
    ("PullCubeTool-v1", "xarm6_robotiq"),
    ("PushT-v2", "xarm6_robotiq_closed"),
    ("RollBall-v1", "xarm6_robotiq_closed"),
    ("PegInsertionSide-v1", "xarm6_robotiq"),
    # ("PokeCube-v2", "ur10e_stick"),
    ("PullCube-v2", "panda"),
    ("PullCube-v2", "ur10e_stick"),
    # Non-stick tasks
    # ("StackCube-v1", "panda"),
    ("RollBall-v1", "panda_closed"),
    ("PegInsertionSide-v1", "panda"),
    ("RollBall-v1", "ur10e_stick"),
    ("PushT-v2", "panda_closed"),
    ("PushT-v2", "ur10e_stick"),
    ("PokeCube-v2", "panda"),
    ("PullCubeTool-v1", "panda"),
]  # 0-14

# Task-specific hyperparameters based on datalib/baselines.sh
TASK_HYPERPARAMS = {
    "PushT-v2": {
        "total_timesteps": 50_000_000,
        "update_epochs": 8,
        "num_minibatches": 32,
        "num_steps": 16 * 32,
        "num_eval_steps": 100,
        "gamma": 0.99,
    },
    "PokeCube-v2": {
        "total_timesteps": 50_000_000,
        "update_epochs": 8,
        "num_minibatches": 32,
        "num_steps": 4 * 32,
        "num_eval_steps": 50,
    },
    "PullCube-v2": {
        "total_timesteps": 50_000_000,
        "update_epochs": 8,
        "num_minibatches": 32,
        "num_steps": 4 * 32,
        "num_eval_steps": 50,
    },
    "StackCube-v1": {
        "total_timesteps": 50_000_000,
        "update_epochs": 8,
        "num_minibatches": 32,
        "num_steps": 16 * 32,
        "num_eval_steps": 50,
    },
    "RollBall-v1": {
        "total_timesteps": 50_000_000,
        "update_epochs": 8,
        "num_minibatches": 32,
        "num_steps": 16 * 32,
        "num_eval_steps": 80,
        "gamma": 0.95,
    },
    "PegInsertionSide-v1": {
        "total_timesteps": 75_000_000,
        "update_epochs": 8,
        "num_minibatches": 32,
        "num_steps": 16 * 32,
        "num_eval_steps": 100,
        "gamma": 0.97,
        "gae_lambda": 0.95,
    },
    "PullCubeTool-v1": {
        "total_timesteps": 50_000_000,
        "update_epochs": 8,
        "num_minibatches": 32,
        "num_steps": 16 * 32,
        "num_eval_steps": 100,
    },
}


def get_python_executable():
    """Return the path to the python executable in the current venv or fallback to sys.executable."""
    venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


class CheckpointPool:
    def __init__(self, root_dir, bucket_step=0.1, max_per_bucket=5):
        self.root_dir = root_dir
        self.bucket_step = bucket_step
        self.max_per_bucket = max_per_bucket
        self.pool_json_path = os.path.join(root_dir, "pool.json")
        self.checkpoints_dir = os.path.join(root_dir, "checkpoints")
        self.videos_dir = os.path.join(root_dir, "videos")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        if os.path.exists(self.pool_json_path):
            with open(self.pool_json_path, "r") as f:
                self.pool_data = json.load(f)
        else:
            self.pool_data = {}  # bucket_key -> list of metadata

    def is_full(self):
        for i in range(1, 11):
            bucket_key = f"{i / 10:.1f}"
            if len(self.pool_data.get(bucket_key, [])) < self.max_per_bucket:
                return False
        return True

    def save(
        self,
        success_rate,
        trial_id,
        state_dict,
        hyperparams,
        exp_dir,
        reward=0.0,
        seed=None,
    ):
        if success_rate < 0.05:
            return

        bucket = round(round(success_rate / self.bucket_step) * self.bucket_step, 1)
        if bucket > 1.0:
            bucket = 1.0
        bucket_key = f"{bucket:.1f}"

        bucket_ckpt_dir = os.path.join(self.checkpoints_dir, bucket_key)
        bucket_video_dir = os.path.join(self.videos_dir, bucket_key)
        os.makedirs(bucket_ckpt_dir, exist_ok=True)
        os.makedirs(bucket_video_dir, exist_ok=True)

        timestamp = int(time.time())
        filename_base = f"ckpt_{success_rate:.3f}_t{trial_id}_{timestamp}"
        ckpt_path = os.path.join(bucket_ckpt_dir, filename_base + ".pt")

        torch.save(state_dict, ckpt_path)

        video_src = None
        video_dest = None
        video_search_dirs = [
            os.path.join(exp_dir, "videos"),
            os.path.join(exp_dir, "train_videos"),
        ]

        all_video_files = []
        for v_dir in video_search_dirs:
            if os.path.exists(v_dir):
                for f in os.listdir(v_dir):
                    if f.endswith(".mp4") or f.endswith(".webm"):
                        full_v_path = os.path.join(v_dir, f)
                        all_video_files.append(
                            (full_v_path, os.path.getmtime(full_v_path))
                        )

        if all_video_files:
            all_video_files.sort(key=lambda x: x[1])
            video_src = all_video_files[-1][0]
            video_dest = os.path.join(
                bucket_video_dir, filename_base + os.path.splitext(video_src)[1]
            )
            shutil.copy(video_src, video_dest)

        if bucket_key not in self.pool_data:
            self.pool_data[bucket_key] = []

        metadata = {
            "trial_id": trial_id,
            "seed": seed,
            "success_rate": float(success_rate),
            "reward": float(reward),
            "hyperparams": hyperparams,
            "ckpt_path": os.path.relpath(ckpt_path, self.root_dir),
            "video_path": os.path.relpath(video_dest, self.root_dir)
            if video_dest
            else None,
            "timestamp": timestamp,
        }
        self.pool_data[bucket_key].append(metadata)

        if len(self.pool_data[bucket_key]) > self.max_per_bucket:
            trial_counts = {}
            for item in self.pool_data[bucket_key]:
                tid = item["trial_id"]
                trial_counts[tid] = trial_counts.get(tid, 0) + 1
            max_tid = max(trial_counts, key=trial_counts.get)
            trial_items = [
                (idx, item)
                for idx, item in enumerate(self.pool_data[bucket_key])
                if item["trial_id"] == max_tid
            ]
            trial_items.sort(key=lambda x: x[1]["timestamp"])
            evict_idx, oldest = trial_items[0]
            self.pool_data[bucket_key].pop(evict_idx)
            old_ckpt = os.path.join(self.root_dir, oldest["ckpt_path"])
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
            if oldest["video_path"]:
                old_video = os.path.join(self.root_dir, oldest["video_path"])
                if os.path.exists(old_video):
                    os.remove(old_video)

        with open(self.pool_json_path, "w") as f:
            json.dump(self.pool_data, f, indent=4)

        print(f"Saved checkpoint and metadata to {bucket_key}")


def objective(trial, config_idx, pool, base_args, pool_args: "SearchArgs"):
    # Suggest hyperparameters
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)

    # Prepare Args
    task, robot = VALID_CONFIGS[config_idx]
    # 1. Start with TASK_HYPERPARAMS as baseline for the task
    args = Args()
    if task in TASK_HYPERPARAMS:
        for k, v in TASK_HYPERPARAMS[task].items():
            setattr(args, k, v)

    # 2. Override with base_args from CLI (user provided overrides)
    # Since base_args contains many default values, we only want to apply
    # those that the user actually passed. However, tyro doesn't easily distinguish.
    # A simple but effective way is to just apply ALL of them if they are non-default
    # or just trust they are what the user wants if they passed them.
    # For now, let's just copy everything from base_args over.
    for k, v in vars(base_args).items():
        setattr(args, k, v)

    # 3. Set config-specific fixed values
    args.env_id = task
    args.robot_uid = robot
    args.seed = random.choice([7, 13, 17, 19, 23, 42, 1337, 9351, 4796, 1788])

    # 4. Set search-suggested values (these MUST take the highest precedence)
    args.gamma = gamma
    args.gae_lambda = gae_lambda

    print(
        f"Trial {trial.number}: gamma={gamma:.4f}, gae_lambda={gae_lambda:.4f}, seed={args.seed}"
    )

    run_name = f"pool_{task}_{robot}_trial_{trial.number}"
    args.exp_name = run_name
    args.base_run_dir = "runs/SearchRuns-CPU"
    # Use a stable directory name for the "active" run of this config to support seamless resume
    exp_dir = os.path.join(args.base_run_dir, run_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 5. Check for Resuming, then pretrained checkpoint dir
    latest_ckpt_path = os.path.join(exp_dir, "latest.pt")
    if os.path.exists(latest_ckpt_path):
        print(f"Resuming Trial {trial.number} from {latest_ckpt_path}")
        args.checkpoint = latest_ckpt_path
    elif getattr(pool_args, "pretrained_checkpoint_dir", None):
        # Pick one checkpoint from the directory (random for diversity across trials)
        pt_files = glob.glob(
            os.path.join(pool_args.pretrained_checkpoint_dir, "*.pt")
        )
        if pt_files:
            chosen = random.choice(pt_files)
            args.checkpoint = os.path.abspath(chosen)
            print(
                f"Finetuning Trial {trial.number} from pretrained checkpoint: {args.checkpoint}"
            )

    # Construct PPO command (ppo_cpu_able with CPU sim)
    python_exe = get_python_executable()
    cmd = [
        python_exe,
        "-m",
        "datalib.ppo_cpu_able",
        "--force-cpu-sim",
        "--env-id",
        task,
        "--robot-uid",
        robot,
        "--seed",
        str(args.seed),
        "--gamma",
        str(gamma),
        "--gae-lambda",
        str(gae_lambda),
        "--exp-name",
        run_name,
        "--base-run-dir",
        args.base_run_dir,
        "--no-progress",
    ]
    if pool_args.mask_obs:
        cmd.append("--mask-obs")
    # Add other PPO args that might have been overridden
    # We skip gamma/gae_lambda/seed/exp-name as they are already handled
    skip_args = [
        "gamma",
        "gae_lambda",
        "seed",
        "exp_name",
        "base_run_dir",
        "env_id",
        "robot_uid",
        "no_progress",
        "checkpoint",
    ]
    # pool_args (SearchArgs) are not part of args (Args); only args.ppo is merged into args
    for k, v in vars(args).items():
        if k not in skip_args and v is not None:
            # Convert to CLI flag format
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(v)])

    metadata_dir = os.path.join(exp_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    cmd.extend(["--metadata-dir", metadata_dir])

    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])

    print(f"Running command: {' '.join(cmd)}")

    # Subprocess execution
    log_path = os.path.join(exp_dir, "log.txt")

    seen_metadata = set()
    final_reward = 0.0
    process = None
    trial_start_time = time.time()
    total_iterations = args.total_timesteps // (args.num_envs * args.num_steps)

    try:
        with open(log_path, "w") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            while process.poll() is None:
                # Read stdout/stderr line by line (non-blocking if possible, but Popen.stdout is a stream)
                # Actually, we can just monitor the files instead of stdout for robustness
                time.sleep(1.0)  # Check every second

                # 2. Check for new metrics JSON updates
                metric_pattern = os.path.join(metadata_dir, "metrics_*.json")
                metric_files = glob.glob(metric_pattern)
                for m_file in sorted(metric_files):
                    if m_file not in seen_metadata:
                        seen_metadata.add(m_file)
                        try:
                            with open(m_file, "r") as f:
                                m_data = json.load(f)

                            iteration = m_data["iteration"]
                            global_step = m_data["global_step"]
                            final_reward = m_data["reward"]
                            success = m_data.get("success", 0.0)

                            # Calculate ETA
                            elapsed = time.time() - trial_start_time
                            if iteration > 0:
                                time_per_iter = elapsed / iteration
                                remaining_iters = total_iterations - iteration
                                eta_seconds = time_per_iter * remaining_iters
                                eta_str = f"{int(eta_seconds // 60)}m{int(eta_seconds % 60):02d}s"
                            else:
                                eta_str = "N/A"

                            msg = f"Trial {trial.number} | Step {global_step} | Iter {iteration}/{total_iterations} | Reward: {final_reward:.3f} | Success: {success:.2f} | ETA: {eta_str}"
                            print(msg)

                        except (json.JSONDecodeError, KeyError, PermissionError):
                            # Silently ignore if file is still being written
                            pass

                # Check for new metadata JSON files
                meta_pattern = os.path.join(metadata_dir, "metadata_*.json")
                meta_files = glob.glob(meta_pattern)
                for meta_file in meta_files:
                    if meta_file not in seen_metadata:
                        seen_metadata.add(meta_file)
                        try:
                            with open(meta_file, "r") as f:
                                meta_data = json.load(f)

                            iteration = meta_data["iteration"]
                            success_rate = meta_data["success"]
                            final_reward = meta_data["reward"]
                            ckpt = meta_data["ckpt_path"]

                            # Calculate ETA
                            elapsed = time.time() - trial_start_time
                            if iteration > 0:
                                time_per_iter = elapsed / iteration
                                remaining_iters = total_iterations - iteration
                                eta_seconds = time_per_iter * remaining_iters
                                eta_str = f"{int(eta_seconds // 60)}m{int(eta_seconds % 60):02d}s"
                            else:
                                eta_str = "N/A"

                            msg = f"[red][EVAL][/red] Trial {trial.number} | Step {global_step} | Iter {iteration}/{total_iterations} | Reward: {final_reward:.3f} | Success: {success_rate:.2f} | ETA: {eta_str}"
                            print(msg)

                            # Save to pool
                            try:
                                state_dict = torch.load(ckpt, map_location="cpu")
                                hparams = {
                                    "gamma": gamma,
                                    "gae_lambda": gae_lambda,
                                    "num_steps": args.num_steps,
                                    "total_timesteps": args.total_timesteps,
                                    "num_envs": args.num_envs,
                                    "update_epochs": args.update_epochs,
                                    "num_minibatches": args.num_minibatches,
                                }

                                pool.save(
                                    success_rate,
                                    trial.number,
                                    state_dict,
                                    hparams,
                                    exp_dir,
                                    reward=final_reward,
                                    seed=args.seed,
                                )
                            except Exception as e:
                                print(f"Error loading checkpoint {ckpt}: {e}")

                        except Exception as e:
                            print(f"Error reading metadata {meta_file}: {e}")

                # Check for final metadata
                final_meta_path = os.path.join(metadata_dir, "final_metadata.json")
                if (
                    os.path.exists(final_meta_path)
                    and final_meta_path not in seen_metadata
                ):
                    seen_metadata.add(final_meta_path)
                    try:
                        with open(final_meta_path, "r") as f:
                            final_data = json.load(f)
                        final_reward = final_data.get("reward", final_reward)
                    except Exception as e:
                        print(f"Error reading final metadata: {e}")

    except Exception as e:
        if not isinstance(e, optuna.exceptions.TrialPruned):
            print(f"Trial failed with error: {e}")
        if process:
            process.terminate()
    finally:
        if process:
            process.wait()
            if process.returncode > 0:
                print(
                    f"\n[bold red]{'=' * 20} TRIAL {trial.number} FAILED (Exit Code: {process.returncode}) {'=' * 20}[/bold red]"
                )
                print(f"[yellow]Internal logs from {log_path}:[/yellow]")
                try:
                    with open(log_path, "r") as f:
                        log_content = f.read()
                        print(log_content)
                except Exception as e:
                    print(f"[red]Could not read logs: {e}[/red]")
                print(f"[bold red]{'=' * 60}[/bold red]\n")

        # Cleanup logic
        # returncode == 0: Success, always clean up
        # returncode > 0: Failure, clean up unless keep_failed is True
        # returncode < 0: Interrupted (e.g. SIGINT), keep for resume

        should_cleanup = False
        if process and process.returncode == 0:
            should_cleanup = True
        elif process and process.returncode > 0:
            if not pool_args.keep_failed:
                should_cleanup = True

        if should_cleanup and os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
            print(f"Cleaned up temporary run folder: {exp_dir}")

    return final_reward


@dataclass
class SearchArgs:
    config_idx: int
    """Index of VALID_CONFIGS"""
    gpu: int = 0
    """GPU ID to use"""
    n_trials: Optional[int] = None
    """Number of trials to run (None for infinite)"""
    seed: int = 9351
    """Seed for the experiment"""
    max_per_bucket: int = 20
    """Maximum checkpoints per success rate bucket"""
    stop_on_full: bool = False
    """Stop search if all success rate buckets are full"""
    storage: Optional[str] = None
    """Optuna storage URI (defaults to runs/Pool/optuna.db)"""
    capture_video: bool = False
    """whether to capture videos (passed to PPO)"""
    keep_failed: bool = False
    """whether to keep failed trial directories for debugging"""
    evaluate: bool = False
    """whether to run evaluation on the existing pool instead of searching"""
    mask_obs: bool = False
    """whether to mask the 6th and 12th element of the observation"""
    pretrained_checkpoint_dir: Optional[str] = None
    """Directory of *.pt checkpoints to finetune from; one is chosen per trial (random). Ignored if resuming from latest.pt."""
    ppo: Args = field(default_factory=Namespace)  # do not change this line!
    """PPO arguments"""


def evaluate_pool(pool, args, task, robot):
    print(f"Starts massive evaluation on {task} with {robot} (Pool: {pool.root_dir})")

    # 1. Gather all checkpoints to evaluate
    to_eval = []
    if not pool.pool_data:
        print("[yellow]No checkpoints found in pool.[/yellow]")
        return

    for bucket_key, items in pool.pool_data.items():
        if not items:
            continue
        # Pick best by reward
        best_item = max(items, key=lambda x: x["reward"])
        to_eval.append((bucket_key, best_item))

    # Sort just for nice output
    to_eval.sort(key=lambda x: float(x[0]))

    if not to_eval:
        print("[yellow]No valid checkpoints found in pool buckets.[/yellow]")
        return

    results = []

    eval_root = "runs/Pool-eval"
    os.makedirs(eval_root, exist_ok=True)

    python_exe = get_python_executable()

    for bucket_key, item in to_eval:
        ckpt_full_path = os.path.join(pool.root_dir, item["ckpt_path"])
        if not os.path.exists(ckpt_full_path):
            print(f"[red]Warning: Checkpoint {ckpt_full_path} not found[/red]")
            continue

        print(f"Evaluating bucket {bucket_key} (Reward: {item['reward']:.2f})...")

        # Output dir for this evaluation
        run_name = f"{args.config_idx}_{task}_{robot}_s{bucket_key}"
        eval_output_dir = os.path.join(eval_root, run_name)

        # Temp metadata dir
        meta_dir = os.path.join(eval_output_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)

        # Construct command (ppo_cpu_able with CPU sim for evaluation)
        cmd = [
            python_exe,
            "-m",
            "datalib.ppo_cpu_able",
            "--force-cpu-sim",
            "--env-id",
            task,
            "--robot-uid",
            robot,
            "--checkpoint",
            ckpt_full_path,
            "--evaluate",
            "--eval-output-dir",
            eval_output_dir,
            "--metadata-dir",
            meta_dir,
            "--no-progress",
        ]
        if args.mask_obs:
            cmd.append("--mask-obs")

        # Pass args.ppo overrides
        # We skip 'checkpoint' as we pass it explicitly
        # We also need to be careful not to duplicate flags if they are in args.ppo
        skip_args = [
            "checkpoint",
            "env_id",
            "robot_uid",
            "evaluate",
            "eval_output_dir",
            "metadata_dir",
            "no_progress",
            "exp_name",
            "base_run_dir",
        ]
        for k, v in vars(args.ppo).items():
            if k not in skip_args and v is not None:
                flag = "--" + k.replace("_", "-")
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(v)])

        # Force capture video if args.capture_video is True (it defaults to True in SearchArgs)
        # Note: ppo.py defaults capture_video=True. If args.ppo.capture_video is False, it might disable it.
        # But args.capture_video is the top-level SearchArgs.
        if args.capture_video:
            if "--capture-video" not in cmd:
                cmd.append("--capture-video")
        else:
            # If user explicitly wants no video, we should ensure it's off if ppo defaults to on?
            # ppo.py uses store_true? No, it uses bool with Tyro which handles --flag and --no-flag maybe?
            # Tyro handles bools as --flag / --no-flag.
            # args.ppo is Namespace from tyro? No, it's Args class.
            pass

        # Execute
        try:
            # Clean up previous run's metadata if any to avoid confusion?
            if os.path.exists(os.path.join(meta_dir, "final_metadata.json")):
                os.remove(os.path.join(meta_dir, "final_metadata.json"))

            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )

            # Read result
            meta_file = os.path.join(meta_dir, "final_metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file, "r") as f:
                    data = json.load(f)
                success = data.get("success", 0.0)
                results.append(
                    {
                        "bucket": bucket_key,
                        "planned_success": float(bucket_key),
                        "eval_success": success,
                    }
                )
                # Rename video if possible? Or verify its location
                # Video should be in eval_output_dir
                print(f"  -> Result: Planned {bucket_key}, Actual {success:.2f}")
            else:
                print("  -> [red]Failed to read metadata[/red]")
        except subprocess.CalledProcessError as e:
            print(f"  -> [red]Evaluation failed: {e}[/red]")
            import traceback

            traceback.print_exc()

    # Summary
    print("\nEvaluation Summary:")
    print("Bucket | Planned Success | Actual Success")
    print("-------|-----------------|---------------")
    for r in results:
        print(
            f"{r['bucket']:<6} | {r['planned_success']:<15.2f} | {r['eval_success']:.2f}"
        )


def main():
    args = tyro.cli(SearchArgs)
    # Clear sys.argv to prevent other libraries from parsing our search-specific args
    sys.argv = [sys.argv[0]]

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Set seed
    args.ppo.seed = args.seed
    args.ppo.capture_video = args.capture_video

    task, robot = VALID_CONFIGS[args.config_idx]
    pool_root = f"runs/Pool-cpu/{args.config_idx}_{task}_{robot}"
    pool = CheckpointPool(pool_root, max_per_bucket=args.max_per_bucket)

    if args.evaluate:
        evaluate_pool(pool, args, task, robot)
        return

    if args.storage is None:
        os.makedirs(pool_root, exist_ok=True)
        args.storage = f"sqlite:///{pool_root}/optuna.db"

    study = optuna.create_study(
        study_name=f"{task}_{robot}_search",
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.NopPruner(),
    )

    # 5. Enqueue baseline trial if it's a new study
    if len(study.trials) == 0:
        baseline_gamma = 0.8
        baseline_gae_lambda = 0.9
        # Check TASK_HYPERPARAMS
        if task in TASK_HYPERPARAMS:
            baseline_gamma = TASK_HYPERPARAMS[task].get("gamma", baseline_gamma)
            baseline_gae_lambda = TASK_HYPERPARAMS[task].get(
                "gae_lambda", baseline_gae_lambda
            )

        print(
            f"Enqueuing baseline trial: gamma={baseline_gamma}, gae_lambda={baseline_gae_lambda}"
        )
        study.enqueue_trial(
            {
                "gamma": baseline_gamma,
                "gae_lambda": baseline_gae_lambda,
            }
        )

    def stop_check(study, trial):
        if args.stop_on_full and pool.is_full():
            study.stop()

    study.optimize(
        lambda t: objective(t, args.config_idx, pool, args.ppo, args),
        n_trials=args.n_trials,
        callbacks=[stop_check],
    )


if __name__ == "__main__":
    main()

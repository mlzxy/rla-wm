import os
import torch.multiprocessing as mp
import argparse
import time
import torch
import numpy as np
import random
import importlib
from easydict import EasyDict as edict
import pickle
from torch.utils.data import DataLoader
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.spinner import Spinner
from rich.console import Group
from rich import print
import redis

from utils.misc import load_config, move_to_device
from datalib.dpipe import RedisQueue


def upload_worker(upload_queue, redis_kwargs):
    """
    Worker process to upload data to Redis in the background.
    """
    try:
        # Create separate Redis connection in this process
        redis_queue = RedisQueue(**redis_kwargs)
        print("[UploadWorker] Started. Waiting for data...")

        # Initial cleanup on startup (orphans from previous crashes)
        redis_queue.prune_orphaned_files(min_age_seconds=600)

        while True:
            batch = upload_queue.get()
            if batch is None:
                # Sentinel to stop
                break

            # Blocking put to Redis
            redis_queue.put_batch(batch, block=True)
            # Note: No periodic pruning here - file cleanup happens on consumer side
            # after successful read (in _unpack_and_restore). This avoids race conditions
            # where files are deleted before consumers can read them.

    except Exception as e:
        print(f"[UploadWorker] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("[UploadWorker] Stopping.")


@torch.inference_mode()
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Experiment config file (under runs/)"
    )
    parser.add_argument(
        "--data_config_override",
        type=str,
        required=True,
        help="usually is the config file for physics learning",
    )
    parser.add_argument(
        "--worker_module",
        type=str,
        default="src.datasets.data_worker.physics_transition_dataset",
        help="Module path to the worker implementation",
    )
    parser.add_argument("--debug", help="Debug mode", action="store_true")
    parser.add_argument(
        "--max_size", type=int, default=512, help="max redis queue size"
    )
    parser.add_argument("--host", type=str, default="", help="redis host")
    parser.add_argument(
        "--max_items",
        type=int,
        default=-1,
        help="Max items to process (-1 for infinite)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size (worker efficiency)",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    args = parser.parse_args()

    device = "cuda"

    # Setup random seed
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    print(f"Worker initializing with random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load worker module
    print(f"Loading worker module: {args.worker_module}")
    module = importlib.import_module(args.worker_module)
    worker_class = None
    from src.datasets.data_worker.base import DataWorker

    for attr in dir(module):
        val = getattr(module, attr)
        if (
            isinstance(val, type)
            and issubclass(val, DataWorker)
            and val is not DataWorker
        ):
            worker_class = val
            break

    if worker_class is None:
        raise ImportError(
            f"Could not find a DataWorker subclass in {args.worker_module}"
        )

    cfg = edict(load_config(args.config))
    if "output_dir" not in cfg:
        print(
            f"[yellow]Warning: output_dir not found in config, using config directory {os.path.dirname(args.config)}[/yellow]",
        )
        cfg.output_dir = os.path.dirname(args.config)
    data_override = edict(load_config(args.data_config_override))

    # Instantiate worker
    worker = worker_class(
        cfg=cfg,
        data_override=data_override,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        debug=args.debug,
    )

    dataset = worker.get_dataset()
    collate_fn = worker.get_collate_fn()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    console = Console()

    # Redis configuration usually comes from data_override
    dataset_args = data_override.dataset.args
    if args.host:
        if args.host != "localhost" and not args.host.endswith(".amarel.rutgers.edu"):
            args.host = args.host.split(".")[0] + ".amarel.rutgers.edu"
        dataset_args.host = args.host
        print(f"Overriding redis host to {args.host}")

    redis_host = dataset_args.host
    redis_port = dataset_args.port
    queue_name = dataset_args.queue_name
    image_keys = worker.get_image_keys()

    redis_kwargs = dict(
        host=redis_host,
        port=redis_port,
        queue_name=queue_name,
        image_keys=image_keys,
        max_size=args.max_size,
        password=dataset_args.password,
        shared_dir=dataset_args.shared_dir,
    )

    queue_proxy = None
    upload_process = None

    try:
        r_monitor = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=dataset_args.password
        )
        r_monitor.ping()
    except Exception as e:
        print(f"[yellow]Warning: Could not connect to Redis for monitoring: {e}[/yellow]")
        r_monitor = None

    if not args.debug:
        # Setup Multiprocessing Upload
        upload_queue = mp.Queue(maxsize=4)  # Small buffer
        upload_process = mp.Process(
            target=upload_worker, args=(upload_queue, redis_kwargs)
        )
        upload_process.start()
        queue_proxy = upload_queue
    else:
        # In debug mode, we might not need the upload process
        pass

    console.print(
        f"[bold green]Starting data worker loop...[/bold green] (Batch Size: {args.batch_size})"
    )
    count = 0
    total_samples = 0
    last_enqueue_time = 0.0

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                for batch in dataloader:
                    batch_start_time = time.time()

                    batch_cuda = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch_cuda[k] = v.to(device)
                        else:
                            batch_cuda[k] = v

                    # Process batch using the worker
                    process_start_time = time.time()
                    transitions = worker.process_batch(batch_cuda)
                    process_time = time.time() - process_start_time

                    if args.debug:
                        worker.visualize(transitions, count)

                        os.makedirs("runs", exist_ok=True)
                        save_path = "runs/debug_transitions.pkl"
                        with open(save_path, "wb") as f:
                            pickle.dump(transitions, f)
                        print(f"Saved {len(transitions)} debug samples to {save_path}")

                        for sample in transitions:
                            if not worker.check_integrity(sample):
                                print(f"Verification FAILED for a sample!")

                    # Batch put
                    upload_start_time = time.time()
                    cpu_transitions = []
                    for t in transitions:
                        cpu_transitions.append(move_to_device(t, "cpu"))

                    if args.debug:
                        print("Debug mode enabled. Exiting...")
                        return

                    if not args.debug:
                        try:
                            queue_proxy.put(
                                cpu_transitions, timeout=10
                            )  # 10s timeout to avoid infinite hang if worker dies
                        except Exception as e:
                            print(f"Failed to put to upload queue: {e}")

                    upload_time = time.time() - upload_start_time
                    total_samples += len(transitions)
                    last_enqueue_time = time.time()

                    count += 1

                    # Update Visualization
                    table = Table(box=box.ROUNDED, show_header=False)
                    table.add_column("Metric", style="cyan", no_wrap=True)
                    table.add_column("Value", style="magenta")

                    q_size_str = "N/A"
                    if hasattr(queue_proxy, "qsize"):
                        try:
                            q_size_str = f"{queue_proxy.qsize()}"
                        except:
                            pass

                    redis_q_size_str = "N/A"
                    if r_monitor is not None:
                        try:
                            redis_q_size_str = f"{r_monitor.llen(queue_name)}"
                        except Exception:
                            pass

                    table.add_row(
                        "Status", Spinner("dots", text="Running", style="green")
                    )
                    table.add_row(
                        "Total Time",
                        f"{(time.time() - batch_start_time) * 1000:.1f} ms",
                    )
                    table.add_row("Process Time", f"{process_time * 1000:.1f} ms")
                    table.add_row("Upload/Conv Time", f"{upload_time * 1000:.1f} ms")
                    table.add_row("Total Samples Processed", f"{total_samples}")
                    table.add_row("Upload Buffer Size", q_size_str)
                    table.add_row("Main Redis Queue Size", redis_q_size_str)

                    if last_enqueue_time > 0:
                        time_since = time.time() - last_enqueue_time
                        table.add_row("Time Since Last Enqueue", f"{time_since:.1f} s")
                    else:
                        table.add_row("Time Since Last Enqueue", "N/A")

                    renderable = table
                    if r_monitor is not None:
                        try:
                            monitor_keys = r_monitor.keys("local_queue:size:*")
                            local_stats = []
                            for key in monitor_keys:
                                try:
                                    key_str = key.decode("utf-8")
                                    parts = key_str.split(":")
                                    if len(parts) >= 4:
                                        host_val = parts[2]
                                        pid_val = parts[3]
                                        size_val = r_monitor.get(key).decode("utf-8")
                                        local_stats.append(
                                            {"host": host_val, "pid": pid_val, "size": size_val}
                                        )
                                except Exception:
                                    pass

                            local_stats.sort(
                                key=lambda x: (
                                    x["host"],
                                    int(x["pid"]) if x["pid"].isdigit() else x["pid"],
                                )
                            )

                            if local_stats:
                                t_local = Table(
                                    box=box.ROUNDED, show_header=True, title="Local Worker Queues"
                                )
                                t_local.add_column("Host", style="cyan")
                                t_local.add_column("PID", style="blue")
                                t_local.add_column("Size", style="magenta", justify="right")

                                for stat in local_stats:
                                    t_local.add_row(stat["host"], stat["pid"], stat["size"])
                                
                                renderable = Group(table, t_local)
                        except Exception:
                            pass

                    live.update(
                        Panel(
                            renderable,
                            title=worker.__class__.__name__,
                            border_style="blue",
                        )
                    )

                    if args.max_items > 0 and total_samples >= args.max_items:
                        console.print("[yellow]Max items reached.[/yellow]")
                        return

    except KeyboardInterrupt:
        console.print("[yellow]Stopping worker...[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error in worker loop:[/bold red] {e}")
        import traceback

        traceback.print_exc()
    finally:
        if upload_process and upload_process.is_alive():
            print("Terminating upload worker...")
            queue_proxy.put(None)  # Sentinel
            upload_process.join(timeout=5)
            if upload_process.is_alive():
                upload_process.terminate()


if __name__ == "__main__":
    main()

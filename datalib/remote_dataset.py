import torch
from torch.utils.data import Dataset, DataLoader
import time
from typing import Iterator, Optional, Any, Dict, List
from datalib.dpipe import RedisQueue
import pickle
import random
import os
from utils.misc import move_to_device
import threading
import queue
import atexit
import socket


class RemoteQueueDataset(Dataset):
    """
    Dataset that consumes data from a Redis queue populated by remote workers.

    This is a map-style Dataset where `__getitem__` ignores the index and returns
    the next available item from the queue. This design allows it to be used
    safely with standard DataLoaders (including `num_workers > 0`), where each
    worker process acts as a competing consumer for the Redis queue.

    Args:
        host (str): Redis server host.
        port (int): Redis server port.
        queue_name (str): Redis queue name.
        timeout (int): Timeout in seconds for blocking get.
        password (str): Redis password.
        image_keys (dict): Schema for image compression (passed to RedisQueue).
                           Format: {'rgb': [...], 'monochrome': [...]}
        pseudo_size (int): Virtual size of the dataset to control epoch length.
        debug_mode (bool): If True, load data from a local pickle file instead of Redis.
        debug_pickle_path (str): Path to the pickle file to use in debug mode.
        shared_dir (str): Optional directory for offloading large payloads.
    """

    def __init__(
        self,
        _,
        host: str = "localhost",
        port: int = 6379,
        queue_name: str = "pytorch_data_queue",
        timeout: int = 10,
        password: str = None,
        image_keys: Optional[Dict[str, List[str]]] = None,
        pseudo_size: int = 100000,
        debug_mode: bool = False,
        debug_pickle_path: Optional[str] = None,
        prefetch_size: int = 64,
        shared_dir: Optional[str] = None,
        reading_mode: bool = True,
        kwargs: dict = {},
    ):
        self.host = host
        self.port = port
        self.queue_name = queue_name
        self.timeout = timeout
        self.password = password
        self.image_keys = image_keys
        self.pseudo_size = pseudo_size
        self.queue = None
        self.debug_mode = debug_mode
        self.debug_pickle_path = debug_pickle_path
        self.local_data = None
        self.prefetch_size = prefetch_size
        self.shared_dir = shared_dir
        self.reading_mode = reading_mode

        # Threaded prefetching resources (initialized lazily)
        self.local_queue = None
        self._thread = None
        self._stop_event = None
        self._lock = None

        if self.debug_mode:
            if not self.debug_pickle_path or not os.path.exists(self.debug_pickle_path):
                raise ValueError(
                    f"debug_mode is True, but debug_pickle_path '{self.debug_pickle_path}' is invalid."
                )
            # We defer loading to _connect/lazy loading or here?
            # If we load here, it will be pickled to workers.
            # Ideally each worker loads it or we use shared memory, but for debug simple loading is fine.
            with open(self.debug_pickle_path, "rb") as f:
                self.local_data = [
                    move_to_device(sample, "cpu") for sample in pickle.load(f)
                ]

            print(
                f"[RemoteQueueDataset] DEBUG MODE: Loaded {len(self.local_data)} samples from {self.debug_pickle_path}."
            )

    def _connect(self):
        """Connect to Redis queue."""
        if self.debug_mode:
            return

        if self.queue is None:
            self.queue = RedisQueue(
                host=self.host,
                port=self.port,
                queue_name=self.queue_name,
                password=self.password,
                image_keys=self.image_keys,
                shared_dir=self.shared_dir,
                reading_mode=self.reading_mode,
            )

    def _prefetch_loop(self):
        """Background thread loop to prefetch data."""
        # Create a separate connection for the thread if needed,
        # but RedisQueue uses a thread-safe Redis client usually.
        # However, to be safe and avoid fork issues if any, we ensure connection here too.
        self._connect()

        # Monitoring setup
        hostname = socket.gethostname()
        pid = os.getpid()
        monitor_key = f"local_queue:size:{hostname}:{pid}"
        last_monitor_time = 0
        monitor_interval = 1.0  # Update every 1 second

        while not self._stop_event.is_set():
            # Monitoring heartbeat
            try:
                now = time.time()
                if now - last_monitor_time > monitor_interval:
                    qsize = self.local_queue.qsize()
                    # Set with TTL (e.g. 5s) so dead workers expire
                    self.queue.r.set(monitor_key, qsize, ex=5)
                    last_monitor_time = now
            except Exception:
                pass  # Ignore monitoring errors

            try:
                if self.local_queue.full():
                    time.sleep(0.01)
                    continue

                # Fetch batch
                # We block for short time on Redis to allow checking stop_event frequently
                items = self.queue.get_batch(self.prefetch_size, block=True, timeout=1)

                if items:
                    for item in items:
                        while not self._stop_event.is_set():
                            try:
                                self.local_queue.put(item, timeout=0.1)
                                break
                            except queue.Full:
                                pass

            except Exception as e:
                print(
                    f"[RemoteQueueDataset] Prefetch thread error (sleep a while): {e}"
                )
                time.sleep(0.5)

    def _start_prefetch(self):
        """Start the prefetch thread if not running."""
        # Lazy initialization for worker process
        if self.local_queue is None:
            self.local_queue = queue.Queue(maxsize=self.prefetch_size * 2)
            self._stop_event = threading.Event()
            self._lock = threading.Lock()

        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
                self._thread.start()

    def __getitem__(self, index: int) -> Any:
        """
        Get an item from the queue. Index is ignored.
        """
        if self.debug_mode:
            return random.choice(self.local_data)

        self._start_prefetch()

        while True:
            try:
                # Wait for item from local queue
                return self.local_queue.get(timeout=self.timeout)
            except queue.Empty:
                print(
                    f"[RemoteQueueDataset] Warning: Local Queue empty for {self.timeout}s (Redis starvation?)."
                )

    def __len__(self):
        """
        Return pseudo size.
        """
        return self.pseudo_size

    def clone(self):
        """
        Create a fresh copy of the dataset without active thread locks.
        """
        return RemoteQueueDataset(
            None,
            host=self.host,
            port=self.port,
            queue_name=self.queue_name,
            timeout=self.timeout,
            password=self.password,
            image_keys=self.image_keys,
            pseudo_size=self.pseudo_size,
            debug_mode=self.debug_mode,
            debug_pickle_path=self.debug_pickle_path,
            prefetch_size=self.prefetch_size,
            shared_dir=self.shared_dir,
            reading_mode=self.reading_mode,
        )

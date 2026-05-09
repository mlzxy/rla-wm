import io
import pickle
import time
from typing import Any, Dict, List, Optional, Union, Tuple, Set
import os
import uuid
import glob

import torch
import numpy as np
import redis
from PIL import Image
from rich import print
import socket
import concurrent.futures

# Try importing lz4 for compression, fallback to None if not available
try:
    import lz4.frame

    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


class RedisQueue:
    """
    A simple Redis-based queue for distributed data loading.

    Args:
        host (str): Redis server host.
        port (int): Redis server port.
        queue_name (str): Name of the list key in Redis.
        max_size (int): Maximum number of items in the queue (0 for infinite).
        db (int): Redis database index.
        password (str): Redis password.
        image_keys (dict): Dictionary defining keys to compress as images.
                           Format: {'rgb': ['key1', ...], 'monochrome': ['key2', ...]}
        shared_dir (str): Optional directory for offloading large payloads to disk.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        queue_name: str = "pytorch_data_queue",
        max_size: int = 1024,
        db: int = 0,
        password: str = None,
        image_keys: Optional[Dict[str, List[str]]] = None,
        shared_dir: Optional[str] = None,
        reading_mode: bool = False,
    ):
        self.r = redis.Redis(host=host, port=port, db=db, password=password)
        self.queue_name = queue_name
        self.max_size = max_size
        self.image_keys = image_keys if image_keys else {"rgb": [], "monochrome": []}
        self.shared_dir = shared_dir

        # Worker-scoped isolation
        # Use uuid to ensure uniqueness even if multiple workers start on the same host/pid simultaneously
        self.worker_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.storage_dir = None
        self._put_executor = None
        if self.shared_dir and not reading_mode:
            self.storage_dir = os.path.join(
                self.shared_dir, self.queue_name, self.worker_id
            )
            os.makedirs(self.storage_dir, exist_ok=True)
            print(f"[bold blue]Using worker storage: {self.storage_dir}[/bold blue]")

        # Verify connection
        try:
            self.r.ping()
        except redis.ConnectionError:
            print(f"Warning: Could not connect to Redis at {host}:{port}")

    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        return self.r.llen(self.queue_name)

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def full(self) -> bool:
        """Return True if the queue is full, False otherwise."""
        if self.max_size <= 0:
            return False
        return self.qsize() >= self.max_size

    def put(
        self,
        item: Dict[str, Any],
        image_keys: Optional[Dict[str, List[str]]] = None,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Put an item into the queue.
        """
        return self.put_batch([item], image_keys, block, timeout)

    def put_batch(
        self,
        items: List[Dict[str, Any]],
        image_keys: Optional[Dict[str, List[str]]] = None,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Put a batch of items into the queue.
        """
        keys = image_keys if image_keys is not None else self.image_keys

        packed_items = [None] * len(items)

        def pack_single_item(idx, item):
            # 1. Prepare (compress/cpu move)
            prepared = prepare_batch(item, keys)

            if self.storage_dir:
                # 2a. Disk Offload
                uid = str(uuid.uuid4())
                # Use relative path in Redis to allow worker-scoping
                rel_path = f"{self.worker_id}/{uid}.pt"
                file_path = os.path.join(self.storage_dir, f"{uid}.pt")
                # Use torch.save for the prepared, CPU-resident dict
                # Write to Memory buffer first, then atomic NFS write
                buffer = io.BytesIO()
                torch.save(prepared, buffer)
                with open(file_path + ".tmp", "wb") as f:
                    f.write(buffer.getvalue())
                os.rename(file_path + ".tmp", file_path)

                # Push reference
                ref_item = {"__disk_ref__": rel_path}
                packed_items[idx] = pickle.dumps(ref_item)
            else:
                # 2b. Standard Pickle + Compression
                data = pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL)
                if HAS_LZ4:
                    data = lz4.frame.compress(data, compression_level=0)
                packed_items[idx] = data

        if self.storage_dir:
            if self._put_executor is None:
                self._put_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(items), 16)
                )
            list(self._put_executor.map(lambda arg: pack_single_item(*arg), enumerate(items)))
        else:
            # Fallback to sequential if no storage dir (unlikely for big batches, but safe)
            for idx, item in enumerate(items):
                pack_single_item(idx, item)

        if self.max_size <= 0:
            self.r.rpush(self.queue_name, *packed_items)
            return True

        start_time = time.time()
        while True:
            # Check size
            if self.r.llen(self.queue_name) < self.max_size:
                self.r.rpush(self.queue_name, *packed_items)
                return True

            if not block:
                return False  # Queue Full

            if timeout is not None and (time.time() - start_time) > timeout:
                return False  # Timeout

            time.sleep(0.01)  # Wait a bit

    def get(self, block: bool = True, timeout: Optional[int] = None) -> Any:
        """
        Get an item from the queue.
        """
        items = self.get_batch(1, block, timeout)
        if items:
            return items[0]
        return None

    def get_batch(
        self, count: int, block: bool = True, timeout: Optional[int] = None
    ) -> List[Any]:
        """
        Get a batch of items from the queue.
        """
        if count <= 0:
            return []

        items_data = []

        if block:
            t = timeout if timeout is not None else 0
            # blpop returns (key, val)
            first = self.r.blpop(self.queue_name, timeout=t)
            if first is None:
                return []
            items_data.append(first[1])
            remaining = count - 1
        else:
            remaining = count

        if remaining > 0:
            # Pipeline the rest
            pipe = self.r.pipeline()
            for _ in range(remaining):
                pipe.lpop(self.queue_name)
            results = pipe.execute()

            # Filter out Nones (empty queue)
            for res in results:
                if res is not None:
                    items_data.append(res)

        final_items = []
        for data in items_data:
            try:
                final_items.append(self._unpack_and_restore(data))
            except Exception as e:
                print(f"[RedisQueue] Warning: Skipped item due to error: {e}")

        return final_items

    def _unpack_and_restore(self, data: bytes) -> Any:
        """
        Internal helper to handle redis payload -> pickle/lz4 -> disk load? -> restore
        """
        # 1. Compression/Pickle Layer
        # Note: If self.shared_dir is set, we expect a pickled dict (ref), not lz4 compressed usually.
        # However, to be robust, we try lz4 if available.
        if HAS_LZ4:
            try:
                decoded = lz4.frame.decompress(data)
                data = decoded
            except (RuntimeError, ValueError):
                # Not lz4 or error
                pass

        try:
            item = pickle.loads(data)
        except pickle.UnpicklingError:
            # Fallback or re-raise
            raise

        # 2. Disk Ref Layer
        if isinstance(item, dict) and "__disk_ref__" in item:
            rel_path = item["__disk_ref__"]
            if not self.shared_dir:
                raise ValueError(
                    "Received disk reference from queue but shared_dir is not configured."
                )

            # 1. Try relative path (Scoped behavior)
            file_path = os.path.join(self.shared_dir, self.queue_name, rel_path)
            if not file_path.endswith(".pt"):
                file_path += ".pt"

            # 2. Try legacy path (Legacy behavior for backward compatibility)
            if not os.path.exists(file_path):
                # If rel_path is just a UID (no slash), it's legacy
                if "/" not in rel_path:
                    file_path = os.path.join(self.shared_dir, rel_path)
                    if not file_path.endswith(".pt"):
                        file_path += ".pt"

            # Wait infinitely for the file to appear (handles NFS latency)
            wait_start = time.time()
            last_warn_time = wait_start
            while not os.path.exists(file_path):
                time.sleep(0.5)
                now = time.time()
                if now - last_warn_time > 60:
                    print(
                        f"[RedisQueue] [bold yellow]Warning:[/bold yellow] Still waiting for offloaded file (elapsed {now - wait_start:.1f}s): {file_path}"
                    )
                    last_warn_time = now

            for attempt in range(5):
                try:
                    with open(file_path, "rb") as f:
                        buffer = io.BytesIO(f.read())
                    item = torch.load(buffer)
                    break
                except Exception as e:
                    if attempt < 4:
                        time.sleep(1.0)
                    else:
                        print(f"[RedisQueue] Error loading torch file {file_path} after 5 attempts: {e}")
                        raise e

            # Delete file
            try:
                os.remove(file_path)
            except OSError:
                pass  # Already deleted?

        # 3. Restoration Layer
        return restore_batch(item, self.image_keys)

    def clear(self, clear_disk: bool = False):
        """Clear the queue. Optionally clear disk files in shared_dir."""
        self.r.delete(self.queue_name)
        if clear_disk and self.shared_dir:
            self.cleanup_all_files()

    def cleanup_all_files(self):
        """Delete all .pt files in shared_dir."""
        if not self.shared_dir:
            return

        # Safety check: ensure we are not deleting root or something crazy
        # shared_dir is managed by us.
        files = glob.glob(os.path.join(self.shared_dir, "*.pt"))
        print(f"[RedisQueue] Cleaning up {len(files)} files in {self.shared_dir}...")
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

    def prune_orphaned_files(self, min_age_seconds: int = 60):
        """
        Delete files in self.storage_dir that are NOT referenced by the current Redis queue.
        Aborts entirely if ANY items in the queue fail to parse to prevent accidental deletion.
        Uses a Redis lock to ensure only one worker prunes at a time.
        """
        if not self.storage_dir:
            return

        # 1. Try to acquire pruning lock (per queue)
        lock_key = f"lock:prune:{self.queue_name}"
        if not self.r.set(lock_key, 1, nx=True, ex=30):
            # Another worker is already pruning or recently pruned
            return

        try:
            print(
                f"[RedisQueue] Starting isolated pruning (worker={self.worker_id}, min_age={min_age_seconds}s)..."
            )

            active_refs = set()

            # 2. Gather all active IDs from Redis
            try:
                # Optimized: We only need to check refs that belong to THIS worker
                items = self.r.lrange(self.queue_name, 0, -1)
                for data in items:
                    try:
                        if HAS_LZ4:
                            try:
                                data = lz4.frame.decompress(data)
                            except Exception:
                                pass

                        item = pickle.loads(data)
                        if isinstance(item, dict) and "__disk_ref__" in item:
                            active_refs.add(item["__disk_ref__"])
                    except Exception as e:
                        # CRITICAL: If we fail to parse even one item, we cannot guarantee safety.
                        # Abort pruning entirely.
                        print(
                            f"[RedisQueue] Pruning ABORTED: Failed to parse queue item: {e}"
                        )
                        return
            except Exception as e:
                print(f"[RedisQueue] Pruning error reading redis: {e}")
                return

            print(f"[RedisQueue] Found {len(active_refs)} active references in queue.")

            # 3. Scan ONLY this worker's directory
            files = glob.glob(os.path.join(self.storage_dir, "*.pt"))
            deleted_count = 0
            now = time.time()

            for f in files:
                basename = os.path.basename(f)
                # Reconstruct the relative path used in Redis
                rel_path = f"{self.worker_id}/{basename}"

                if rel_path not in active_refs:
                    try:
                        stats = os.stat(f)
                        mtime = stats.st_mtime
                        if now - mtime > min_age_seconds:
                            os.remove(f)
                            deleted_count += 1
                    except OSError:
                        pass

            if deleted_count > 0:
                print(
                    f"[RedisQueue] Pruned {deleted_count} orphaned files from {self.worker_id}."
                )

        finally:
            # Release lock
            self.r.delete(lock_key)


# --- Simplified Packing Logic (No Heuristics) ---


def _compress_png(tensor: torch.Tensor, mode: str) -> bytes:
    """
    Compress a tensor to PNG bytes. Stack usage for arbitrary shapes.
    mode: 'L' (monochrome) or 'RGB'.
    """
    tensor = tensor.detach().cpu()

    shape = tensor.shape

    if mode == "RGB":
        # Assume (..., 3, H, W). Permute to (..., H, W, 3) for stacking
        if tensor.shape[-3] == 3:
            # Move C to end
            dims = list(range(tensor.ndim))
            dims[-3], dims[-2], dims[-1] = dims[-2], dims[-1], dims[-3]
            tensor = tensor.permute(*dims)

        # Now tensor is (..., H, W, 3)
        H, W = tensor.shape[-3], tensor.shape[-2]
    else:
        # Mono 'L'
        if len(shape) >= 3 and tensor.shape[-3] == 1:
            tensor = tensor.squeeze(-3)

        H, W = tensor.shape[-2], tensor.shape[-1]

    if mode == "RGB":
        num_images = int(np.prod(tensor.shape[:-3]))
        t_view = tensor.reshape(num_images * H, W, 3)
        t_np = t_view.numpy()  # uint8
        img = Image.fromarray(t_np, mode="RGB")
    else:
        # Mono
        num_images = int(np.prod(tensor.shape[:-2]))
        t_view = tensor.reshape(num_images * H, W)
        t_np = t_view.numpy()
        img = Image.fromarray(t_np, mode="L")

    with io.BytesIO() as bio:
        img.save(bio, format="PNG", optimize=True)
        return bio.getvalue()


def _decompress_png(
    data: bytes, original_shape: tuple, original_dtype: torch.dtype, mode: str
) -> torch.Tensor:
    """Decompress PNG bytes to tensor with original shape."""
    with io.BytesIO(data) as bio:
        img = Image.open(bio)
        t_np = np.array(img)

    if mode == "RGB":
        tensor = torch.from_numpy(t_np)  # (TotalH, W, 3)

        # Check if original had 3 at -3 or -1
        if original_shape[-1] == 3:
            # (..., H, W, 3)
            tensor = tensor.view(original_shape)
        else:
            # (..., 3, H, W) - we stored as (..., H, W, 3)
            # Reshape to (..., H, W, 3)
            temp_shape = list(original_shape)
            temp_shape[-3], temp_shape[-2], temp_shape[-1] = (
                temp_shape[-2],
                temp_shape[-1],
                temp_shape[-3],
            )
            tensor = tensor.view(*temp_shape)
            # Permute back to (..., 3, H, W)
            dims = list(range(tensor.ndim))
            # (..., H, W, C) -> (..., C, H, W)
            dims[-3], dims[-2], dims[-1] = dims[-1], dims[-3], dims[-2]
            tensor = tensor.permute(*dims)

    else:
        # Mono
        tensor = torch.from_numpy(t_np)  # (TotalH, W)
        tensor = tensor.view(original_shape)

    return tensor.to(dtype=original_dtype)


def _process_value(
    v: Any, key: Optional[str], rgb_keys: Set[str], mono_keys: Set[str]
) -> Any:
    # Compression check
    if key is not None and isinstance(v, torch.Tensor):
        if key in rgb_keys:
            return (_compress_png(v, "RGB"), v.shape, v.dtype)
        if key in mono_keys:
            return (_compress_png(v, "L"), v.shape, v.dtype)

    if isinstance(v, dict):
        return {k: _process_value(val, k, rgb_keys, mono_keys) for k, val in v.items()}

    if isinstance(v, (list, tuple)):
        # Recurse
        empty = set()
        return type(v)(_process_value(x, None, empty, empty) for x in v)

    if isinstance(v, torch.Tensor):
        return v.detach().cpu()

    return v


def prepare_batch(
    item: Dict[str, Any], image_keys: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Prepare item for serialization: compress images, move to CPU.
    Returns the dictionary ready to be saved/pickled.
    """
    rgb_keys = set(image_keys.get("rgb", []))
    mono_keys = set(image_keys.get("monochrome", []))

    return {k: _process_value(v, k, rgb_keys, mono_keys) for k, v in item.items()}


def pack_batch(item: Dict[str, Any], image_keys: Dict[str, List[str]]) -> bytes:
    """
    Pack a dictionary item (Legacy/Direct use).
    """
    prepared = prepare_batch(item, image_keys)

    # Pickle
    data = pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL)

    if HAS_LZ4:
        return lz4.frame.compress(data, compression_level=0)
    return data


def restore_batch(
    item: Dict[str, Any], image_keys: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Restore item from its prepared state (decompress images).
    """
    rgb_keys = set(image_keys.get("rgb", []))
    mono_keys = set(image_keys.get("monochrome", []))

    # Recursive unpack
    return _unpack_recursive_dict(item, rgb_keys, mono_keys)


def unpack_batch(data: bytes, image_keys: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Unpack data using the provided schema.
    """
    if HAS_LZ4:
        try:
            data = lz4.frame.decompress(data)
        except RuntimeError:
            pass

    item = pickle.loads(data)
    return restore_batch(item, image_keys)


def _unpack_recursive_dict(d: dict, rgb_keys: set, mono_keys: set):
    new_d = {}
    for k, v in d.items():
        if k in rgb_keys or k in mono_keys:
            # Expect tuple (bytes, shape, dtype)
            if isinstance(v, tuple) and len(v) == 3 and isinstance(v[0], bytes):
                mode = "RGB" if k in rgb_keys else "L"
                new_d[k] = _decompress_png(v[0], v[1], v[2], mode)
            else:
                new_d[k] = v
        elif isinstance(v, dict):
            new_d[k] = _unpack_recursive_dict(v, rgb_keys, mono_keys)
        else:
            new_d[k] = v
    return new_d


# Helper for backward compatibility or simple usage if needed
def pack_object(obj, image_keys=None):
    if image_keys:
        return pack_batch(obj, image_keys)
    data = pickle.dumps(obj)
    if HAS_LZ4:
        return lz4.frame.compress(data, compression_level=0)
    return data


def unpack_object(data):
    if HAS_LZ4:
        try:
            data = lz4.frame.decompress(data)
        except:
            pass
    return pickle.loads(data)


if __name__ == "__main__":
    print("Running dpipe self-test (Schema Mode)...")

    keys = {"rgb": ["color"], "monochrome": ["depth", "mask"]}

    # Test Normal
    queue = RedisQueue(queue_name="test_schema", image_keys=keys)

    data = {
        "color": torch.randint(0, 255, (10, 3, 64, 64), dtype=torch.uint8),
        "depth": torch.randint(0, 255, (10, 64, 64), dtype=torch.uint8),
        "meta": {"info": "test"},
    }

    print("Putting...")
    queue.clear()
    queue.put(data)

    print("Getting...")
    retrieved = queue.get()

    assert torch.equal(retrieved["color"], data["color"])
    assert torch.equal(retrieved["depth"], data["depth"])
    assert retrieved["meta"] == data["meta"]

    print("Schema Test Passed!")

    # Test Offload
    print("\nTesting Disk Offload...")
    import shutil

    shared_tmp = "/tmp/redis_shared_test"
    if os.path.exists(shared_tmp):
        shutil.rmtree(shared_tmp)

    queue_offload = RedisQueue(
        queue_name="test_offload", image_keys=keys, shared_dir=shared_tmp
    )
    queue_offload.clear()

    queue_offload.put(data)

    # Check if file exists
    files = os.listdir(shared_tmp)
    print(f"Files in shared dir: {len(files)}")
    assert len(files) == 1

    retrieved_offload = queue_offload.get()

    assert torch.equal(retrieved_offload["color"], data["color"])
    assert torch.equal(retrieved_offload["depth"], data["depth"])
    assert retrieved_offload["meta"] == data["meta"]

    # Check if deleted
    files_after = os.listdir(shared_tmp)
    print(f"Files after get: {len(files_after)}")
    assert len(files_after) == 0

    print("Offload Test Passed!")

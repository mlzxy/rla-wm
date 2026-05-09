from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from datalib.dpipe import pack_batch, unpack_batch
from src.modules.sparse import SparseTensor
from src.datasets.trajectory_dataset import TrajectoryDataset
from datalib.remote_dataset import RemoteQueueDataset


def check_data_integrity_helper(sample: Dict[str, Any], image_keys: List[str]) -> bool:
    """
    Verify that pack_batch and unpack_batch preserve data integrity.

    Args:
        sample: The sample dictionary to verify.
        image_keys: List of keys in the sample that represent image data.

    Returns:
        True if the sample is preserved exactly after packing and unpacking, False otherwise.
    """
    packed = pack_batch(sample, image_keys)
    unpacked = unpack_batch(packed, image_keys)

    def check_equal(o1: Any, o2: Any, path: str = "") -> bool:
        if type(o1) is not type(o2):
            print(f"Type mismatch at {path}: {type(o1)} != {type(o2)}")
            return False
        if isinstance(o1, torch.Tensor):
            o1 = o1.cpu()
            o2 = o2.cpu()
            if o1.dtype != o2.dtype:
                print(f"Dtype mismatch at {path}: {o1.dtype} != {o2.dtype}")
                return False
            if o1.shape != o2.shape:
                print(f"Shape mismatch at {path}: {o1.shape} != {o2.shape}")
                return False
            if not torch.equal(o1, o2):
                diff = (o1.float() - o2.float()).abs().max()
                if diff > 1e-6:
                    print(f"Tensor value mismatch at {path}: diff={diff}")
                    return False
            return True
        elif isinstance(o1, SparseTensor):
            if not check_equal(o1.coords, o2.coords, path + ".coords"):
                return False
            if not check_equal(o1.feats, o2.feats, path + ".feats"):
                return False
            return True
        elif isinstance(o1, dict):
            if set(o1.keys()) != set(o2.keys()):
                print(f"Dict keys mismatch at {path}: {o1.keys()} != {o2.keys()}")
                return False
            for k in o1:
                if not check_equal(o1[k], o2[k], path + f"[{k}]"):
                    return False
            return True
        elif isinstance(o1, (list, tuple)):
            if len(o1) != len(o2):
                print(f"List/Tuple length mismatch at {path}: {len(o1)} != {len(o2)}")
                return False
            for i, (e1, e2) in enumerate(zip(o1, o2)):
                if not check_equal(e1, e2, path + f"[{i}]"):
                    return False
            return True
        else:
            if o1 != o2:
                print(f"Value mismatch at {path}: {o1} != {o2}")
                return False
            return True

    return check_equal(sample, unpacked)


class DataWorker(ABC):
    """
    Abstract base class for data workers.
    Each worker implementation should inherit from this class and implement the abstract methods.
    """

    @abstractmethod
    def __init__(
        self,
        cfg: edict,
        data_override: edict,
        device: Union[str, torch.device],
        batch_size: int,
        num_workers: int,
        debug: bool = False,
    ):
        """
        Initialize the worker with configurations and parameters.

        Args:
            cfg: Experiment configuration.
            data_override: Data configuration override.
            device: Device to run computations on.
            batch_size: Batch size for inference.
            num_workers: Number of workers for DataLoader.
            debug: Whether to run in debug mode.
        """
        pass

    def get_dataset(self) -> Dataset:
        """
        Return the source dataset used by this worker.
        """
        return self.dataset

    def get_collate_fn(self) -> Any:
        """
        Return the collate function for the DataLoader.
        """
        return TrajectoryDataset.collate_fn

    @abstractmethod
    def process_batch(self, batch_cuda: Dict[str, Any]) -> List[Any]:
        """
        Process a batch of data on the GPU and return a list of samples.

        Args:
            batch_cuda: A dictionary of tensors already moved to the target device.

        Returns:
            A list of generated samples (e.g., TransitionSamples).
        """
        pass

    def check_integrity(self, sample: Any) -> bool:
        """
        Verify the integrity of a generated sample.
        Default implementation uses a generic deep comparison helper.

        Args:
            sample: The sample to check.

        Returns:
            True if integrity is preserved, False otherwise.
        """
        return check_data_integrity_helper(sample, self.get_image_keys())

    @abstractmethod
    def visualize(self, samples: List[Any], count: int) -> None:
        """
        Visualize the generated samples (used in debug mode).

        Args:
            samples: List of generated samples.
            count: Current batch count for unique naming.
        """
        pass

    def get_image_keys(self) -> List[str]:
        """
        Return the list of keys corresponding to image data in the samples.
        These keys will be handled specially by the Redis uploader (e.g., PNG compression).
        """
        return self.image_keys


class SimpleTransitionDataset(RemoteQueueDataset):
    """
    Dataset class for remote queue transitions.
    """

    value_range = (0.0, 1.0)

    @staticmethod
    def collate_fn(samples: List[dict]) -> dict:
        """
        Custom collate function to handle SparseTensors in a batch.
        """
        if not samples:
            return {}

        batch = {}
        for key in samples[0].keys():
            value = samples[0][key]
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], torch.Tensor)
                and isinstance(value[1], torch.Tensor)
            ):
                # SparseTensor case: (coords, feats)
                list_of_tuples = [s[key] for s in samples]
                batched_coords = []
                batched_feats = []
                for i, (coords, feats) in enumerate(list_of_tuples):
                    # coords: (P, 3) -> (P, 4) with batch index at 0
                    batch_idx = torch.full(
                        (coords.shape[0], 1),
                        i,
                        dtype=coords.dtype,
                        device=coords.device,
                    )
                    batched_coords.append(torch.cat([batch_idx, coords], dim=1))
                    batched_feats.append(feats)

                batch[key] = SparseTensor(
                    coords=torch.cat(batched_coords, dim=0),
                    feats=torch.cat(batched_feats, dim=0),
                )
            elif isinstance(value, str):
                batch[key] = [s[key] for s in samples]
            elif isinstance(value, torch.Tensor):
                try:
                    batch[key] = torch.stack([s[key] for s in samples])
                except Exception:
                    # In case of inconsistent shapes, fallback to list
                    batch[key] = [s[key] for s in samples]
            elif isinstance(value, list) and len(value) > 0:
                # Handle lists of features (structure, unstructure)
                if (
                    isinstance(value[0], tuple)
                    and len(value[0]) == 2
                    and isinstance(value[0][0], torch.Tensor)
                ):
                    # list of (coords, feats) tuples (structure)
                    num_frames = len(value)
                    batched_frames = []
                    for t in range(num_frames):
                        list_of_tuples = [s[key][t] for s in samples]
                        batched_coords = []
                        batched_feats = []
                        for i, (coords, feats) in enumerate(list_of_tuples):
                            batch_idx = torch.full(
                                (coords.shape[0], 1),
                                i,
                                dtype=coords.dtype,
                                device=coords.device,
                            )
                            batched_coords.append(torch.cat([batch_idx, coords], dim=1))
                            batched_feats.append(feats)
                        batched_frames.append(
                            SparseTensor(
                                coords=torch.cat(batched_coords, dim=0),
                                feats=torch.cat(batched_feats, dim=0),
                            )
                        )
                    batch[key] = batched_frames
                elif isinstance(value[0], torch.Tensor):
                    # list of Tensors (unstructure)
                    num_frames = len(value)
                    batched_frames = []
                    for t in range(num_frames):
                        try:
                            batched_frames.append(
                                torch.stack([s[key][t] for s in samples])
                            )
                        except Exception:
                            batched_frames.append([s[key][t] for s in samples])
                    batch[key] = batched_frames
                else:
                    batch[key] = [s[key] for s in samples]
            else:
                # Fallback to list
                batch[key] = [s[key] for s in samples]

        return batch

import os
import pdb
import pickle
import random
import sys
sys.path.append(os.getcwd() + '/data_preparation')
from helper import get_path_training_data
from typing import Callable, List, Optional, Tuple, Any, TYPE_CHECKING

import logging
logger = logging.getLogger("qet-predictor")

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import scipy.signal
import torch
from torchpack.datasets.dataset import Dataset

__all__ = ["CircDataset", "Circ"]

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

if TYPE_CHECKING:
    from numpy._typing import NDArray


random.seed(1234)


def load_training_data():
    """Loads and returns the training data from the training data folder.
    """
    file = open(Path(os.getcwd()) / "datav1" / "training_data_log.npy", "rb")
    training_data = pickle.load(file)
    file.close()
    return training_data


class CircDataset:
    def __init__(self, split_ratio: List[float], shuffle=True):
        super().__init__()
        self.split_ratio = split_ratio
        self.raw = {}
        self.mean = {}
        self.std = {}

        self.shuffle = shuffle

        self._load()
        self._preprocess()
        self._split()

        self.instance_num = len(self.raw["dataset"])
        

    def _load(self):
        self.raw["dataset"] = load_training_data()
        for data in self.raw["dataset"]:
            data.global_features = data.global_features.unsqueeze(0)


    def _preprocess(self):
        pass
        

    def _split(self):
        import numpy as np

        indices_file = Path(os.getcwd()) / "datav1" / "split_indices.pth"

        if indices_file.exists():
            saved = torch.load(str(indices_file), weights_only=True)
            train_idx = saved["train"]
            valid_idx = saved["valid"]
            test_idx  = saved["test"]
            print(f"Loaded split indices from {indices_file}", flush=True)
        else:
            y_vals  = np.array([float(d.y) for d in self.raw["dataset"]])
            n_bins  = 10
            bins    = np.percentile(y_vals, np.linspace(0, 100, n_bins + 1))
            bin_ids = np.digitize(y_vals, bins[1:-1])  # 0 .. n_bins-1

            train_idx, valid_idx, test_idx = [], [], []
            rng = np.random.default_rng(seed=1234)

            for b in range(n_bins):
                idxs = np.where(bin_ids == b)[0]
                if len(idxs) == 0:
                    continue
                rng.shuffle(idxs)
                n_train = max(1, int(self.split_ratio[0] * len(idxs)))
                n_valid = max(1, int(self.split_ratio[1] * len(idxs)))
                train_idx.extend(idxs[:n_train].tolist())
                valid_idx.extend(idxs[n_train : n_train + n_valid].tolist())
                test_idx.extend(idxs[n_train + n_valid :].tolist())

            torch.save({"train": train_idx, "valid": valid_idx, "test": test_idx}, str(indices_file))
            print(f"Saved split indices to {indices_file}", flush=True)

        dataset = self.raw["dataset"]
        self.raw["train"] = [dataset[i] for i in train_idx]
        self.raw["valid"] = [dataset[i] for i in valid_idx]
        self.raw["test"]  = [dataset[i] for i in test_idx]
        print(f"Split — train: {len(self.raw['train'])}  val: {len(self.raw['valid'])}  test: {len(self.raw['test'])}", flush=True)

    def get_data(self, device, split):
        return [data.to(device) for data in self.raw[split]]

    def __getitem__(self, index: int):
        data_this = {"dag": self.raw["dataset"][index]}
        return data_this

    def __len__(self) -> int:
        return self.instance_num


class Circ(Dataset):
    def __init__(
        self,
        root: str,
        split_ratio: List[float]
    ):
        self.root = root

        super().__init__(
            {
                split: CircDataset(
                    root=root,
                    split=split,
                    split_ratio=split_ratio,
                )
                for split in ["train", "valid", "test"]
                # for split in ["train", "test"]
            }
        )


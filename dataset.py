import os
import glob
import numpy as np
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore

class SDFDataset(Dataset):
    def __init__(self, root_dir: str):
        self.files = sorted(glob.glob(os.path.join(root_dir, "shape_*.npz")))
        if len(self.files) == 0:
            raise RuntimeError(f"No shape_*.npz files found in {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])

        pts = torch.from_numpy(data["pts"]).float()   # (N, 3)
        sdf = torch.from_numpy(data["sdf"]).float()   # (N,)
        sid = int(data["sid"])

        return {
            "pts": pts,
            "sdf": sdf,
            "sid": sid,
            "idx": idx,
        }
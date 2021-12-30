import torch
import numpy as np
from torch.utils.data import Dataset


class TechnoDataset(Dataset):
    def __init__(self, dat_location="/fast-1/tmp/techno.dat") -> None:
        super().__init__()

        self.samples = np.memmap(
            dat_location,
            dtype="float32",
            mode="r",
        ).reshape(-1, 32000)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(np.copy(self.samples[index])).float()
    
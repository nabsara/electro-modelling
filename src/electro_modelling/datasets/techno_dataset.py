import torch
import numpy as np
from torch.utils.data import Dataset


class TechnoDatasetWav(Dataset):
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
        # return torch.from_numpy(np.copy(self.samples[index])).float()
        return np.copy(self.samples[index])
    
    
class TechnoDatasetSpectrogram(Dataset):
    def __init__(self, tensors,transform=None):
        self.tensors = tensors
        self.transform = transform

    def __len__(self):
        return self.tensors.shape[0]

    def __getitem__(self, index):
        x = self.tensors[index]
        if self.transform:
            x = self.transform(x)
        return(x)

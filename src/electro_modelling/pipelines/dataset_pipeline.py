# -*- coding: utf-8 -*-

"""

"""

from electro_modelling.helpers.utils import save_pickle
from electro_modelling.datasets.techno_dataset import TechnoDatasetWav
from electro_modelling.datasets.signal_processing import SignalOperators

import torch
from tqdm import tqdm
import numpy as np


class TechnoDatasetPipeline:
    """

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(
        self,
        nfft=1024,
        sr=16000,
        dataset_location="/fast-1/tmp/techno.dat",
        save_location="/fast-1/tmp/techno_spectrograms.pkl",
    ):
        self.operator = SignalOperators(
            nfft=nfft,
            nmels=int(nfft / 2),
            sr=sr,
            ntimes=128,
            signal_length=32000,
            phase_rec_method="Griffin-Lim",
        )
        self.dataset = TechnoDatasetWav(dataset_location)
        self.save_location = save_location
        self.len_dataset = len(self.dataset)

    def process_dataset(self, nb_samples=-1):
        """

        Parameters
        ----------
        nb_samples

        Returns
        -------

        """
        spectrograms = []
        if nb_samples == -1:
            nb_samples = self.len_dataset
        for i in tqdm(range(nb_samples)):
            signal = self.dataset[i]
            STFT_mel = self.operator.forward(signal)
            spectrograms.append(STFT_mel)
        spectrograms = np.asarray(spectrograms).astype("float32")
        tensor_spectrograms = torch.tensor(spectrograms)

        save_pickle(tensor_spectrograms, self.save_location)

from electro_modelling.helpers.utils import save_pickle
from electro_modelling.datasets.techno_dataset import TechnoDatasetWav
from electro_modelling.datasets.signal_processing import SignalOperators

import torch
from tqdm import tqdm

class TechnoDatasetPipeline:
    def __init__(self, Nfft,Nmels,sr,dataset_location="/fast-1/tmp/techno.dat",save_location="/fast-1/tmp/techno_spectrograms.pkl"):
        self.operator = SignalOperators(Nfft,Nmels,sr)
        self.dataset = TechnoDatasetWav(dataset_location)
        self.save_location = save_location
        self.len_dataset = self.dataset.__len__()
        
    def process_dataset(self,nb_samples=None):
        spectrograms  = []
        if nb_samples == None:
            nb_samples = self.len_dataset
        for i in tqdm(range(nb_samples)):
            signal = self.dataset.__getitem__(i)
            STFT_mel = self.operator.forward(signal)
            spectrograms.append(STFT_mel)
        
        tensor_spectrograms = torch.tensor(spectrograms)
        
        save_pickle(tensor_spectrograms,self.save_location)
        
            
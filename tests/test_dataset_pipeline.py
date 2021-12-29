import sys
sys.path.append('./src/')

import os 
import tqdm
import torch
os.chdir(r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\electro-modelling\src')
from electro_modelling.pipelines.dataset_pipeline import TechnoDatasetPipeline
from electro_modelling.helpers.helpers_audio import plot_spectrogram
from electro_modelling.datasets.techno_dataloader import techno_data_loader
from electro_modelling.datasets.signal_processing import SignalOperators
from electro_modelling.helpers.helpers_audio import *
#Signal processing parameters


#File locations
data_path = r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\data\\'
dataset_location =  data_path + r'\techno.dat'


#Instanciate pipeline
# pipeline = TechnoDatasetPipeline(dataset_location=dataset_location,save_location=save_location)


# #Run the dataset process
# pipeline.process_dataset(nb_samples = 32)

# #Check if it worked

operator = SignalOperators(nfft =1024,nmels=128)
data_loader = techno_data_loader(4, dataset_location,operator,phase_method='griff')
mins,maxs=[],[]
imgs = next(iter(data_loader))

# for data in tqdm.tqdm(data_loader[:2]):
#         mins.append(torch.min(data))
#         maxs.append(torch.max(data))
        

# MAX : 2.2926
# MIN : -6
figure = image_grid_spectrograms(imgs)



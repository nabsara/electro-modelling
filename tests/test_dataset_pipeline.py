import sys
sys.path.append('./src/')

import os 
os.chdir(r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\electro-modelling\src')
from electro_modelling.pipelines.dataset_pipeline import TechnoDatasetPipeline
from electro_modelling.helpers.helpers_audio import plot_spectrogram
from electro_modelling.datasets.techno_dataloader import techno_data_loader


#Signal processing parameters


#File locations
data_path = r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\data\\'
dataset_location =  data_path + r'\techno.dat'
save_location = data_path + r'\techno_spectrograms.pkl'


#Instanciate pipeline
pipeline = TechnoDatasetPipeline(dataset_location=dataset_location,save_location=save_location)


#Run the dataset process
pipeline.process_dataset(nb_samples = 32)

# #Check if it worked
# data_loader = techno_data_loader(32,save_location)

# for i,data in enumerate(data_loader):
#     stft_mel = data[0]
#     plot_spectrogram(stft_mel[0],stft_mel[1],pipeline.operator.freqs,pipeline.operator.times)
    
    

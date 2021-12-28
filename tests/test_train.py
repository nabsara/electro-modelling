

import sys

import torch
sys.path.append('./src/')

import os 
import tqdm
os.chdir(r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\electro-modelling\src')

from electro_modelling.pipelines.techno_pipeline import TechnoPipeline
from torch.utils.tensorboard import SummaryWriter


#File locations


data_dir =r"C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\data"
dataset_dir = data_dir + r'\techno_spectrograms.pkl'
models_dir = data_dir
batch_size = 32 
z_dims = 256
model = "dcgan"
n_epochs = 1 
learning_rate = 0.0002 
k_disc_steps = 1
display_step = 5
show = True

# pipeline = TechnoPipeline( model, data_dir, dataset_dir,models_dir, batch_size, z_dims,phase_method = 'griff')
# pipeline.train(
#     learning_rate=learning_rate,
#     k_disc_steps=k_disc_steps,
#     n_epochs=n_epochs,
#     display_step=display_step,
#     show_fig=show
# )



from electro_modelling.models.dcgan import DCGAN
from electro_modelling.helpers.helpers_audio import *
from electro_modelling.datasets.signal_processing import SignalOperators
from electro_modelling.datasets.techno_dataloader import techno_data_loader

logdir = r"C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\electro-modelling\data\tensorboard"
file_writer = SummaryWriter(logdir)        


model = DCGAN(z_dims, model, init_weights=True,dataset='techno',img_chan=1)
sr = 16000
Nfft =1024
Nmels = int(Nfft/2)
operator = SignalOperators(Nfft,Nmels,sr)


noise = model.get_noise(2)

imgs = model.generator.forward(noise).detach().cpu()


# data_loader = techno_data_loader(4,dataset_dir)
# imgs = next(iter(data_loader))*10.05-3.76

imgs = imgs*10.05-3.76
sounds_tensor = model.get_sounds(imgs)
        
figure = image_grid_spectrograms(imgs)
            
file_writer.add_figure('Fig3', figure)
file_writer.add_audio('Audio_test4/1', sounds_tensor[0],global_step = 0,sample_rate = 16000)
file_writer.add_audio('Audio_test4/2', sounds_tensor[1],global_step = 0,sample_rate = 16000)

    
def get_sounds(self,fakes):
    sounds_list = []
    for i,fake in enumerate(fakes):
        STFT_mel_amp = fake[0].numpy()
        sound = self.operator.backward(STFT_mel_amp)
        sounds_list.append(torch.tensor(sound))
    sounds_tensor=torch.stack(sounds_list)
    return (sounds_tensor)
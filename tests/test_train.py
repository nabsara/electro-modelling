

import sys
sys.path.append('./src/')

import os 
import tqdm
os.chdir(r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\electro-modelling\src')

from electro_modelling.pipelines.techno_pipeline import TechnoPipeline


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

pipeline = TechnoPipeline( model, data_dir, dataset_dir,models_dir, batch_size, z_dims,phase_method = 'griff')
pipeline.train(
    learning_rate=learning_rate,
    k_disc_steps=k_disc_steps,
    n_epochs=n_epochs,
    display_step=display_step,
    show_fig=show
)



# from electro_modelling.models.dcgan import DCGAN
# from electro_modelling.helpers.helpers_audio import plot_spectrogram_mag

# model = DCGAN(z_dims, model, init_weights=True,dataset='techno',img_chan=1)


# noise = model.get_noise(10)

# imgs = model.generator.forward(noise).detach().cpu()
# plot_spectrogram_mag(imgs[0][0])
# # print(imgs.shape)
# # print(imgs.dtype)
# # labels = model.discriminator.forward(imgs)
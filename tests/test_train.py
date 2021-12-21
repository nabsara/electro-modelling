

import sys
sys.path.append('./src/')

# import os 
# import tqdm
# os.chdir(r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\electro-modelling\src')

from electro_modelling.pipelines.techno_pipeline import TechnoPipeline


#File locations


model_name = 'dcgan'
dataset_dir = data_dir + r'\techno_spectrograms.pkl'
data_dir = "./data"
models_dir = "./models" 
batch_size = 128 
z_dims = 256
model = "dcgan"
n_epochs = 1 
learning_rate = 0.0002 
k_disc_steps = 1
display_step = 500
show = False

pipeline = TechnoPipeline( model_name, data_dir, dataset_dir,models_dir, batch_size, z_dims,phase_method = 'griff')
pipeline.train(
    learning_rate=learning_rate,
    k_disc_steps=k_disc_steps,
    n_epochs=n_epochs,
    display_step=display_step,
    show_fig=show
)
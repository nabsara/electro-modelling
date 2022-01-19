import sys 
sys.path.append('./src/')
import os 
os.chdir(r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\electro-modelling\src')
from playsound import playsound
import sounddevice as sd
import threading

from electro_modelling.models.base import GAN
from electro_modelling.helpers.helpers_audio import *
from electro_modelling.datasets.signal_processing import SignalOperators
from electro_modelling.datasets.techno_dataloader import techno_data_loader
import torch

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation


sr = 16000
nfft =1024
nmels = int(nfft/8)
z_dims = 256
model = "wgan"
nb_samples=4

save_path = r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\Runs\models_test_git\generator__wgan_img_size_128_128__init_kernel_2_2_minibatch_std__z_256__lr_0.0001__k_5__e_20.pt'
gif_path = r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\Runs\models_test_git'

operator = SignalOperators(nfft,nmels,sr)
model = GAN(z_dims, model, init_weights=True,dataset='techno',img_chan=1,operator=operator)
generator = model.generator

#Load Generator
checkpoint = torch.load(save_path,map_location=torch.device('cpu'))
generator.load_state_dict(checkpoint['model_state_dict'])
model.generator=generator

#Sample
samples = model.get_noise(nb_samples)

fakes = model.generator.forward(samples).detach().cpu()
fakes = fakes[:,:,:,:121]
sounds = model.get_sounds(fakes)

imix = 400
for i in range(sounds.shape[0]-1):
    sounds[i,-imix:] = sounds[i,-imix:] + sounds[i+1,:imix]

sounds = sounds[:,imix:] 

len_seconds = 2 * sounds[0].shape[0]/32000
len_total = len_seconds*nb_samples
sound_track = sounds.reshape(-1)


write_wav('test_audio.wav', sound_track)
def play():
    sd.play(sound_track,16000)

# play()

music_thread = threading.Thread(target=play,)

fakes = torch.squeeze(fakes).numpy()
point_zero = fakes[0]
fig,ax = plt.subplots()


def add_colorbar(fig, ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

times = np.linspace(0, 2 * 128 / 121, point_zero.shape[1])
freqs = librosa.mel_frequencies(
    n_mels=point_zero.shape[0], fmin=0.0, fmax=16000 / 2, htk=True
)

X, Y = np.meshgrid(times, freqs)
im0 = ax.pcolormesh(X, Y, point_zero, cmap="magma",vmin=-1,vmax=1)
add_colorbar(fig, ax, im0)
labelRow = "Temps"
labelCol = "Fréquences"
ax.set_xlabel(labelRow)
ax.set_ylabel(labelCol)

def animate(iter):
    if iter == 0:
        music_thread.start()
    fig.suptitle('Sample nb : '+str(iter))
    STFT_amp = fakes[iter]
    im0.set_array(STFT_amp[:-1, :-1])
    return im0


def init():
    pass

anim = animation.FuncAnimation(fig,animate,frames=samples.shape[0],interval=len_seconds*1000,blit=False,repeat=False,init_func = init)
# plt.show()
# anim.save(filename=gif_path+"courbe.mp4", dpi =80, fps=1,extra_args=['-vcodec', 'h264', 
#                       '-pix_fmt', 'yuv420p'])


# FFwriter=animation.FFMpegWriter(fps=1)#, extra_args=['-vcodec', 'libx264'])
# anim.save(filename=gif_path+"\courbe.avi", writer=FFwriter)
# # my_writer=animation.PillowWriter(fps=0.5, codec='libx264', bitrate=2)
# # anim.save(filename=gif_path+'gif_test.mp4', writer=my_writer)
# print('hi')
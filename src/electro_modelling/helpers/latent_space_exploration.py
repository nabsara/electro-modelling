
import torch
import numpy as np
import librosa
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

from electro_modelling.datasets.signal_processing import SignalOperators
from electro_modelling.models.base import GAN


nmels=128



def load_model(nmels,checkpoint_path):
    operator = SignalOperators(nfft=1024,nmels=nmels,sr=16000)
    model = GAN(z_dim=256, model_name='wgan', init_weights=True,dataset='techno',img_chan=1,operator=operator)
    generator = model.generator
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['model_state_dict'])
    model.generator=generator
    return(operator,model)


def create_trajectories(model,nb_latent_points,nb_interpolation,latent_dim):
    def generate_latent_points(latent_dim, n_samples):
        # generate points in the latent space
        x_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        return(z_input/np.linalg.norm(z_input,axis=1,keepdims=True))
        
    def slerp(val, low, high):
        omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
        so = np.sin(omega)
        if so == 0:
            # L'Hopital's rule/LERP
            return (1.0-val) * low + val * high
        return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
    
    def interpolate_points(p1, p2, n_steps=10):
    #    interpolate ratios between the points
        ratios = np.linspace(0, 1, num=n_steps)
        # linear interpolate vectors
        vectors = list()
        for ratio in ratios:
            v = slerp(ratio, p1, p2)
            vectors.append(v)
        return np.asarray(vectors)

    latent_points = generate_latent_points(latent_dim, nb_latent_points)
    trajectories = np.empty((nb_latent_points-1,nb_interpolation,latent_dim))
    for i in range(nb_latent_points-1):
           # spherical interpolation between 2 consecutive points
        trajectories[i,:] = interpolate_points(latent_points[i],latent_points[i+1],nb_interpolation)
        
    return trajectories
    


def create_spec_sound(model,trajectory):
    fakes = model.generator.forward(trajectory).detach().cpu() #Pass forward
    fakes = fakes[:,:,:,:121]                         #Remove silence at the end of the samples
    sounds = model.get_sounds(fakes).numpy()
    
    all_fakes = torch.cat([fake for fake in fakes],dim=2)
    all_fakes = torch.unsqueeze(all_fakes,dim=0)
    sounds_compiled = model.get_sounds(all_fakes)[0].numpy()
    fakes = torch.squeeze(fakes).numpy()
    return(sounds,fakes,sounds_compiled,all_fakes)

    
def compute_metrics(curSignal,metrics_names):
    dataStruct = dict.fromkeys(metrics_names)
    dataStruct["Loudness"] = librosa.feature.rms(curSignal)
    # Compute the spectral centroid. [y, sr, S, n_fft, ...]
    dataStruct["Centroid"] = librosa.feature.spectral_centroid(curSignal)
    # Compute the spectral bandwidth. [y, sr, S, n_fft, ...]
    dataStruct["Bandwidth"] = librosa.feature.spectral_bandwidth(curSignal)
    # Compute spectral contrast [R16] , sr, S, n_fft, ...])	
    # dataStruct["Contrast"] = librosa.feature.spectral_contrast(curSignal)
    # Compute the spectral flatness. [y, sr, S, n_fft, ...]
    dataStruct["Flatness"] = librosa.feature.spectral_flatness(curSignal)
    # Compute roll-off frequency
    # dataStruct["Rolloff"] = librosa.feature.spectral_rolloff(curSignal)
    return (dataStruct)



def create_plot(model,sounds,sound_compiled,fakes,metrics_names):
    
    # sound_compiled = sounds.reshape(-1)
    len_sample_sec = sounds[0].shape[0]/model.operator.sr
    len_total_sec = sound_compiled.shape[0]/model.operator.sr
    nb_samples = sounds.shape[0]

    metrics_arr = []
    for sound in sounds:
        metrics_arr.append(compute_metrics(sound,metrics_names))
    fig = plt.figure(figsize=(18,10))
    spec = fig.add_gridspec(6,1,height_ratios = [0.75, 0.25, 0.15,0.15,0.15,0.15])
    ax_quad = fig.add_subplot(spec[0, :])
    ax_signal = fig.add_subplot(spec[1, :])
    ax0 = fig.add_subplot(spec[2, :])
    ax1 = fig.add_subplot(spec[3, :])
    ax2 = fig.add_subplot(spec[4, :])
    ax3 = fig.add_subplot(spec[5, :])
    ax_metrics = [ax0,ax1,ax2,ax3]
    
    
    #SPECTROGRAM
    times_spec = np.linspace(0, 2 , fakes[0].shape[1])
    freqs_spec = librosa.mel_frequencies(
        n_mels=fakes[0].shape[0], fmin=0.0, fmax=16000 / 2, htk=True
    ) 
    def add_colorbar(fig, ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
    X, Y = np.meshgrid(times_spec, freqs_spec)
    im0 = ax_quad.pcolormesh(X, Y, fakes[0], cmap="magma",vmin=-1,vmax=1)
    add_colorbar(fig, ax_quad, im0)
    ax_quad.set_ylabel('Frequency')
    ax_quad.set_title('Spectrogram')
    ax_quad.set_xticks([])
    
    #WAVEFORM
    
    # SIGNAL
    total_times = np.linspace(0,len_total_sec,sound_compiled.shape[0])
    ax_signal.set_xlim(0,total_times[-1])
    color = ['whitesmoke','grey']
    ax_signal.set_xticks([])
    ax_signal.set_ylabel('Waveform',weight='bold')
    
    ax_signal.fill_between(total_times,sound_compiled)
    
    for j in range(nb_samples):
        ax_signal.axvspan(len_sample_sec*j,len_sample_sec*(j+1),alpha=0.075,color=color[j%2])
        
    vl = ax_signal.axvline(0, ls='-', color='r',alpha=0.5,linewidth=0.9)
        
    #METRICS  
    for n,ax in enumerate(ax_metrics):
        name = metrics_names[n]
        ax.set_ylabel(name,weight='bold')
        ax.set_xlim(0,len_total_sec)
        if n!=len(ax_metrics)-1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time (s)')
        
        for j in range(nb_samples):
            metric = metrics_arr[j][name][0]
            metric_times = np.linspace(len_sample_sec*j,len_sample_sec*(j+1),metric.shape[0])
            

        
            ax.plot(metric_times,metric,color='navy',alpha=0.7,linewidth=0.7)
            
            ax.axvspan(len_sample_sec*j,len_sample_sec*(j+1),alpha=0.075,color=color[j%2])
            mean_value = np.mean(metric)
            ax.hlines(mean_value,len_sample_sec*j,len_sample_sec*(j+1),color='k',linewidth=0.85)
       

        
  
    
    
    
    
    
    
    

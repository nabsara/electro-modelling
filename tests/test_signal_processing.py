import sys
import soundfile as sf
sys.path.append('./src/')


from electro_modelling.datasets.techno_dataset import TechnoDatasetWav
from electro_modelling.datasets.signal_processing import SignalOperators
from electro_modelling.helpers.helpers_audio import write_wav,plot_spectrogram


data_path = r'C:\Users\NILS\Documents\ATIAM\Informatique\PROJET\data\\'
dataset_file = data_path + r'\techno.dat'


dataset_techno = TechnoDatasetWav(dat_location=dataset_file)
audio_signal = dataset_techno.__getitem__(1)


sr = 16000
Nfft =1024
Nmels = int(Nfft/2+1)

write_wav(data_path+'test.wav',audio_signal,sr)

operator = SignalOperators(Nfft,Nmels,sr)

stft_mel = operator.forward(audio_signal)

plot_spectrogram(stft_mel[0],stft_mel[1],operator.freqs,operator.times)


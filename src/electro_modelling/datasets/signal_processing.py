import numpy as np
import librosa



class SignalOperators:
    
    def __init__(self, Nfft,Nmels,sr,signal_length=32000):
        self.Nfft = Nfft
        self.Nfft2 = int(Nfft/2+1)
        self.Nmels = self.Nfft2
        self.hop = int(self.Nfft/4)
        self.win_length = self.Nfft
        self.sr = sr
        self.signal_length = signal_length
        self.nb_trames = int((self.signal_length-self.win_length)/self.hop)
    
        self.freqs = np.linspace(0,self.sr/2,self.Nfft2)
        self.times = np.linspace(0,self.signal_length/self.sr,self.nb_trames)
        self.mel_freqs = librosa.mel_frequencies(n_mels=self.Nmels, fmin=0.0, fmax=sr/2,htk=True)

    def forward(self,signal):
        STFT = self.get_stft(signal)
        STFT_mel = self.stft_to_mel(STFT)
        return(STFT_mel)
    
    def backward(self,STFT_mel):
        #to be continued
        pass
        

    def get_IF_from_phase(self,phase):
        unwrapped_phase = np.unwrap(phase,axis=1)
        IF = np.pad(np.diff(unwrapped_phase,axis=1)/np.pi,([0,0],[1,0]))  
        return(IF)   
    
    def get_phase_from_if(self,IF):
         return(np.cumsum(IF*np.pi,axis=1))
    
    def get_stft(self,signal):
        STFT = np.zeros((self.Nfft2,self.nb_trames),dtype='complex')
        for j in range(self.nb_trames):
            i0 = j*self.hop
            iend = min(i0+self.win_length,signal.size)
            signal_w = signal[i0:iend]
            fen = np.hanning(len(signal_w))
            signal_w = signal_w*fen
            STFT[:,j] = np.fft.rfft(signal_w,self.Nfft)
        return STFT
    
    
    def stft_to_mel(self,STFT):
        fbank = librosa.filters.mel(sr=self.sr, n_fft=self.Nfft, n_mels=self.Nmels,fmin=0,fmax=self.sr/2,htk=True,norm=False)
        STFT_mel_amp =np.dot(fbank, np.abs(STFT))
        STFT_mel_amp = np.log10(STFT_mel_amp+1e-6 )       
        STFT_mel_if = np.dot(fbank,self.get_IF_from_phase(np.angle(STFT)))
        STFT_mel = np.stack((STFT_mel_amp,STFT_mel_if))
        return(STFT_mel)
    
    
    def rebuild_signal(self,STFT):
        N = self.hop*STFT.shape[1]+self.win_length
        fen = np.hanning(self.win_length)# définition de la fenetre d'analyse
        ws = fen; # définition de la fenêtre de synthèse
        y = np.zeros((N)) # signal de synthèse
        for u in np.arange(0,STFT.shape[1]).reshape(-1): # boucle sur les trames
            deb = u * self.hop  # début de trame
            fin = deb + self.win_length # fin de trame
            y[deb:fin]=y[deb:fin]+(np.real(np.fft.irfft(STFT[:,u])*ws))
        return(y)


    def mel_to_stft(self,STFT_mel,griffin=False):
        STFT_mel_amp,STFT_mel_IF = STFT_mel[0],STFT_mel[1]
        #to be continued..
        # return(STFT)
        pass
        
    def griffin_phase(self,STFT_amp):
        #To be continued
        #return(phase)
        pass
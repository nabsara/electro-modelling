import numpy as np
import librosa


class SignalOperators:
    def __init__(
        self,
        nfft=1024,
        nmels=512,
        sr=16000,
        ntimes=128,
        signal_length=32000,
        phase_rec_method="Griffin-Lim",
    ):

        # Parameters for the STFT analysis and synthesis
        self.ntimes = ntimes
        self.nfft = nfft
        self.nfft2 = int(nfft / 2 + 1)
        self.hop = int(self.nfft / 4)
        self.win_length = self.nfft
        self.sr = sr
        self.signal_length = signal_length
        self.nb_trames = int((self.signal_length - self.win_length) / self.hop)

        # Parameters for the MEL spectrograms
        self.nmels = nmels
        self.freqs = np.linspace(0, self.sr / 2, self.nfft2)
        self.times = np.linspace(
            0, self.signal_length / self.sr * self.ntimes / self.nb_trames, self.ntimes
        )
        self.mel_freqs = librosa.mel_frequencies(
            n_mels=self.nmels, fmin=0.0, fmax=sr / 2, htk=True
        )
        self.mel_fbank = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.nfft,
            n_mels=self.nmels,
            fmin=0,
            fmax=self.sr / 2,
            htk=True,
            norm=False,
        )
        # self.inverse_fbank = self.mel_fbank.transpose() # TODO : check if use of transpose or Moore pseudo inverse
        self.inverse_fbank = np.linalg.pinv(self.mel_fbank)  # TODO : check if use of transpose or Moore pseudo inverse

        # Phase reconstruction method
        self.phase_rec_method = phase_rec_method

        # If the chosen phase reconstruction method is the Griffin-Lim algorithm, needed parameters
        if self.phase_rec_method == "Griffin-Lim":
            self.griffin_phase_init = None
            self.griffin_nb_iterations = 50
            self.griffin_required_loss = False

    def forward(self, signal):
        STFT = self.get_stft(signal)
        STFT_mel = self.stft_to_mel(STFT)
        return STFT_mel
    
    def backward(self,STFT_mel):
        if self.phase_rec_method == 'Griffin-Lim': # Use the Griffin-Lim algorithm to reconstruct the phase from a STFT amplitude
            STFT_mel_amp_log = STFT_mel
            STFT_mel_amp = 10**(STFT_mel_amp_log)
            STFT_amp = self.mel_to_stft_griffin(STFT_mel_amp)
            STFT_phase = self.griffin_phase(STFT_amp)
        elif (
            self.phase_rec_method == "IF"
        ):  # Use the instantaneous frequencies generated by the network to reconstruct the phase
            STFT_amp, STFT_IF = self.mel_to_stft_IF(STFT_mel)
            STFT_phase = self.get_phase_from_if(STFT_IF)
        else:
            raise AssertionError("This phase reconstruction method is not implemented.")
        STFT = STFT_amp * np.exp(1j * STFT_phase)
        signal = self.rebuild_signal(STFT)
        signal = signal/np.max(abs(signal))
        return signal

    def get_IF_from_phase(self, phase):
        unwrapped_phase = np.unwrap(phase, axis=1)
        IF = np.pad(np.diff(unwrapped_phase, axis=1) / np.pi, ([0, 0], [1, 0]))
        return IF

    def get_phase_from_if(self, IF):
        return np.cumsum(IF * np.pi, axis=1)

    def get_stft(self, signal):
        STFT = np.zeros((self.nfft2, self.nb_trames), dtype="complex")
        for j in range(self.nb_trames):
            i0 = j * self.hop
            iend = min(i0 + self.win_length, signal.size)
            signal_w = signal[i0:iend]
            fen = np.hanning(len(signal_w))
            signal_w = signal_w * fen
            STFT[:, j] = np.fft.rfft(signal_w, self.nfft)
        STFT = np.pad(STFT, ([0, 0], [0, self.ntimes - self.nb_trames]))
        return STFT

    def stft_to_mel(self, STFT):
        STFT_mel_amp = np.dot(self.mel_fbank, np.abs(STFT))
        STFT_mel_amp = np.log10(STFT_mel_amp + 1e-6)
        STFT_mel_if = np.dot(self.mel_fbank, self.get_IF_from_phase(np.angle(STFT)))
        STFT_mel = np.stack((STFT_mel_amp, STFT_mel_if))
        return STFT_mel

    def rebuild_signal(self, STFT):
        N = self.hop * STFT.shape[1] + self.win_length
        fen = np.hanning(self.win_length)  # définition de la fenetre d'analyse
        ws = fen
        # définition de la fenêtre de synthèse
        y = np.zeros((N))  # signal de synthèse
        for u in np.arange(0, STFT.shape[1]).reshape(-1):  # boucle sur les trames
            deb = u * self.hop  # début de trame
            fin = deb + self.win_length  # fin de trame
            y[deb:fin] = y[deb:fin] + (np.real(np.fft.irfft(STFT[:, u]) * ws))
        return y

    def mel_to_stft_griffin(self, STFT_mel_amp):
        STFT_amp = np.dot(self.inverse_fbank, STFT_mel_amp)
        return STFT_amp

    def mel_to_stft_IF(self, STFT_mel):
        STFT_amp = np.dot(self.inverse_fbank, STFT_mel[0, :])
        STFT_IF = np.dot(self.inverse_fbank, STFT_mel[1, :])
        return STFT_amp, STFT_IF

    def griffin_phase(self, STFT_amp):

        phase_init = self.griffin_phase_init
        nb_iterations = self.griffin_nb_iterations
        required_loss = self.griffin_required_loss

        if required_loss:
            loss = np.empty(
                nb_iterations
            )  # to show the evolution of the loss (MSE between input STFT amplitude and estimated STFT amplitude) over the iterations

        if phase_init is None:
            STFT_hat = STFT_amp*np.exp( 1j*np.random.rand(STFT_amp.shape[0], STFT_amp.shape[1])*np.pi/2)
        else:
            STFT_hat = STFT_amp * np.exp(1j * phase_init)

        for ind_iteration in range(nb_iterations):
            x_recovered = self.rebuild_signal(STFT_hat)

            STFT_temp = self.get_stft(x_recovered)

            STFT_hat = STFT_amp * np.exp(1j * np.angle(STFT_temp))
            if required_loss:
                loss[ind_iteration] = np.linalg.norm(np.abs(STFT_temp) - STFT_amp)

        if required_loss:
            return np.angle(STFT_hat), loss
        else:
            return np.angle(STFT_hat)

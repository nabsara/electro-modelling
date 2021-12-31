import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa


def write_wav(path, signal, sr=16000):
    sf.write(path, signal, sr)


def plot_spectrogram(
    STFT_amp,
    STFT_phase,
    freqs,
    times,
    labelRow="Temps",
    labelCol="Fréquences",
    titles=["Amplitudes", "Phase"],
    figsize=(15, 6),
):
    def add_colorbar(fig, ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    X, Y = np.meshgrid(times, freqs)

    im0 = ax[0].pcolor(X, Y, STFT_amp, cmap="magma")
    im1 = ax[1].pcolor(X, Y, STFT_phase, cmap="magma")

    add_colorbar(fig, ax[0], im0)
    add_colorbar(fig, ax[1], im1)
    ax[0].set_xlabel(labelRow)
    ax[0].set_ylabel(labelCol)

    ax[0].set_xlabel(labelRow)
    ax[0].set_ylabel(labelCol)
    plt.show()


def plot_spectrogram_mag(STFT_amp, fig=None, ax=None):
    def add_colorbar(fig, ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    times = np.linspace(0, 2 * 128 / 121, STFT_amp.shape[1])
    freqs = librosa.mel_frequencies(
        n_mels=STFT_amp.shape[0], fmin=0.0, fmax=16000 / 2, htk=True
    )

    X, Y = np.meshgrid(times, freqs)

    im0 = ax.pcolor(X, Y, STFT_amp, cmap="magma")

    add_colorbar(fig, ax, im0)
    labelRow = "Temps"
    labelCol = "Fréquences"
    ax.set_xlabel(labelRow)
    ax.set_ylabel(labelCol)

    # plt.show()


def image_grid_spectrograms(fakes):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    fig, axs = plt.subplots(1, fakes.shape[0], figsize=(13, 5))
    for i, ax in enumerate(axs):
        plot_spectrogram_mag(fakes[i][0], fig, ax)
    fig.tight_layout()
    return fig

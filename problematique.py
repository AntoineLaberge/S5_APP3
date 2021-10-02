import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

filePath = "./note_guitare_LAd.wav"

def getTFD(x_n):
    X_m = np.fft.fft(x_n)

    X_phase = np.angle(X_m)
    X_magnitude = np.abs(X_m)

    return X_m, X_phase, X_magnitude

def analyzeWav(file):
    fe, x_n = wavfile.read(file)

    X_m, X_phase, X_magnitude = getTFD(x_n)

    plt.subplot(211)
    plt.title("Fonction")
    plt.stem(x_n)

    plt.subplot(223)
    plt.title("Phase")
    plt.stem(X_phase)

    plt.subplot(224)
    plt.title("Magnitude")
    plt.stem(X_magnitude)

    plt.show()

    for i in range(32):
        m = np.argmax(X_m)


if __name__ == "__main__":
    analyzeWav(filePath)


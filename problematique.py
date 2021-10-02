# Problematique APP3
# Laba0902 - degj2706

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

filePath = "./note_guitare_LAd.wav"

def getTFD(x_n):
    X_m = np.fft.fft(x_n)

    X_phase = np.angle(X_m)
    X_magnitude = np.abs(X_m)

    return X_m, X_phase, X_magnitude

def getPrincipaleSin(X_m, X_phase, X_magnitude, max_amplitude):
    indexes = []
    for magnitude in range(len(X_magnitude)):
        if (X_magnitude[magnitude] > 0.1* max_amplitude and X_magnitude[magnitude] > X_magnitude[magnitude-1] and X_magnitude[magnitude] > X_magnitude[magnitude+1]):
            indexes.append(magnitude)

    principal_X_m = [X_m[i] for i in indexes]
    principal_magnitude = [X_magnitude[i] for i in indexes]
    principal_phase = [X_phase[i] for i in indexes]

    return principal_X_m, principal_phase, principal_magnitude

# Not finished
def getFilterRIF():
    w_barre = np.pi/1000
    w_c = 0

# Not finished
def getSignalEnvelope(x_n):
    signal = np.array(x_n)
    absolute_signal = np.abs(signal)
    filter = getFilterRIF()

# Not finished
def normalizeSignal(nb_ech, signal):
    print()

def analyzeWav(file):
    fe, x_n = wavfile.read(file)

    X_m, X_phase, X_magnitude = getTFD(x_n)

    max_amplitude = np.amax(X_magnitude)

    principal_X_m, principal_phase, principal_magnitude = getPrincipaleSin(X_m, X_phase, X_magnitude, max_amplitude)

    plt.subplot(211)
    plt.title("Fonction")
    plt.stem(x_n)

    plt.subplot(223)
    plt.title("Phase")
    plt.stem(principal_phase)

    plt.subplot(224)
    plt.title("Magnitude")
    plt.stem(principal_magnitude)

    plt.show()

if __name__ == "__main__":
    analyzeWav(filePath)
    print("")


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

filePath = "./note_guitare_LAd.wav"

def getTFD(x_n):
    X_m = np.fft.fft(x_n)

    X_phase = np.angle(X_m)
    X_magnitude = np.abs(X_m)

    return X_m, X_phase, X_magnitude

def getPrincipalSinusoids(X_phase, X_magnitude, max):
    indexes = []
    treshhold = 0.1

    for i in range (len(X_magnitude)):
        if(X_magnitude[i] > treshhold*max and X_magnitude[i] > X_magnitude[i-1] and X_magnitude[i] > X_magnitude[i+1]):
            indexes.append(i)

    principal_phases = [X_phase[i] for i in indexes]
    principal_magnitudes = [X_magnitude[i] for i in indexes]

    return principal_phases, principal_magnitudes;

def analyzeWav(file):
    fe, x_n = wavfile.read(file)

    X_m, X_phase, X_magnitude = getTFD(x_n)

    max_amplitude = np.amax(X_magnitude)

    principal_phases, principal_magnitudes = getPrincipalSinusoids(X_phase, X_magnitude, max_amplitude)

    print(principal_phases)
    print(principal_magnitudes)

    plt.subplot(211)
    plt.title("Fonction")
    plt.plot(x_n)

    plt.subplot(223)
    plt.title("Phase")
    plt.stem(principal_phases)

    plt.subplot(224)
    plt.title("Magnitude")
    plt.stem(principal_magnitudes)

    plt.show()




if __name__ == "__main__":
    analyzeWav(filePath)


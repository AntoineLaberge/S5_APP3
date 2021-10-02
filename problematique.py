import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

filePath = "./note_guitare_LAd.wav"


def getTFD(x_n):
    X_m = np.fft.fft(x_n)

    X_phase = np.angle(X_m)
    X_magnitude = np.abs(X_m)

    return X_m, X_phase, X_magnitude


def getPrincipalSinusoids(X_m, X_phase, X_magnitude, max):
    indexes = []
    threshold = 0.1

    for i in range(len(X_magnitude)):
        if X_magnitude[i] > threshold * max and X_magnitude[i] > X_magnitude[i - 1] and X_magnitude[i] > X_magnitude[
            i + 1]:
            indexes.append(i)

    principal_X_m = [X_m[i] for i in indexes]
    principal_phases = [X_phase[i] for i in indexes]
    principal_magnitudes = [X_magnitude[i] for i in indexes]

    return principal_X_m, principal_phases, principal_magnitudes


def normalizeSignal(w_barre, nb_ech):
    return (2 * np.pi * w_barre) / nb_ech


def getSampleCount(w_barre, limit):
    sample_count = 0
    for w in w_barre:
        if w < limit:
            sample_count += 1
        else:
            break
    return sample_count


def getRIF(w_barre, N):
    m = int((N * w_barre) / (2 * np.pi))
    K = 2 * m + 1
    h = [(1 / N) * (np.sin(np.pi * n * K / N) / np.sin(np.pi * n / N)) if
         n != 0 else K / N for n in range(int(-N / 2), int(N / 2) + 1)]
    return h

def testRIFs(w_barre):
    orders = [600]
    for N in orders:
        h_n = getRIF(w_barre, N)
        plt.stem(h_n)
        plt.title('RIF N : ' + str(N))
        plt.show()
        h_m, h_m_phase, h_m_magnitude = getTFD(h_n)
        print(h_m_magnitude)
        plt.plot(h_m_magnitude)
        plt.show()



def getSignalEnvelope(x_n, w_barre):
    signal = np.array(x_n)
    absolute_signal = np.abs(signal)
    testRIFs(w_barre)


def analyzeWav(file):
    fe, x_n = wavfile.read(file)

    w_barre_coupure = np.pi / 1000

    X_m, X_phase, X_magnitude = getTFD(x_n)

    max_amplitude = np.amax(X_magnitude)

    principal_X_m, principal_phases, principal_magnitudes = getPrincipalSinusoids(X_m, X_phase, X_magnitude,
                                                                                  max_amplitude)
    #
    # w = normalize_signal(np.arange(len(X_m)), len(X_m))
    # k = getSampleCount(w, w_barre_coupure)
    # print(k)

    getSignalEnvelope(x_n, w_barre_coupure)

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

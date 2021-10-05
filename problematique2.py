import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Informations
#   - Creer un filtre coupe-bande a partir d'un filtre passe-bas
#   - Gain DC = 0 dB
#   - Effet coupe-bande de 960 a 1040Hz
#   - Ordre : P = 6000
#   - Concevoir un filtre passe-bas an ayant recours a la methode de la fenetre

filePath = "./note_basson_plus_sinus_1000_Hz.wav"

def get_tfd(x_n):
    X_m = np.fft.fft(x_n)

    X_phase = np.angle(X_m)
    X_magnitude = np.abs(X_m)

    return X_m, X_phase, X_magnitude

def to_db(h_n):
    return 20 * np.log10(np.abs(h_n))


def reponse_impulsionnelle_lp(P, K):
    hlp_n = [K / P if n == 0 else ((1 / P) * ((np.sin((np.pi * n * K) / P)) / (np.sin((np.pi * n) / P)))) for n in range(int((-P / 2) + 1), int((P / 2)))]

    return hlp_n


def reponse_impulsionnelle_bs(hlp_n, w0):
    delta_n = [1 if n == int(len(hlp_n)/2) else 0 for n in range(len(hlp_n))]

    hbs_n = [delta_n[n] - (2 * hlp_n[n] * np.cos(np.multiply(w0, n))) for n in range(len(hlp_n))]

    return hbs_n


def coupe_bande_rif(file):
    fe, x_n = wavfile.read(file)
    P = 6000
    f1 = 1000
    f2 = 40

    w0 = (2 * np.pi * f1) / fe
    w_barre = (2 * np.pi * f2) / fe

    m = (f2 * P) / fe
    K = (2 * m) + 1

    hlp_n = reponse_impulsionnelle_lp(P, K)
    hbs_n = reponse_impulsionnelle_bs(hlp_n, w0)

    window = np.hanning(len(x_n))
    x_n_hanning = x_n * window

    X_m, X_phase, X_magnitude = get_tfd(x_n_hanning)

    signal_filtered = np.convolve(np.abs(x_n), hbs_n)

    wavfile.write("basson_filtered.wav", fe, signal_filtered)

    plt.figure()
    plt.stem(hbs_n)
    plt.show()

if __name__ == "__main__":
    coupe_bande_rif(filePath)

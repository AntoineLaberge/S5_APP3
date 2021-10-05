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
    hlp_n = [K / P if n == 0 else ((1 / P) * ((np.sin((np.pi * n * K) / P)) / (np.sin((np.pi * n) / P)))) for n in range(int((-P / 2) + 1), int((P / 2) + 1))]

    return np.hanning(P) * hlp_n


def reponse_impulsionnelle_bs(hlp_n, w0, P):
    n = np.arange(-int(P / 2) + 1, int(P / 2) + 1)

    delta_n = np.array([int(val==0) for val in n])

    hbs_n = delta_n - (2 * hlp_n * np.cos(w0 * n))

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

    hlp_n = reponse_impulsionnelle_lp(P, int(K))
    hbs_n = reponse_impulsionnelle_bs(hlp_n, w0, P)

    Hlp = np.abs(np.fft.fft(hlp_n))
    Hbs = np.abs(np.fft.fft(hbs_n))
    freqs = [m/len(Hbs)*fe for m in range(len(Hbs))]


    signal_filtered = np.convolve(hbs_n, x_n)

    wavfile.write("basson_filtered.wav", fe, signal_filtered.astype(np.int16))

    # Graphique de la reponse a l'impulsion hbs(n) normalise en frequence
    plt.figure("Figure 1")
    plt.stem(freqs, Hbs)
    plt.xlim([900, 1100])
    plt.title("Réponse impulsionnelle du filtre coupe-bande normalisée en fréquence")
    plt.xlabel("Frequence")
    plt.ylabel("Amplitude")

    # Graphique de la reponse a l'impulsion hbs(n)
    plt.figure("Figure 2")
    plt.plot(hbs_n)
    plt.title("Réponse impulsionnelle du filtre coupe-bande (Ordre = 6000)")
    plt.xlabel("n")
    plt.ylabel("Amplitude")

    # Graphique de la réponse à une sinusoïde de 1000 Hz
    sine = [np.sin(2 * np.pi * 1000 * x/fe) for x in range(len(x_n))]
    filtered_sine = np.convolve(sine, hbs_n)
    plt.figure("Figure 3")
    plt.plot(filtered_sine)
    plt.title("Réponse à une sinusoïde de 1000 Hz")
    plt.xlabel("n")
    plt.ylabel("Amplitude")

    # Graphiques amplitude et phase de la réponse en fréquence
    plt.figure("Figure 4")
    plt.subplot(211)
    plt.plot(freqs, np.abs(np.fft.fft(hbs_n)))
    plt.title("Amplitude de l'impulsion hbs(m) normalisée en fréquence")
    plt.xlabel("Frequence")
    plt.ylabel("Amplitude")
    plt.subplot(212)
    plt.plot(freqs, np.angle(np.fft.fft(hbs_n)))
    plt.title("Phase de l'impulsion hbs(m) normalisée en fréquence")
    plt.xlabel("Frequence")
    plt.ylabel("Phase")

    # Graphiques des spectres d’amplitude des signaux basson avant et après filtrage
    plt.figure("Figure 5")
    plt.subplot(211)
    plt.plot(x_n)
    plt.title("Spectre d'amplitude des signaux basson avant filtrage")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.subplot(212)
    plt.plot(signal_filtered)
    plt.title("Spectre d'amplitude des signaux basson après filtrage")
    plt.xlabel("n")
    plt.ylabel("Amplitude")

    plt.show()

if __name__ == "__main__":
    coupe_bande_rif(filePath)

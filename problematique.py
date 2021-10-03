import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

filePath = "./note_guitare_LAd.wav"


def synthetize_signal(phases, magnitudes, enveloppe):
    w = [2 * np.pi * m / len(magnitudes) for m in range(len(magnitudes))]

    recomposed_sine = np.zeros(len(enveloppe))

    for n in range(len(enveloppe)):
        recomposed_sine[n] = np.sum(magnitudes * np.sin(np.multiply(n,w) + phases))

    plt.plot(recomposed_sine)
    plt.show()

    recomposed_note = recomposed_sine * enveloppe

    plt.plot(recomposed_note)
    plt.show()

    return recomposed_note

def get_tfd(x_n):
    X_m = np.fft.fft(x_n)

    X_phase = np.angle(X_m)
    X_magnitude = np.abs(X_m)

    return X_m, X_phase, X_magnitude


def get_principal_sinusoids(X_m, X_phase, X_magnitude, max):
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


def to_db(values):
    return 20 * np.log10(values);


def get_rif_impulse_response(w_barre, p, nb_ech):
    m = int((p * w_barre) / (2 * np.pi))
    K = 2 * m + 1
    h_n = [(1 / p) * (np.sin(np.pi * n * K / p) / np.sin(np.pi * n / p)) if
           n != 0 else K / p for n in range(int(-(p / 2) + 1), int(p / 2))]
    zeros = np.zeros(nb_ech - len(h_n))
    padded_h_n = np.concatenate((h_n, zeros))
    return padded_h_n


def get_rif_freq_response(h_n):
    H_m, H_m_phase, H_m_magnitude = get_tfd(h_n)
    return H_m


def get_tested_freq_responses(w_barre, orders, nb_ech):
    H_ms = []
    for p in orders:
        # Get equation for filter with order p
        h_n = get_rif_impulse_response(w_barre, p, nb_ech)

        # Plot stem
        # plt.stem(h_n)
        # plt.title('RIF p : ' + str(p))
        # plt.show()

        # Get amplitudes and store it in array
        H_m = get_rif_freq_response(h_n)
        H_ms.append(H_m)
    return H_ms


def plot_normalized_freq_responses(H_ms, w_barre, nb_ech, orders, x_range, y_range):
    # Plot H(w), normalize freq to get w instead of m
    w = [2*np.pi*m/nb_ech for m in range(nb_ech)]

    for i in range(len(H_ms)):
        H_m_magnitude_db = to_db(abs(H_ms[i]))
        plt.plot(w, H_m_magnitude_db, label='p='+str(orders[i]))
        plt.xlim(x_range)
        plt.ylim(y_range)

    plt.legend()
    plt.show()
    return


def get_abs_signal(x_n):
    signal = np.array(x_n)
    absolute_signal = np.abs(signal)


def find_valid_rif(w_barre, orders, nb_ech, x_range, y_range):
    # We must test multiple filter orders to find the one that fits
    H_ms = get_tested_freq_responses(w_barre, orders, nb_ech)
    plot_normalized_freq_responses(H_ms, w_barre, nb_ech, orders, x_range, y_range)


def get_signal_envelope(x_n_abs, h_n):
    # To get signal enveloppe we must take the abs of the signal and
    # convolve it with a FIR filter with pi/100 -> -3db gain
    return np.convolve(x_n_abs, h_n)

def analyzeWav(file):
    # Read file
    fe, x_n = wavfile.read(file)

    w_barre_coupure = np.pi / 1000

    plt.plot(x_n)
    plt.show()

    #Hanning ?
    # window = np.hanning(len(x_n))
    # x_n_hanning = x_n*window

    # Caclulate tfd on signal from file
    X_m, X_phase, X_magnitude = get_tfd(x_n)

    # Fin max amplitude from x_n
    max_amplitude = np.amax(X_magnitude)

    # Get max 32 best sinusoids
    principal_X_m, principal_phases, principal_magnitudes = get_principal_sinusoids(X_m, X_phase, X_magnitude,
                                                                                    max_amplitude)

    # Find a valid filter that match our condition pi/100 -> -3db
    # orders = range(64, 2048, 64)
    # x_range = [w_barre_coupure - 0.0005, w_barre_coupure + 0.0005]
    # y_range = [-3 - 0.05, -3 + 0.05]
    # find_valid_rif(w_barre_coupure, orders, len(x_n), x_range, y_range)

    ## From this analysis, we see that higher orders are better, we will test around 900
    # orders = range(850, 950, 10)
    # x_range = [w_barre_coupure - 0.0002, w_barre_coupure + 0.0002]
    # y_range = [-3 - 0.05, -3 + 0.05]
    # find_valid_rif(w_barre_coupure, orders, len(x_n), x_range, y_range)

    #Results show between 880 and 890
    orders = range(880, 890, 1)
    x_range = [w_barre_coupure - 0.00001, w_barre_coupure + 0.00001]
    y_range = [-3 - 0.02, -3 + 0.02]
    # find_valid_rif(w_barre_coupure, orders, len(x_n), x_range, y_range)

    #Conclusion : p = 884 , gets 5 x 10^-7 difference from pi/100 for -3db
    p = 884
    h_n = get_rif_impulse_response(w_barre_coupure, p, len(x_n))
    enveloppe = get_signal_envelope(abs(x_n), h_n)
    plt.plot(enveloppe)
    plt.show()

    synthetized_signal = synthetize_signal(principal_phases, principal_magnitudes, enveloppe)

    wavfile.write("note_guitare_LAd_output.wav", fe, synthetized_signal)

    # Get enveloppe from signal
    # get_signal_envelope(x_n, w_barre_coupure)
    #
    # plt.subplot(211)
    # plt.title("Fonction")
    # plt.plot(x_n)
    #
    # plt.subplot(223)
    # plt.title("Phase")
    # plt.stem(principal_phases)
    #
    # plt.subplot(224)
    # plt.title("Magnitude")
    # plt.stem(principal_magnitudes)

if __name__ == "__main__":
    analyzeWav(filePath)

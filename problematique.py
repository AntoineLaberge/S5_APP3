import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

filePath = "./note_guitare_LAd.wav"

def plot_spectrum(signal, title):
    magnitude = np.abs(signal)
    magnitude_db = to_db(magnitude)

    plt.figure()
    plt.plot(magnitude_db)
    plt.title(title)
    plt.xlabel("m (frequences, Hz)")
    plt.ylabel("Amplitude (dB)")

def convert_m_to_f(m_indexes, nb_ech, fe):
    #F/fe = m/n
    #F = Fe * m/n
    return [fe*(m/nb_ech) for m in m_indexes]

def print_signal_information(frequencies, magnitudes, phases):
    two_dec_freqs = [float("{:.2f}".format(f)) for f in frequencies]
    two_dec_magnitudes = [float("{:.2f}".format(m)) for m in magnitudes]
    two_dec_phases= [float("{:.2f}".format(p)) for p in phases]
    print("freqs: " + str(frequencies))
    print("two decimals freqs: " + str(two_dec_freqs))
    print("magns: " + str(magnitudes))
    print("two decimals magns: " + str(two_dec_magnitudes))
    print("phases" + str(phases))
    print("two decimals phases: " + str(two_dec_phases))
    return


def get_note_freq(note):
    mapping = {
        "DO": 261.6,
        "DO#": 277.2,
        "RE": 293.7,
        "RE#": 311.1,
        "MI": 329.6,
        "FA": 349.2,
        "FA#": 370,
        "SOL": 392,
        "SOL#": 415.3,
        "LA": 440,
        "LA#": 466.2,
        "SI": 493.9
    }
    return mapping[note]


def get_k_index_from_note(note):
    mapping = {
        "DO": -10,
        "DO#": -9,
        "RE": -8,
        "RE#": -7,
        "MI": -6,
        "FA": -5,
        "FA#": -4,
        "SOL": -3,
        "SOL#": -2,
        "LA": -1,
        "LA#": 0,
        "SI": 1
    }
    return mapping[note]


def synthetize_song(X_m, m_indexes, envelope, nb_ech, notes):
    final_sound = []
    nb_notes_kept = int(len(envelope) / len(notes))
    last_note = ""
    last_synthetized_note = []
    for note in notes:
        if note == last_note and len(last_synthetized_note) != 0:
            final_sound = np.concatenate((final_sound, last_synthetized_note))
            continue
        last_note = note
        last_synthetized_note = synthetize_signal(X_m, m_indexes, envelope, nb_ech, note)[:nb_notes_kept]
        final_sound = np.concatenate((final_sound, last_synthetized_note))
    return final_sound


def synthetize_signal(X_m, m_indexes, envelope, nb_ech, note):
    if note == "SILENCE":
        return np.zeros(len(envelope))
    k_index = get_k_index_from_note(note)
    factor = 2 ** (k_index / 12)
    phases = np.angle(X_m)
    magnitudes = np.abs(X_m)

    ws = [2 * np.pi * m * factor / nb_ech for m in m_indexes]
    ws_normalized = [w - (2 * np.pi) if w > np.pi else w for w in ws]

    sum_sines = [np.sum(magnitudes * np.sin(np.multiply(n, ws_normalized) + phases)) for n in range(len(envelope))]

    synthetized_x_n = sum_sines * envelope

    max_magnitude = np.amax(synthetized_x_n)
    synthetized_x_n_corrected = [magnitude/max_magnitude for magnitude in synthetized_x_n]

    return synthetized_x_n_corrected


def get_tfd(x_n):
    X_m = np.fft.fft(x_n)

    X_phase = np.angle(X_m)
    X_magnitude = np.abs(X_m)

    return X_m, X_phase, X_magnitude


def get_m_fund_freq_indexes(X_m, fund_freq):
    m_fund_freq_indexes = []

    for i in range(len(X_m)):
        if i != 0 and i % fund_freq == 0:
            m_fund_freq_indexes.append(i)

    return m_fund_freq_indexes


def is_principal_sinusoid(current, prev, next, threshold):
    # check if current > threshold and current > neighbours
    if current > threshold and current > prev and current > next:
        return True
    return False


def find_n_largest(array, n):
    indexes = np.argpartition(array, -n)[-n:]
    return np.sort(indexes)


def get_principal_sinusoids(X_m, note, fe):
    #Transform fund_freq to fund_m using relation f/fe = m/n
    fund_freq = int(get_note_freq(note))
    fund_m = int((fund_freq/fe)*len(X_m))

    # Get indexes for X_ms that are a multiple of fundamental m.
    m_fund_freq_indexes = get_m_fund_freq_indexes(X_m, fund_m)

    # Retrieve principal X_ms
    principal_X_ms = [X_m[m] for m in m_fund_freq_indexes]

    # Get 32 highest peaks
    principal_X_ms_magnitudes = np.abs(principal_X_ms)
    # We must trace back the original index (0->466, 1->466*2, etc) so (m+1) * fund_freq
    # Because indexes that we found with find_n_largest gives indexes between 0 and len(array)
    principal_X_ms_indexes = [(m + 1) * fund_m for m in find_n_largest(principal_X_ms_magnitudes, 32)]

    principal_X_ms = [X_m[m] for m in principal_X_ms_indexes]

    return principal_X_ms_indexes, principal_X_ms


def to_db(values):
    return 20 * np.log10(values)


def get_rif_impulse_response(w_barre, p, nb_ech, needs_padding):
    m = int((p * w_barre) / (2 * np.pi))
    K = 2 * m + 1
    h_n = [(1 / p) * (np.sin(np.pi * n * K / p) / np.sin(np.pi * n / p)) if
           n != 0 else K / p for n in range(int(-(p / 2) + 1), int(p / 2))]
    zeros = np.zeros(nb_ech - len(h_n))
    padded_h_n = np.concatenate((h_n, zeros))
    return padded_h_n if needs_padding else h_n


def get_rif_freq_response(h_n):
    H_m, H_m_phase, H_m_magnitude = get_tfd(h_n)
    return H_m


def get_tested_freq_responses(w_barre, orders, nb_ech):
    H_ms = []
    for p in orders:
        # Get equation for filter with order p
        h_n = get_rif_impulse_response(w_barre, p, nb_ech, True)

        # Get amplitudes and store it in array
        H_m = get_rif_freq_response(h_n)
        H_ms.append(H_m)
    return H_ms


def plot_normalized_freq_responses(H_ms, w_barre, nb_ech, orders, x_range, y_range):
    # Plot H(w), normalize freq to get w instead of m
    w = [2 * np.pi * m / nb_ech for m in range(nb_ech)]

    for i in range(len(H_ms)):
        H_m_magnitude_db = to_db(abs(H_ms[i]))
        plt.plot(w, H_m_magnitude_db, label='p=' + str(orders[i]))
        plt.title("Réponses des filtres RIF - " + str(np.amin(orders)) + " < p < " + str(np.amax(orders)))
        plt.xlabel("w (rad/ech)")
        plt.ylabel("Amplitude (db)")
        plt.xlim(x_range)
        plt.ylim(y_range)

    plt.legend()
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


def designRIF(w_barre_coupure, nb_ech):
    # Find a valid filter that match our condition pi/100 -> -3db
    orders = range(100, 2000, 100)
    x_range = [w_barre_coupure - 0.0005, w_barre_coupure + 0.0005]
    y_range = [-3 - 0.05, -3 + 0.05]
    find_valid_rif(w_barre_coupure, orders, nb_ech, x_range, y_range)

    # From this analysis, we see that higher orders are better, we will test around 900
    orders = range(850, 950, 10)
    x_range = [w_barre_coupure - 0.0002, w_barre_coupure + 0.0002]
    y_range = [-3 - 0.05, -3 + 0.05]
    find_valid_rif(w_barre_coupure, orders, nb_ech, x_range, y_range)

    # Results show between 880 and 890
    orders = range(880, 890, 1)
    x_range = [w_barre_coupure - 0.00001, w_barre_coupure + 0.00001]
    y_range = [-3 - 0.02, -3 + 0.02]
    find_valid_rif(w_barre_coupure, orders, nb_ech, x_range, y_range)


def analyzeWav(file):
    # Read file
    fe, x_n = wavfile.read(file)

    w_barre_coupure = np.pi / 1000

    # Plot initial signal
    plt.figure()
    plt.plot(x_n)
    plt.title("Signal discret x[n] - LA#")
    plt.xlabel("n (s)")
    plt.ylabel("Amplitude")

    # Uncomment next line to show plots that led to p = 884 for FIR order
    # Conclusion : p = 884 , gets 5 x 10^-7 difference from pi/100 for -3db
    # designRIF(w_barre_coupure, len(x_n))

    # Compute signal envelope
    p = 884
    h_n = get_rif_impulse_response(w_barre_coupure, p, len(x_n), False)
    envelope = get_signal_envelope(np.abs(x_n), h_n)
    plt.figure()
    plt.plot(envelope)
    plt.title("Enveloppe du signal - LA#")
    plt.xlabel("n (s)")
    plt.ylabel("Amplitude")

    # Hanning
    window = np.hanning(len(x_n))
    x_n_hanning = x_n * window
    plt.figure()
    plt.plot(x_n_hanning)
    plt.title("Fenêtre de hanning applique sur le signal - LA#")
    plt.xlabel("n (s)")
    plt.ylabel("Amplitude")

    # Caclulate tfd on signal after hanning was applied
    X_m, X_phase, X_magnitude = get_tfd(x_n_hanning)

    plot_spectrum(X_m, "Spectre de Fourier de la note LA#")

    # Get max 32 best sinusoids
    m_indexes, principal_X_ms = get_principal_sinusoids(X_m, "LA#", fe)

    #Print frequencies, magnitudes, phases
    print_signal_information(convert_m_to_f(m_indexes, len(x_n), fe), np.abs(principal_X_ms), np.angle(principal_X_ms))

    start_LA = time.time()
    # Synthetize signal by adding all best sinusoids and multiplying by the enveloppe
    synthetized_signal = synthetize_signal(principal_X_ms, m_indexes, envelope, len(x_n), "LA#")
    plt.figure()
    plt.plot(synthetized_signal)
    plt.title("Note synthétisée - LA#")
    plt.xlabel("n (s)")
    plt.ylabel("Amplitude")
    wavfile.write("note_guitare_LAd_output.wav", fe, np.array(synthetized_signal, dtype=np.float32))
    end_LA = time.time()
    print("time for synthetizing LA# : " + str(end_LA - start_LA))

    start_beethoven = time.time()
    # Beethoven 5th symphony
    # SOL SOL SOL MI bémol (silence) FA FA FA RE.
    notes = ["SOL", "SOL", "SOL", "RE#", "SILENCE", "FA", "FA", "FA", "RE"]
    synthetized_song = synthetize_song(principal_X_ms, m_indexes, envelope, len(x_n), notes)
    plt.figure()
    plt.plot(synthetized_song)
    plt.title("5e symphonie de Beethoven synthetisée")
    plt.xlabel("n (s)")
    plt.ylabel("Amplitude")
    wavfile.write("beethoven.wav", fe, np.array(synthetized_song, dtype=np.float32))
    end_beethoven = time.time()
    print("time for synthetizing 5th symphony : " + str(end_beethoven - start_beethoven))
    print("time for both synths : " + str(end_beethoven - start_LA))
    #Uncomment to next line to show plots
    #plt.show()

if __name__ == "__main__":
    analyzeWav(filePath)

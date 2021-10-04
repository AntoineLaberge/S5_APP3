import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def getTFD(x_n):
    X_m = np.fft.fft(x_n)

    X_phase = np.angle(X_m)
    X_magnitude = np.abs(X_m)

    return X_m, X_phase, X_magnitude


def question1():
    # Section a

    x1_n = [np.sin((0.1 * np.pi * n) + (np.pi / 4)) for n in range(20)]

    X1_m, X1_phase, X1_magnitude = getTFD(x1_n)

    plt.figure("TFD (x1_n)")
    plt.subplot(211)
    plt.title("Fonction")
    plt.stem(x1_n)

    plt.subplot(223)
    plt.title("Phase")
    plt.stem(X1_phase)

    plt.subplot(224)
    plt.title("Magnitude")
    plt.stem(X1_magnitude)

    x2_n = [(-1) ** n for n in range(10)]

    X2_m, X2_phase, X2_magnitude = getTFD(x2_n)

    plt.figure("TFD (x2_n)")
    plt.subplot(211)
    plt.title("Fonction")
    plt.stem(x2_n)

    plt.subplot(223)
    plt.title("Phase")
    plt.stem(X2_phase)

    plt.subplot(224)
    plt.title("Magnitude")
    plt.stem(X2_magnitude)

    x3_n = [1 if n == 10 else 0 for n in range(11)]

    X3_m, X3_phase, X3_magnitude = getTFD(x3_n)

    plt.figure("TFD (x3_n)")
    plt.subplot(211)
    plt.title("Fonction")
    plt.stem(x3_n)

    plt.subplot(223)
    plt.title("Phase")
    plt.stem(X3_phase)

    plt.subplot(224)
    plt.title("Magnitude")
    plt.stem(X3_magnitude)

    #    plt.show()

    # Section b

    w1 = [np.pi * 2 * m / len(x1_n) for m in range(20)]

    plt.figure("TFD (x1_w)")
    plt.subplot(211)
    plt.title("Phase")
    plt.stem(w1, X1_phase)

    plt.subplot(212)
    plt.title("Magnitude")
    plt.stem(w1, X1_magnitude)

    w2 = [np.pi * 2 * m / len(x2_n) for m in range(10)]

    plt.figure("TFD (x2_w)")
    plt.subplot(211)
    plt.title("Phase")
    plt.stem(w2, X2_phase)

    plt.subplot(212)
    plt.title("Magnitude")
    plt.stem(w2, X2_magnitude)

    #    plt.show()

    # Section c
    N1 = 25
    x1 = [np.sin(0.1 * np.pi * n + (np.pi / 4)) for n in range(N1)]

    m = np.linspace(-N1 / 2, N1 / 2 - 1, N1)
    window = np.hanning(N1)

    plt.figure()
    plt.plot(2 * 3.14 * m / N1, x1)
    plt.plot(2 * 3.14 * m / N1, window * x1)
    plt.title("x1 avec et sans fenetre de Hanning")
    plt.legend(['sans fenetre', 'avec fenetre'])
    plt.show()


def question2():
    fc = 2000
    fe = 16000
    N = 64
    n = [n for n in range(int((-N / 2) + 1), int((N / 2)+1))]
    m = (fc * N) / fe

    K = (2 * m) + 1

    h_n = [K / N if n == 0 else ((1 / N) * ((np.sin((np.pi * n * K) / N)) / (np.sin((np.pi * n) / N)))) for n in range(int((-N / 2) + 1), int((N / 2)+1))]

    plt.figure("Reponse impulsionnelle h[n]")
    plt.stem(n, h_n)
    plt.show()


if __name__ == "__main__":
    question1()

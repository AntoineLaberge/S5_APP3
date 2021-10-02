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

    x1_n = [np.sin((0.1*np.pi*n)+(np.pi/4)) for n in range(20)]

    X1_m, X1_phase, X1_magnitude = getTFD(x1_n)

    plt.subplot(211)
    plt.title("Fonction")
    plt.stem(x1_n)

    plt.subplot(223)
    plt.title("Phase")
    plt.stem(X1_phase)

    plt.subplot(224)
    plt.title("Magnitude")
    plt.stem(X1_magnitude)

    plt.show()

    x2_n = [(-1)**n for n in range(10)]

    X2_m, X2_phase, X2_magnitude = getTFD(x2_n)

    plt.subplot(211)
    plt.title("Fonction")
    plt.stem(x2_n)

    plt.subplot(223)
    plt.title("Phase")
    plt.stem(X2_phase)

    plt.subplot(224)
    plt.title("Magnitude")
    plt.stem(X2_magnitude)

    plt.show()

    x3_n = [1 if n==10 else 0 for n in range(11)]

    X3_m, X3_phase, X3_magnitude = getTFD(x3_n)

    plt.subplot(211)
    plt.title("Fonction")
    plt.stem(x3_n)

    plt.subplot(223)
    plt.title("Phase")
    plt.stem(X3_phase)

    plt.subplot(224)
    plt.title("Magnitude")
    plt.stem(X3_magnitude)

    plt.show()

    # Section b

    

if __name__ == "__main__":
    question1()
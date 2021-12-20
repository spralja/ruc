import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Planck, electron_mass

if __name__ == '__main__':
    # Physical constants
    h = Planck
    h_bar = h / (2 * np.pi)
    m = Planck ** 2 # Mass of electron

    # Constants in the problem
    a = np.pi / 7
    b = np.pi / 7
    U_0 = 2.8  # idk

    # Chosen values for N
    n1 = 3
    n2 = 4

    # Calculated constents
    k = (n1 * np.pi) / (a + b)

    B = (n2 * np.pi) / b
    print("B = ", B)

    A = np.sqrt(complex(((n2 ** 2) * (h ** 2)) + (8 * m * (b ** 2) * U_0), 0)) / (2 * h_bar * b)
    print("A = ", A)


    # Greek constants
    Beta_2 = 1  # Input
    print("Beta_2 = ", Beta_2)

    # Beta_2 * (((2 * B) / A) * (np.e ** (1j * A * a)) + (np.e ** (1j * k * (a + b))) * ((2 * (-B / A) * np.cos(B * b) + 1j * np.sin(B * b)))) / ((1 + (B / A)) * (np.e ** (1j * a * A))) + (((1-B)/A) * (np.e ** (-1j * A * a))) - (2 * (np.e ** (1j * k * (a + b))) * (np.e ** (-1j * B * b)))

    Alpha_2 = -Beta_2 * ((1 - B / A) * np.e ** (1j * A * a) + (1 + B / A) * np.e ** (-1j * A * a) - 2 * np.e ** (
                1j * (k * a + (k + B) * b))) / (
                          (1 + B / A) * np.e ** (1j * A * a) + (1 - B / A) * np.e ** (-1j * A * a) - 2 * np.e ** (
                              1j * (k * a + (k - B) * b)))
    print("Alpha_2 = ", Alpha_2)

    Alpha_1 = ((1 + B / A) * Alpha_2 + (1 - B / A) * Beta_2) / 2
    print("Alpha_1 = ", Alpha_1)

    Beta_1 = ((1 - B / A) * Alpha_2 + (1 + B / A) * Beta_2) / 2
    print("Beta_1 = ", Beta_1)


    # 2 * Beta_2 * ( (( (-2 * B * (np.e**(-1j*A*a))) / A ) + ((np.e**(1j * k * (a + b))) * 2 * ((1j * np.sin(B * b)) + (B * np.cos(B * b) / A)))) / ( ((1 + (B / A)) * (np.e ** (1j * A * a))) + ( (1 - (B / A))  * (np.e ** (-1j * A * a)) ) - (2 * ((np.e ** (1j * k * (a+b))) * (np.e ** (-1j * B * b))) ) ) )

    def Psi1(_x):
        return (Alpha_1 * (np.e ** (1j * A * _x))) + (Beta_1 * (np.e ** (-1j * A * _x)))


    def Psi2(_x):
        return (Alpha_2 * (np.e ** (1j * B * _x))) + (Beta_2 * (np.e ** (-1j * B * _x)))


    def PsiReal(_x):
            if _x % (b + a) <= a:
                return Psi1(_x).real
            else:
                return Psi2(_x).real


    def PsiImag(_x):
            if (_x ) % (b + a) <= a:
                return Psi1(_x).imag
            else:
                return Psi2(_x).imag

    N = 10000
    aa = 0
    bb = 2 * np.pi / 7

    x = np.linspace(aa, bb, N)

    yReal = [PsiReal(_x) for _x in x]
    yImag = [PsiImag(_x) for _x in x]

    _sum = 0
    for y in yReal:
        _sum += y ** 2
        print(_sum)

    for y in yImag:
        _sum += y ** 2
        print(_sum)

    _sum *= (bb - aa) / N
    print(_sum)

    yRealHat = [y / np.sqrt(_sum) for y in yReal]
    yImagHat = [y / np.sqrt(_sum) for y in yImag]

    _sum = 0
    for y in yRealHat:
        _sum += y ** 2

    for y in yImagHat:
        _sum += y ** 2

    _sum *= (bb - aa) / N
    print(_sum)
    plt.plot(x, yRealHat, x, yImagHat)
    plt.legend(["real", "imag"])
    plt.show()

    print(Psi1(0.9))
    print(Psi2(0.9))
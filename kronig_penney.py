import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Physical constants
    h = 1
    h_bar = h / (2 * np.pi)
    m = 1  # Mass of electron

    # Constants in the problem
    a = 1
    b = 9
    U_0 = 1  # idk

    # Chosen values for N
    n1 = 3
    n2 = 4

    # Calculated constents
    k = (n1 * np.pi) / (a + b)

    A = (n2 * np.pi) / a
    print("A = ", A)
    B = np.sqrt(complex(((n2 ** 2) * (h ** 2)) - (8 * m * (a ** 2) * U_0), 0)) / (2 * h_bar * a)
    print("B = ", B)

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

    def Psi1(x):
        return (Alpha_1 * (np.e ** (1j * A * x))) + (Beta_1 * (np.e ** (-1j * A * x)))


    def Psi2(x):
        return (Alpha_2 * (np.e ** (1j * B * x))) + (Beta_2 * (np.e ** (-1j * B * x)))


    def PsiReal(_x):
        arr = []
        for _x in x:
            if _x % (b + a) <= 1:
                arr.append(Psi2(_x).real)
            else:
                arr.append(Psi1(_x).real)

        return arr

    def PsiImag(_x):
        arr = []
        for _x in x:
            if _x % (b + a) <= 1:
                arr.append(Psi2(_x).imag)
            else:
                arr.append(Psi1(_x).imag)

        return arr


    x = np.linspace(0, 10, 1000)
    yReal = PsiReal(x)
    yImag = PsiImag(x)
    plt.plot(x, yReal, x, yImag)
    plt.legend(["real", "imag"])
    plt.show()


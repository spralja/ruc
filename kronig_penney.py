import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  # Physical constants
  h = 1
  h_bar = h / (2 * np.pi)
  m = 1 # Mass of electron

  # Constants in the problem
  a = 1
  b = 9
  U_0 = 1 # idk

  # Chosen values for N
  n1 = 3
  n2 = 1

  # Calculated constents
  k = (n1 * np.pi) / (a + b)

  A = (n2 * np.pi) / a
  B = np.sqrt( ((n2 ** 2) * (h ** 2)) - (8 * m * (a ** 2) * U_0) ) / (2 * h_bar * a)

  # Greek constants
  Beta_2 = 0 # Input
  Beta_1 = Beta_2 * (((2 * B) / A) * (np.e ** (1j * A * a)) + (np.e ** (1j * k * (a + b))) * ((2 * (-B / A) * np.cos(B * b) + 1j * np.sin(B * b)))) / ((1 + (B / A)) * (np.e ** (1j * a * A))) + (((1-B)/A) * (np.e ** (-1j * A * a))) - (2 * (np.e ** (1j * k * (a + b))) * (np.e ** (-1j * B * b)))

  Alpha_1 = 2 * Beta_2 * ( (( (-2 * B * (np.e**(-1j*A*a))) / A ) + ((np.e**(1j * k * (a + b))) * 2 * ((1j * np.sin(B * b)) + (B * np.cos(B * b) / A)))) / ( ((1 + (B / A)) * (np.e ** (1j * A * a))) + ( (1 - (B / A))  * (np.e ** (-1j * A * a)) ) - (2 * ((np.e ** (1j * k * (a+b))) * (np.e ** (-1j * B * b))) ) ) ) 
  Alpha_2 = -Beta_2 * (((1 - (B / A)) * (np.e**(1j * A * a))) + ((1 + (B / A)) * (np.e**(-1j * A * a))) - (2 * (np.e**(1j * ((k * a) + ((k + B) * b))))) ) / (((1 - (B / A)) * (np.e**(-1j * A * a))) + ((1 + (B / A)) * (np.e**(1j * A * a))) - (2 * (np.e**(1j * ((k * a) + ((k - B) * b))))) )

  def Psi1(x):
    (Alpha_1 * (np.e ** (1j * A * x))) + (Beta_1 * (np.e ** (-1j * A * x)))
  
  def Psi2(x):
    (Alpha_2 * (np.e ** (1j * B * x))) + (Beta_2 * (np.e ** (-1j * B * x)))

  def Psi(x):
    if x - b >= 0:
      return Psi1(x)
    return Psi2(x)

  x = np.linspace(-10, 10, 1000)
  y = Psi1(x)

  plt.plot(x, y)
  plt.show()


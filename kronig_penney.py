import numpy as cmath
import matplotlib.pyplot as plt
import cmath

if __name__ == '__main__':
  # Physical constants
  h = 6.62607015e-34
  h_bar = h / (2 * cmath.pi)
  m = 9.10938356e-31

  # Constants in the problem
  a = 1
  b = 9 
  U_0 = 1 # idk

  # Chosen values for N
  n1 = 3
  n2 = 10

  # Calculated constents
  k = (n1 * cmath.pi) / (a + b)

  A = (n2 * cmath.pi) / a
  print(((n2 ** 2) * (h ** 2)) - (8 * m * (a ** 2) * U_0))
  B = cmath.sqrt( ((n2 ** 2) * (h ** 2)) - (8 * m * (a ** 2) * U_0) ) / (2 * h_bar * a)

  # Greek constants
  Beta_2 = 10 # Input
  Beta_1 = Beta_2 * (((2 * B) / A) * (cmath.e ** (1j * A * a)) + (cmath.e ** (1j * k * (a + b))) * ((2 * (-B / A) * cmath.cos(B * b) + 1j * cmath.sin(B * b)))) / ((1 + (B / A)) * (cmath.e ** (1j * a * A))) + (((1-B)/A) * (cmath.e ** (-1j * A * a))) - (2 * (cmath.e ** (1j * k * (a + b))) * (cmath.e ** (-1j * B * b)))

  Alpha_1 = 2 * Beta_2 * ( (( (-2 * B * (cmath.e**(-1j*A*a))) / A ) + ((cmath.e**(1j * k * (a + b))) * 2 * ((1j * cmath.sin(B * b)) + (B * cmath.cos(B * b) / A)))) / ( ((1 + (B / A)) * (cmath.e ** (1j * A * a))) + ( (1 - (B / A))  * (cmath.e ** (-1j * A * a)) ) - (2 * ((cmath.e ** (1j * k * (a+b))) * (cmath.e ** (-1j * B * b))) ) ) ) 
  Alpha_2 = -Beta_2 * (((1 - (B / A)) * (cmath.e**(1j * A * a))) + ((1 + (B / A)) * (cmath.e**(-1j * A * a))) - (2 * (cmath.e**(1j * ((k * a) + ((k + B) * b))))) ) / (((1 - (B / A)) * (cmath.e**(-1j * A * a))) + ((1 + (B / A)) * (cmath.e**(1j * A * a))) - (2 * (cmath.e**(1j * ((k * a) + ((k - B) * b))))) )

  def Psi1(x):
    (Alpha_1 * (cmath.e ** (1j * A * x))) + (Beta_1 * (cmath.e ** (-1j * A * x)))
  
  def Psi2(x):
    (Alpha_2 * (cmath.e ** (1j * B * x))) + (Beta_2 * (cmath.e ** (-1j * B * x)))

  def Psi(x):
    if x - b >= 0:
      return Psi1(x)
    return Psi2(x)

  x = cmath.linspace(-10, 10, 1000)
  y = Psi1(x)

  plt.plot(x, y)
  plt.show()


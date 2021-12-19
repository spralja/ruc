from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

def V_n(P, n, q):
  return (P - (n ** 2))/q


def recursiveFractionSum(P, cur, maxValue, q):
  if (cur < 1):
    raise Exception("recursiveFractionSum - Recursion depth cannot be below 1")
  if (cur == maxValue):
    return V_n(P, cur * 2, q)
  return V_n(P, cur * 2, q) - (1 / recursiveFractionSum(P, cur + 1, maxValue, q))
    

def calculateEigenValue(P, maxValue, q):
  return recursiveFractionSum(P, 1, maxValue, q)

# 0 => A_2 / A_0
# 1 => A_4 / A_0
# 2 => A_6 / A_0
def evenRecursiveCoefficientCalculation(P, q):
  memo = {
   0: P / q,
   1: (((P - 4)*P) / (q * q)) - 2
  }

  def evenRecursive(depth):
    if depth < 0:
      raise Exception('evenRecursiveCoefficientCalculation - Recursion depth cannot be negative')
    if depth in memo:
      return memo[depth]
    else:
      memo[depth] = (((P - (4*(depth**2))) / q) * evenRecursive(depth-1)) - evenRecursive(depth-2)
      return memo[depth]
  
  return evenRecursive

# 0 => B_2 / B_2 == 1
# 1 => B_4 / B_2
# 2 => B_6 / B_2
def oddRecursiveCoefficientCalculation(P, q):
  memo = {
    0: 1,
    1: (P - 4) / q
  }

  def oddRecursive(depth):
    if depth < 0:
      raise Exception('oddRecursiveCoefficientCalculation - Recursion depth cannot be negative')
    if depth in memo:
      return memo[depth]
    else:
      memo[depth] = (((P - (4 * depth * depth)) / q) * oddRecursive(depth - 1)) - oddRecursive(depth - 2)
      return memo[depth]

  return oddRecursive

def getEvenPsi(coefficients):
  def psi(theta):
    sumTotal = 0
    for index, coefficient in enumerate(coefficients):
      sumTotal += (coefficient) * np.cos(index * theta )
    return sumTotal / np.sqrt(np.pi)
  
  return psi

def getOddPsi(coefficients):
  def psi(theta):
    sumTotal = 0
    for index, coefficient in enumerate(coefficients):
      sumTotal += (coefficient) * np.sin((index + 1) * theta )
    return sumTotal / np.sqrt(np.pi)
  
  return psi

def getX_Prime(x, n, phi):
  return ((n * x) + phi) / (2)

def getPotentialFunction(Re_c, Im_c, phi, n):
  cons = np.sqrt((2*((Re_c ** 2) + (Im_c ** 2)))/np.pi)

  def potentialEnergy(x):
    return cons * np.sin((n*x)+phi)

  return potentialEnergy

if __name__ == '__main__':
  n = 7
  Re_c = -0.00348069 #-0.111812
  Im_c = -0.0337321 #-1.27724e-14

  m = 9.10938356e-31
  h_bar = 1.054571817e-34
  
  A = np.sqrt(2*(Re_c ** 2 + Im_c **2) / np.pi)

  q = (4 * m * A) / (n * h_bar * h_bar)
  depth = 100 # Both how deep the 1+1/(1+1/..) goes and how many coefficients - 1


  oddEiegenValueFuncion = lambda P: calculateEigenValue(P, 100, q);
  evenEiegenValueFuncion = lambda P: calculateEigenValue(P, 100, q) - ((2*q) / P);

  print(f"Calculating Coefficients")
  print(f"q = {q}")
  print(f"depth = {depth}")

  # ! Calculating roots of infinite fraction
  # Change this to another function if u want to get another root or smt
  rootfinderFunction = fsolve

  # For now we assume just 1 root
  oddEiegenValue = rootfinderFunction(oddEiegenValueFuncion, [2])[0]
  evenEiegenValue = rootfinderFunction(evenEiegenValueFuncion, [2])[0]

  print("Calculated the roots (Eigen-values)")

  # ! Using those roots to find coefficients / A_0 or / B_2
  evenCoefficientsOverA_0Array = []
  oddCoefficientsOverB_2Array = []

  evenRecursive = evenRecursiveCoefficientCalculation(evenEiegenValue, q)
  evenRecursive = oddRecursiveCoefficientCalculation(oddEiegenValue, q)

  for j in range(0, depth):
    evenCoefficientsOverA_0Array.append(evenRecursive(j))
    oddCoefficientsOverB_2Array.append(evenRecursive(j))
    print(f"Calculating {j} / {depth}")

  print("Calculated all coefficients over the first coefficient")

  A_0SquaredReciprocal = 2 
  for coefficient in evenCoefficientsOverA_0Array:
    A_0SquaredReciprocal += coefficient ** 2
  
  B_2SquaredReciprocal = 0
  for coefficient in oddCoefficientsOverB_2Array:
    B_2SquaredReciprocal += coefficient ** 2
  
  A_0 = np.sqrt(1/A_0SquaredReciprocal)
  B_2 = np.sqrt(1/B_2SquaredReciprocal)

  print("Calculated first coefficient")

  evenCoefficients = [A_0]
  oddCoefficients = [B_2]

  for coefficientOverA_0 in evenCoefficientsOverA_0Array:
    evenCoefficients.append(coefficientOverA_0 * A_0 * np.sqrt(2))

  for coefficientOverB_2 in oddCoefficientsOverB_2Array:
    oddCoefficients.append(coefficientOverB_2 * B_2 * np.sqrt(2))

  print("Calculated coefficients")
  print("Even:")
  print(evenCoefficients)
  print("Odd:")
  print(oddCoefficients)

  evenPsi = getEvenPsi(evenCoefficients)
  oddPsi = getOddPsi(oddCoefficients)

  a = np.pi / n
  x = np.linspace(-4*a, 4*a, 100)

  phi = np.arctan(Re_c / Im_c)
  xPrime = getX_Prime(x, n, phi)

  potentialEnergy = getPotentialFunction(Re_c, Im_c, phi, n);

  E_y = evenPsi(xPrime)
  O_y = oddPsi(xPrime)
  pot_y = potentialEnergy(x)

  composite = lambda x: evenPsi(getX_Prime(x, n, phi)) ** 2
  H = quad(composite, -4*a, 4*a)[0]
  modEvenCoefficients = []

  for coef in evenCoefficients:
    modEvenCoefficients.append(coef * np.sqrt(H))

  newEvenPsi = getEvenPsi(modEvenCoefficients)
  newComposite = lambda x: (newEvenPsi(getX_Prime(x, n, phi))/ np.sqrt(H)) ** 2
  print(quad(newComposite, -4*a, 4*a))

  plt.title("Graph of Psi Obtained by the Mathieu Equation")
  plt.xlabel('-4a < X < 4a')
  plt.ylabel('Psi(X)')

  plt.plot(x, E_y, label="Psi(X) for even coefficients")
  plt.plot(x, O_y, label="Psi(X) for odd coefficients")
  plt.plot(x, pot_y, label="Potential Energy (V(x))")
  plt.legend(loc="upper left")
  plt.ylim(-5, 8)

  plt.show()

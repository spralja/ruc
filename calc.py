from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def calculate(Delta_N, Delta_W, Alpha_W, Beta_W, Gamma_W, a, A, B, C, D):
  return (
    (
      (2 / (a*a)) - (
        (Alpha_W * Gamma_W * Alpha_W * Gamma_W) +
        (Alpha_W * Delta_W * Alpha_W * Delta_W) +
        (Beta_W * Gamma_W * Beta_W * Gamma_W) +
        (Beta_W * Delta_W * Beta_W * Delta_W)
      )
    ) * (
      (
        (A * B * Beta_W * Delta_W) * (A * B * Beta_W * Delta_W)
      ) / ( 
        (
          (A * B * Beta_W * Delta_W) * (A * B * Beta_W * Delta_W) +
          (B * C * Alpha_W * Delta_W) * (B * C * Alpha_W * Delta_W) +
          (A * D * Beta_W * Gamma_W) * (A * D * Beta_W * Gamma_W) +
          (C * D * Alpha_W * Gamma_W) * (C * D * Alpha_W * Gamma_W)
        ) * (
          Delta_N * Delta_N
        )
      )
    ) 
  )
  # return ((2 / (a*a)) - ((Alpha_W * Gamma_W * Alpha_W * Gamma_W) + (Alpha_W * Delta_W * Alpha_W * Delta_W) + (Beta_W * Gamma_W * Beta_W * Gamma_W) + (Beta_W * Delta_W * Beta_W * Delta_W)))* (((A * B * Beta_W * Delta_W) * (A * B * Beta_W * Delta_W)) /  ((A * B * Beta_W * Delta_W) * (A * B * Beta_W * Delta_W) +(B * C * Alpha_W * Delta_W) * (B * C * Alpha_W * Delta_W) +(A * D * Beta_W * Gamma_W) * (A * D * Beta_W * Gamma_W) +(C * D * Alpha_W * Gamma_W) * (C * D * Alpha_W * Gamma_W))) * (Delta_N * Delta_N)

def whichEquation(x, y, a): 
  sec_x = abs( np.trunc(x / a) )
  sec_y = abs( np.trunc(y / a) )
  return (( sec_x + sec_y + (x * y >= 0)) % 2)

class PsiEnvironment:
  def __init__(self, Delta_N, Delta_W, Alpha_W, Beta_W, Gamma_W, a, A, B, C, D):
    self.Delta_N = Delta_N
    self.Delta_W = Delta_W
    self.Alpha_W = Alpha_W
    self.Beta_W = Beta_W
    self.Gamma_W = Gamma_W
    
    self.a = a
    self.A = A
    self.B = B
    self.C = C
    self.D = D

    self.Beta_N = np.sqrt(calculate(Delta_N, Delta_W, Alpha_W, Beta_W, Gamma_W, a, A, B, C, D))

    self.Alpha_N = (C * Alpha_W * self.Beta_N) / (A * Beta_W)
    self.Gamma_N = (D * Gamma_W * Delta_N) / (B * Delta_W)
    
  
  def graph(self, max, a):
    x = np.linspace(-max, max, 1000)
    y = np.linspace(-max, max, 1000)

    X, Y = np.meshgrid(x, y)
    Z = self.Psi(X, Y, a)

    return [X, Y, Z]

  def Psi(self, x, y, a):
    # Numpy forced me to go branchless :DDDDD
    which = whichEquation(x, y, a)
    return ( ((which == 1) * self.Psi_W(x, y)) + ((which == 0) * self.Psi_N(x, y)) )
  
  def Psi_W(self, x, y):
    return (
      ((self.Alpha_W * np.cos(self.A * x)) + (self.Beta_W * np.sin(self.A * x))) 
      * 
      ((self.Gamma_W * np.cos(self.B * y)) + (self.Delta_W * np.sin(self.B * y)))
    )
    
  def Psi_N(self, x, y):
    return (
      ((self.Alpha_N * np.cos(self.C * x)) + (self.Beta_N * np.sin(self.C * x)))
      *
      ((self.Gamma_N * np.cos(self.D * y)) + (self.Delta_N * np.sin(self.D * y)))
    )

def mulPi_A(v, a):
  return (v * np.pi) / a

if __name__ == '__main__':
  fig = plt.figure()
  ax = plt.axes(projection='3d')

  Delta_N = 0.1
  Delta_W = 0.2
  Alpha_W = 0.3
  Beta_W = 0.04
  Gamma_W = 0.05
  a = 5
  A = mulPi_A(1, a)
  B = mulPi_A(2, a)
  C = mulPi_A(3, a)
  D = mulPi_A(4, a)

  calculator = PsiEnvironment(Delta_N, Delta_W, Alpha_W, Beta_W, Gamma_W, a, A, B, C, D)
  #print(calculator.Beta_N) # <-- Comment this line out and uncomment the rest to draw
  [X, Y, Z] = calculator.graph(5, a)

  ax.contour3D(X, Y, Z, 50, cmap='binary')

  ax.set_title('surface');
  plt.show()


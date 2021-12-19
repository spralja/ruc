import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def memoCalc(results):
  memo = {}

  for line in results:
    [k, n, m, F, G] =  line[0].split(',')

    k = int(k)
    n = int(n)
    m = int(m)
    F = float(F)
    G = float(G)

    cplx = complex(F, G)

  
    if k == 0 and n == 0:
      print(m)
      memo[f"{m}"] = cplx
      memo[f"{-m}"] = cplx.conjugate()
    
  memo["0"] = np.sqrt(np.pi / 2) * (3 * np.log(3+(2 * np.sqrt(2))))
  print(memo)
  return memo



if __name__ == '__main__':
  total_mu = 1919

  with open('./results.csv') as csvfile:
    results = csv.reader(csvfile, delimiter='\n')

    x = np.linspace(-3 * np.pi, 3 * np.pi, 1000)

    memo = memoCalc(results)

    print(memo)

    def func(x):
      sum = 0

      for i in range(-100, 100):
        sum += (memo[f"{i}"] * (np.e ** (1j * i * x)))
      
      return sum
  
    # X, Y = np.meshgrid(x, y)
    yCplx = func(x)
    yArr = []

    for v in yCplx:
      yArr.append(v.real)

    y = np.array(yArr)

    yinv = 1 / y

    print('Calculated XYZ')

    fig = plt.figure()
    #ax = plt.axes(projection='3d')

    #plt.plot(x, y)
    plt.title("1 / 1D potential fourier series")
    plt.plot(x, yinv)
    plt.ylabel("1 / 1D Potential")
    plt.xlabel("X")

    #ax.set_title('surface');
    plt.show()



    
    


    
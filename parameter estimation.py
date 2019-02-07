import math
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def getRegPred(parms, data):
  b1 = parms[0]
  b0 = parms[1]
  preds = b0 + (b1 * data[:, 1])

  plt.plot(data[:, 1], data[:, 0], 'o')
  plt.plot(data[:, 1], preds)
  plt.axis(np.array((-2, 2, -2, 2)))
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.show()

  return preds

def wrapper4fmin(pArray, data):
  def bof(parms):
    predictions = getRegPred(parms, data)
    sd = (predictions - data[:, 0]) ** 2
    return math.sqrt(sd.sum() / sd.size)
  return optimize.minimize(bof, pArray)

def estimate(data):
  # parameter estimation
  startParms = np.array([-1, 0.2]) # starting values for slope and intercept
  res = wrapper4fmin(startParms, data)
  print('x = ')
  print(res.x)
  print('fVal = ', res.fun)
  
def generate_data():
  nDataPts = 20
  rho = 0.8
  intercept = 0
  
  data = np.zeros((nDataPts, 2))
  data[:, 1] = np.random.randn(nDataPts)
  # why?
  data[:, 0] = np.random.randn(nDataPts) * math.sqrt(1 - (rho ** 2)) \
               + (data[:, 1] * rho) + intercept
  return(data)

###############################################################################

def main():
  # generate simulated data
  data = generate_data()
  estimate(data)

if __name__ == '__main__':
  main()
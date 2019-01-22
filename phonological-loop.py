import numpy as np

def phonologicalLoop():
  nReps = 1000      # number of replications

  listLength = 5    # number of list items
  initAct = 1       # initial activation of items
  dRate = 0.8       # decay rate (per second)
  delay = 5         # retention interval (seconds)
  minAct = 0        # minimum activation for recall

  rRange = np.linspace(1.5, 4, 15)
  tRange = np.divide(1, rRange)
  pCor = np.zeros(np.size(rRange))

  i = 0             # index for word lengths

  for tPerWord in tRange:
    for rep in range(1, nReps + 1):
      actVals = np.multiply(np.ones(listLength), initAct)
      
      cT = 0        # time elapsed
      itemReh = -1  # the last item rehearsed

      # rehearsal during the retention interval
      while (cT < delay):
        # rehearsing the next item that is intact raises its
        # activation to the initial level
        intact = np.where(actVals > minAct)[0]
        itemList = np.where(intact > itemReh)[0]
        if (itemList.size > 0):
          itemReh = intact[itemList[0]]
        else:
          itemReh = 0
        actVals[itemReh] = initAct

        # decay (linear in this case)
        actVals = np.subtract(actVals, dRate * tPerWord)
        
        cT += tPerWord

      # record proportion correct
      numRecalled = 0
      for val in actVals:
        if (val > minAct): numRecalled += 1
      pCor[i] += numRecalled / listLength

    i += 1

  # plot results
  import matplotlib.pyplot as plt

  plt.scatter(rRange, np.divide(pCor, nReps))
  plt.xlim([0, 4.5])
  plt.ylim([0, 1])
  plt.xlabel('Speech Rate')
  plt.ylabel('Proportion Correct')
  plt.show()

  return

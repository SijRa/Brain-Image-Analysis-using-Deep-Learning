#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

def PlotLoss(LossHistory):
  _filename = "figures/loss"
  plt.figure()
  plt.xlabel("Batch")
  plt.ylabel("Loss")
  batchAxis = np.arange(1, LossHistory.batch_iter + 1,1)
  plt.plot(batchAxis, LossHistory.batch_loss_pvs, label='pMCI vs sMCI')
  plt.plot(batchAxis, LossHistory.batch_loss_risk, label='HighRisk vs LowRisk')
  plt.legend(frameon=False)
  plt.savefig(_filename + ".png")
  
def PlotAcc(LossHistory):
  _filename = "figures/acc"
  plt.figure()
  plt.xlabel("Batch")
  plt.ylabel("Accuracy")
  batchAxis = np.arange(1, LossHistory.batch_iter + 1,1)
  plt.plot(batchAxis, LossHistory.batch_acc_pvs, label='pMCI vs sMCI')
  plt.plot(batchAxis, LossHistory.batch_acc_risk, label='HighRisk vs LowRisk')
  plt.legend(frameon=False)
  plt.savefig(_filename + ".png")
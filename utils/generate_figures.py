#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

def PlotLoss(fold, LossHistory):
  _filename = "figures/loss" + str(fold)
  plt.figure()
  title_string = "Loss - Fold " + str(fold)
  plt.title(title_string)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  epochAxis = np.arange(1, LossHistory.epoch_iter + 1,1)
  plt.plot(epochAxis, LossHistory.epoch_loss_pvs, label='Train: pMCI vs sMCI')
  plt.plot(epochAxis, LossHistory.epoch_val_loss_pvs, label='Test: pMCI vs sMCI')
  plt.plot(epochAxis, LossHistory.epoch_loss_risk, label='Train: HighRisk vs LowRisk')
  plt.plot(epochAxis, LossHistory.epoch_val_loss_risk, label='Test: HighRisk vs LowRisk')
  plt.legend(frameon=False)
  plt.xticks(epochAxis)
  plt.savefig(_filename + ".png")
  
def PlotAcc(fold, LossHistory):
  _filename = "figures/acc" + str(fold)
  plt.figure()
  title_string = "Accuracy - Fold " + str(fold)
  plt.title(title_string)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  epochAxis = np.arange(1, LossHistory.epoch_iter + 1,1)
  plt.plot(epochAxis, LossHistory.epoch_acc_pvs, label='Train: pMCI vs sMCI')
  plt.plot(epochAxis, LossHistory.epoch_val_acc_pvs, label='Test: pMCI vs sMCI')
  plt.plot(epochAxis, LossHistory.epoch_acc_risk, label='Train: HighRisk vs LowRisk')
  plt.plot(epochAxis, LossHistory.epoch_val_acc_risk, label='Test: HighRisk vs LowRisk')
  plt.xticks(epochAxis)
  plt.legend(frameon=False)
  plt.savefig(_filename + ".png")
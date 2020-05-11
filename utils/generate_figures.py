#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

def PlotConversion(data):
  ax = sns.lineplot(x=data.index, y="categorical_accuracy",
                  data=data)
  ax.set(xlabel="epoch", ylabel="categorical_accuracy")
  ax.set(xticks=data.index)
  
def PlotRisk(data):
  ax = sns.lineplot(x=data.index, y="categorical_crossentropy",
                  data=data)
  ax.set(xlabel="epoch", ylabel="categorical_crossentropy")
  ax.set(xticks=data.index)
  
def PlotBatchAcc(data):
  ax = sns.lineplot(x=data.index, y="categorical_crossentropy",
                  data=data)
  ax.set(xlabel="epoch", ylabel="categorical_crossentropy")
  ax.set(xticks=data.index)
  
def SaveFig(figname, fold):
  _filename = "figures/" + figname
  plt.savefig(_filename + ".png")
  plt.figure()
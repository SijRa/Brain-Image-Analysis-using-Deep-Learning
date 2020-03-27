#!/usr/bin/env python
from tensordash.tensordash import Tensordash

from utils.data_loader import MRI_Loader
from utils.callbacks import History, Scheduler
from utils.preprocess import Train_Test_Split
from utils.model import CNN_Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

def PlotLoss():
  plt.figure()
  plt.title("Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  epochAxis = np.arange(1, LossHistory.epoch_iter + 1,1)
  plt.plot(epochAxis, LossHistory.epoch_losses, label='Train Loss')
  plt.plot(epochAxis, LossHistory.epoch_val_losses, label='Test Loss')
  plt.legend()
  plt.savefig("visuals/epoch-loss2.png")
  
def PlotAcc():
  plt.figure()
  plt.title("Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  epochAxis = np.arange(1, LossHistory.epoch_iter + 1,1)
  plt.plot(epochAxis, LossHistory.epoch_acc, label='Train Loss')
  plt.plot(epochAxis, LossHistory.epoch_val_acc, label='Test Loss')
  plt.legend()
  plt.savefig("visuals/epoch-acc2.png") 

target_width = 256  #192 #256
target_height = 256 #192 #256
target_depth = 166  #160 #166

learning_rate=0.001
batch_size = 4
epochs = 60
num_classes = 3

# Load MRI data
mri_loader = MRI_Loader(target_shape=(target_width, target_height, target_depth), load_size=None)
data = mri_loader.Load_Data()
print("DATASET SIZE: " + str(len(data['labels'])))

# Train test split
X_train, y_train, X_test, y_test = Train_Test_Split(data, test_size=0.3)
print("TRAIN SIZE: " + str(y_train.shape[0]))
print("TEST SIZE: " + str(y_test.shape[0]))

mirrored_strategy = tf.distribute.MirroredStrategy()

print("Mirrored Devices:", mirrored_strategy.num_replicas_in_sync)
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs:", len(physical_devices)) 

# Create tf.data dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_dataset = train_dataset.shuffle(X_train.shape[0])

test_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
test_dataset = test_dataset.shuffle(X_test.shape[0])

train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(2)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(2)

# Generate callbacks
LossHistory = History()

dashboard = Tensordash(
 ModelName = 'CPU-CNN',
 email = 'Sijan_Rana@hotmail.com',
 password = 'BMsij0909')

# Train model
with mirrored_strategy.scope():
  try:
    model = CNN_Model(target_shape=(target_width,target_height,target_depth,1), classes=num_classes, learning_rate=learning_rate)
    model.fit(train_dataset, validation_data=(test_dataset), epochs=epochs, verbose=1, shuffle=True, use_multiprocessing=True,
    callbacks=[dashboard, LearningRateScheduler(Scheduler), EarlyStopping(monitor='val_categorical_accuracy', patience=10), LossHistory])
  except:
    dashboard.sendCrash() # send crash to mobile app

# Generate graphs
PlotLoss()
PlotAcc()
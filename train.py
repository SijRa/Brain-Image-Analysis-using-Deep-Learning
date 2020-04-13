#!/usr/bin/env python

from utils.data_loader import MRI_Loader
from utils.callbacks import History, Scheduler
from utils.preprocess import Stratified_KFolds_Generator
from utils.model import CNN_Model

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers

def PlotLoss(fold):
  _filename = "figures/loss" + str(fold)
  plt.figure()
  title_string = "Loss - Fold " + str(fold)
  plt.title(title_string)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  epochAxis = np.arange(1, LossHistory.epoch_iter + 1,1)
  plt.plot(epochAxis, LossHistory.epoch_losses, label='Train')
  plt.plot(epochAxis, LossHistory.epoch_val_losses, label='Test')
  plt.legend()
  plt.savefig(_filename + ".png")
  
def PlotAcc(fold):
  _filename = "figures/acc" + str(fold)
  plt.figure()
  title_string = "Accuracy - Fold " + str(fold)
  plt.title(title_string)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  epochAxis = np.arange(1, LossHistory.epoch_iter + 1,1)
  plt.plot(epochAxis, LossHistory.epoch_acc, label='Train')
  plt.plot(epochAxis, LossHistory.epoch_val_acc, label='Test')
  plt.legend()
  plt.savefig(_filename + ".png")
  
def RecordFoldInformation(fold):
  PlotLoss(fold)
  PlotAcc(fold)
  fold_accuracy.append(LossHistory.epoch_acc[-1])
  fold_val_accuracy.append(LossHistory.epoch_val_acc[-1])
  
target_width = 192  #192 #256
target_height = 192 #192 #256
target_depth = 160  #160 #166

learning_rate=0.002
batch_size = 4
epochs = 20
output_class_dict = {'class':3, 'pvs':1, 'risk':1} 
dropout_rate = 0
load_size = None

k_folds = 4


features_shape_dict = {'mri':(target_width,target_height,target_depth,1), 'clinical':26}

# Load MRI data
mri_loader = MRI_Loader(target_shape=(target_width,target_height,target_depth), load_size=load_size)
features, labels = mri_loader.Load_Data()

# Dataset Information
dataset_size = len(labels['pvs'])
print("\n--- DATASET INFORMATION ---")
print("DATASET SIZE: " + str(dataset_size))

# KFolds Iterator
kGenerator = Stratified_KFolds_Generator(k_folds)
fold_accuracy = []
fold_val_accuracy = []

fold = 0

# KFolds Training
for train_index, test_index in kGenerator.split(np.asarray(features['clinical']), labels['pvs']):
  fold += 1
  
  X_train_mri, X_test_mri = np.take(np.asarray(features['mri']), train_index, axis=0), np.take(np.asarray(features['mri']), test_index, axis=0)
  X_train_clinical, X_test_clinical = np.take(np.asarray(features['clinical']), train_index, axis=0), np.take(np.asarray(features['clinical']), test_index, axis=0)
  
  y_train_pvs, y_test_pvs = np.take(labels['pvs'], train_index, axis=0), np.take(labels['pvs'], test_index, axis=0)
  y_train_risk, y_test_risk = np.take(labels['risk'], train_index, axis=0), np.take(labels['risk'], test_index, axis=0)

  # Create tf.data dataset
  train_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':X_train_mri, 'clinical_features':X_train_clinical},
  {'StableVsUnstable':y_train_pvs,'HighRiskVsLowRisk':y_train_risk}))
  train_dataset = train_dataset.shuffle(len(y_train_pvs))
  train_dataset = train_dataset.batch(batch_size)
  train_dataset = train_dataset.prefetch(5)
  
  test_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':X_test_mri, 'clinical_features':X_test_clinical},
  {'StableVsUnstable':y_test_pvs,'HighRiskVsLowRisk':y_test_risk}))
  test_dataset = test_dataset.shuffle(len(y_test_pvs))
  test_dataset = test_dataset.batch(batch_size)
  test_dataset = test_dataset.prefetch(5)

  # Generate callbacks
  LossHistory = History()
  callbacks = [EarlyStopping(monitor='val_StableVsUnstable_binary_accuracy', patience=5), LossHistory]
  
  # Model training
  model = CNN_Model(input_shapes=features_shape_dict, output_classes=output_class_dict, learning_rate=learning_rate, dropout_rate=dropout_rate)
  if fold == 1: # Model information
    print("\n--- MODEL INFORMATION ---")
    print(model.summary())
    plot_model(model, 'MODEL.png', show_shapes=True)
  print("\n--- TRAINING INFORMATION ---")
  model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=1, shuffle=True, use_multiprocessing=True, callbacks=callbacks)
  # Generate graphs
  RecordFoldInformation(k_folds)

print("\n--- RESULTS ---")
print("Mean Training Accuracy:",np.mean(np.asarray(fold_accuracy)))
print("Mean Validation Accuracy:",np.mean(np.asarray(fold_val_accuracy)))
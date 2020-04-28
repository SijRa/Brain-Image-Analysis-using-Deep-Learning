#!/usr/bin/env python

from utils.data_loader import MRI_Loader
from utils.callbacks import History, Cross_Val_History, Scheduler
from utils.preprocess import Stratified_KFolds_Generator, Train_Test_Split
from utils.model import MudNet
from utils.generate_figures import PlotLoss, PlotAcc

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import tensorflow as tf

# Hide GPU devices with limited memory
physical_devices = tf.config.list_physical_devices('GPU')                   
for gpu in physical_devices:
  # GPU memory growth for dynamic memory allocation                                           
  tf.config.experimental.set_memory_growth(gpu, True)

# Plot callback information
def RecordFoldInformation(LossHistory):
  PlotLoss(LossHistory)
  PlotAcc(LossHistory)
 
# Data parameters
target_width = 192  #192 #256
target_height = 192 #192 #256
target_depth = 160  #160 #166
clinical_features = 26
features_shape_dict = {'mri':(target_width,target_height,target_depth,1), 'clinical':clinical_features}
output_class_dict = {'class':3, 'pvs':1, 'risk':1}
limit_size = None
test_size = 0.3
k_folds = 4

# Model parameters
epochs = 100
cv_epochs = 10
learning_rate = 0.001
batch_size = 12
dropout_rate = {'mri':0.2,'clinical':0.1}
regularizer = {'mri':0.001,'clinical':0.001,'final':0.0001}

# Load MRI data
mri_loader = MRI_Loader(target_shape=(target_width,target_height,target_depth), load_size=limit_size)
features, labels = mri_loader.Load_Data()

# Dataset Information
dataset_size = len(labels['pvs'])
print("\n--- DATASET INFORMATION ---")
print("DATASET SIZE: " + str(dataset_size))

# Train/test split with stratification for Risk label
mri_train, mri_test, clinical_train, clinical_test, pvs_train, pvs_test, risk_train, risk_test = Train_Test_Split(features, labels, test_size)

# tf.data dataset
prefetch_size = batch_size
train_data = tf.data.Dataset.from_tensor_slices(({'mri_features':mri_train, 'clinical_features':clinical_train},
  {'StableVsUnstable':pvs_train,'HighRiskVsLowRisk':risk_train}))
train_data = train_data.shuffle(len(pvs_train))
train_data = train_data.batch(batch_size)
train_data = train_data.prefetch(prefetch_size)
test_data = tf.data.Dataset.from_tensor_slices(({'mri_features':mri_test, 'clinical_features':clinical_test},
  {'StableVsUnstable':pvs_test,'HighRiskVsLowRisk':risk_test}))
test_data = test_data.shuffle(len(pvs_test))
test_data = test_data.batch(batch_size)
test_data = test_data.prefetch(prefetch_size)

# Generate callbacks
LossHistory = History()
callbacks_inital = [EarlyStopping(monitor='val_StableVsUnstable_loss', patience=10), LossHistory]
                                                 
# Multi-GPU training
strategy = tf.distribute.MirroredStrategy(devices=['/GPU:2','/GPU:3'])

with strategy.scope():
  # Model definition
  model = MudNet(features_shape_dict, output_class_dict, regularizer, dropout_rate, learning_rate=learning_rate)
  print("\n--- MODEL INFORMATION ---")
  print(model.summary())
  plot_model(model, 'MODEL.png', show_shapes=True)
  print("\n--- INITAL TRAINING ---")
  model.fit(train_data, epochs=epochs, validation_data=test_data, verbose=1, shuffle=True, use_multiprocessing=True, callbacks=callbacks_inital)
  RecordFoldInformation(LossHistory)

# K-Folds Training
fold = 0
pvs_fold_accuracy = []
pvs_fold_val_accuracy = []
risk_fold_accuracy = []
risk_fold_val_accuracy = []

# K-Folds Iterator
kGenerator = Stratified_KFolds_Generator(k_folds)
for train_index, test_index in kGenerator.split(np.asarray(features['clinical']), labels['risk']):
  fold += 1
  
  # Extract fold data
  X_train_mri, X_test_mri = np.take(np.asarray(features['mri']), train_index, axis=0), np.take(np.asarray(features['mri']), test_index, axis=0)
  X_train_clinical, X_test_clinical = np.take(np.asarray(features['clinical']), train_index, axis=0), np.take(np.asarray(features['clinical']), test_index, axis=0)
  y_train_pvs, y_test_pvs = np.take(labels['pvs'], train_index, axis=0), np.take(labels['pvs'], test_index, axis=0)
  y_train_risk, y_test_risk = np.take(labels['risk'], train_index, axis=0), np.take(labels['risk'], test_index, axis=0)
  
  # Create tf.data train/test dataset
  train_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':X_train_mri, 'clinical_features':X_train_clinical},
  {'StableVsUnstable':y_train_pvs,'HighRiskVsLowRisk':y_train_risk}))
  train_dataset = train_dataset.shuffle(len(y_train_pvs))
  train_dataset = train_dataset.batch(batch_size)
  train_dataset = train_dataset.prefetch(prefetch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':X_test_mri, 'clinical_features':X_test_clinical},
  {'StableVsUnstable':y_test_pvs,'HighRiskVsLowRisk':y_test_risk}))
  test_dataset = test_dataset.shuffle(len(y_test_pvs))
  test_dataset = test_dataset.batch(batch_size)
  test_dataset = test_dataset.prefetch(prefetch_size)
  
  # Display fold number
  print("\n--- FOLD " + str(fold) + " TRAINING ---")
  print("\n--- TRAINING INFORMATION ---")
  
  #K-Fold callbacks
  callback = Cross_Val_History()
  callbacks_kfold = [callback]
  
  # Train model
  with strategy.scope():  
    model.fit(train_dataset, epochs=cv_epochs, validation_data=test_dataset, verbose=1, shuffle=True, use_multiprocessing=True, callbacks=callbacks_kfold)
  
  # Save fold data
  pvs_fold_accuracy.append(callback.epoch_acc_pvs[-1])
  pvs_fold_val_accuracy.append(callback.epoch_val_acc_pvs[-1])
  risk_fold_accuracy.append(callback.epoch_acc_risk[-1])
  risk_fold_val_accuracy.append(callback.epoch_val_acc_risk[-1])

# Ouput K-Fold results
print("\n--- RESULTS ---")
print("sMCI vs pMCI")
print("Mean Training Accuracy:",np.mean(np.asarray(pvs_fold_accuracy)))
print("Mean Validation Accuracy:",np.mean(np.asarray(pvs_fold_val_accuracy)))
print("\nRisk")
print("Mean Training Accuracy:",np.mean(np.asarray(risk_fold_accuracy)))
print("Mean Validation Accuracy:",np.mean(np.asarray(risk_fold_val_accuracy)))
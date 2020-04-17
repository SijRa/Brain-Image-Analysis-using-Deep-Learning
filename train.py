#!/usr/bin/env python

from utils.data_loader import MRI_Loader
from utils.callbacks import History, Scheduler
from utils.preprocess import Stratified_KFolds_Generator
from utils.model import CNN_Model
from utils.generate_figures import PlotLoss, PlotAcc

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
  
def RecordFoldInformation(fold, LossHistory):
  PlotLoss(fold, LossHistory)
  PlotAcc(fold, LossHistory)
  fold_accuracy.append(LossHistory.epoch_acc_pvs[-1])
  fold_val_accuracy.append(LossHistory.epoch_val_acc_pvs[-1])
  fold_loss.append(LossHistory.epoch_loss_pvs[-1])
  fold_val_loss.append(LossHistory.epoch_val_loss_pvs[-1])
  
target_width = 192  #192 #256
target_height = 192 #192 #256
target_depth = 160  #160 #166

output_class_dict = {'class':3, 'pvs':1, 'risk':1}
load_size = None
batch_size = 4
epochs = 10
learning_rate=0.0015
dropout_rate = 0.1
clinical_regularizer=0.05
risk_regularizer=0.1

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
fold_loss = []
fold_val_loss = []

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
  model = CNN_Model(input_shapes=features_shape_dict, output_classes=output_class_dict, learning_rate=learning_rate, dropout_rate=dropout_rate,
    clinical_regularizer=clinical_regularizer, risk_regularizer=risk_regularizer)
  if fold == 1: # Model information
    print("\n--- MODEL INFORMATION ---")
    print(model.summary())
    plot_model(model, 'MODEL.png', show_shapes=True)
  # Display fold number
  print("\n--- FOLD " + str(fold) + " TRAINING ---")
  print("\n--- TRAINING INFORMATION ---")
  model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=1, shuffle=True, use_multiprocessing=True, callbacks=callbacks)
  # Generate graphs
  RecordFoldInformation(fold, LossHistory)

print("\n--- RESULTS ---")
print("Mean Training Accuracy:",np.mean(np.asarray(fold_accuracy)))
print("Mean Validation Accuracy:",np.mean(np.asarray(fold_val_accuracy)))
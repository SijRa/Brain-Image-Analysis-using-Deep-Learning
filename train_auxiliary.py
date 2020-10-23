#!/usr/bin/env python

import time

from utils.data_loader_auxiliary import MRI_Loader
from utils.callbacks import Metrics_Class, LR_Plateau
from utils.preprocess import Stratified_KFolds_Generator, Train_Test_Split_Auxiliary
from utils.models import MudNet_Auxiliary

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import pandas as pd
import tensorflow as tf

# Hide GPU devices with limited memory
physical_devices = tf.config.list_physical_devices('GPU')        
for gpu in physical_devices:
  # GPU memory growth for dynamic memory allocation                                           
  tf.config.experimental.set_memory_growth(gpu, True)

# Data parameters
target_width = 197
target_height = 233
target_depth = 189
clinical_features = 14
features_shape_dict = {'mri':(target_width,target_height,target_depth,1), 'clinical':clinical_features}
output_class_dict = {'ad_cn':1}
limit_size = None
test_size = 0.2

# Model parameters
epochs = 100
learning_rate = 0.05
batch_size = 19
prefetch_size = batch_size
dropout_rate = {'mri':0.5,'clinical':0.1}
regularizer = {'mri':0.005,'clinical':0.005,'fc':0.005}

# Load MRI data
mri_loader = MRI_Loader(target_shape=(target_width,target_height,target_depth), load_size=limit_size)
features, labels = mri_loader.Load_Data()

# Dataset Information
dataset_size = len(labels['class'])
print("\n--- DATASET INFORMATION ---")
print("DATASET SIZE: " + str(dataset_size))

# selective GPUs: devices=['/gpu:1','/gpu:2','/gpu:3']
strategy = tf.distribute.MirroredStrategy()

train_times = []

acc = []
recall = []
auc = []

with strategy.scope():
  # Model definition
  model = MudNet_Auxiliary(features_shape_dict, output_class_dict, regularizer, dropout_rate, learning_rate)
  
  # Display model info
  print("\n--- MODEL INFORMATION ---")
  print(model.summary())
    
  # Generate callbacks
  Record_Metrics = Metrics_Class()
  Plateau_Decay = LR_Plateau(factor=0.1, patience=2)
  callbacks_inital = [EarlyStopping(monitor='val_loss', patience=10), Record_Metrics, Plateau_Decay]
  
  #class_weights = compute_class_weight('balanced', np.unique(labels['class']), labels['class'])
  #class_weight = dict(enumerate(class_weights))
  
  # Create split training/test dataset
  mri_train, mri_test, clinical_train, clinical_test, class_train, class_test = Train_Test_Split_Auxiliary(features, labels, test_size)
  
  # Create tf.data train/test dataset
  train_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':mri_train, 'clinical_features':clinical_train},
  {'ad_cn':class_train}))
  train_dataset = train_dataset.batch(batch_size)
  train_dataset = train_dataset.prefetch(prefetch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':mri_test, 'clinical_features':clinical_test},
  {'ad_cn':class_test}))
  test_dataset = test_dataset.batch(batch_size)
  test_dataset = test_dataset.prefetch(prefetch_size)
  
  # Timer
  start = time.time()
  # Training
  model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=1, shuffle=True, use_multiprocessing=True, callbacks=callbacks_inital)
  end = time.time()
    
  model.save("mudnet_auxiliary")
    
  train_time = (end-start)/60
  print("Total training time: " + str(train_time) + " min")
  train_times.append(train_time)
    
  acc.append(Record_Metrics.acc_score[-1])
  recall.append(Record_Metrics.recall_score[-1])
  auc.append(Record_Metrics.auc_score[-1])

print("\n --- FINAL TEST RESULTS ---")
print("Final avg. training time:", np.mean(np.asarray(train_times)))
print()
for i in range(len(acc)):
  print("Accuracy: " + str(acc[i]) + "\tAUC: " + str(auc[i])+ "\tRecall: " + str(recall[i]))
print()
print("Mean Accuracy:",np.mean(np.asarray(acc)))
print("Mean AUC:",np.mean(np.asarray(auc)))
print("Mean Recall:",np.mean(np.asarray(recall)))
print()
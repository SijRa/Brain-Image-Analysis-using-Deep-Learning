#!/usr/bin/env python

import time

from utils.data_loader import MRI_Loader
from utils.callbacks import Metrics, LR_Plateau
from utils.preprocess import Stratified_KFolds_Generator, Train_Test_Split, One_Hot_Encode
from utils.model import MudNet
from utils.generate_figures import PlotConversion, PlotRisk, SaveFig

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import tensorflow as tf

# Hide GPU devices with limited memory
physical_devices = tf.config.list_physical_devices('GPU')        
for gpu in physical_devices:
  # GPU memory growth for dynamic memory allocation                                           
  tf.config.experimental.set_memory_growth(gpu, True)

# Data parameters
iterations = 10
target_width = 192  #192 #256
target_height = 192 #192 #256
target_depth = 160  #160 #166
clinical_features = 26
features_shape_dict = {'mri':(target_width,target_height,target_depth,1), 'clinical':clinical_features}
output_class_dict = {'class':3, 'conversion':2, 'risk':3}
limit_size = None
test_size = 0.2
k_folds = 4

# Model parameters
epochs = 50
learning_rate = 0.005
batch_size = 24
prefetch_size = batch_size
dropout_rate = {'conv':0.2,'fc':0.1}
regularizer = {'conv':0.001,'clinical':0.001,'fc':0.001} #0.0001

# Load MRI data
mri_loader = MRI_Loader(target_shape=(target_width,target_height,target_depth), load_size=limit_size)
features, labels = mri_loader.Load_Data()

# Dataset Information
dataset_size = len(labels['conversion'])
print("\n--- DATASET INFORMATION ---")
print("DATASET SIZE: " + str(dataset_size))

# Multi-GPU training
strategy = tf.distribute.MirroredStrategy(devices=['/GPU:1'])

test_conversion = []
test_risk = []
train_times = []

with strategy.scope():
  # Iterations of K-Fold testing
  for i in range(iterations):
  
    # Model definition
    model = MudNet(features_shape_dict, output_class_dict, regularizer, dropout_rate, learning_rate)
    # Display model info
    if (i == 0):
      print("\n--- MODEL INFORMATION ---")
      print(model.summary())
      plot_model(model, 'MODEL.png', show_shapes=True)
  
    # Generate callbacks
    Record_Metrics = Metrics()
    Plateau_Decay = LR_Plateau(factor=0.2, patience=5)
    callbacks_inital = [EarlyStopping(monitor='val_loss', patience=10), Plateau_Decay, Record_Metrics]
    
    # Create split training/test dataset
    mri_train, mri_test, clinical_train, clinical_test, conversion_train, conversion_test, risk_train, risk_test = Train_Test_Split(features, labels, test_size)
    
    # One-Hot Encoding
    encoded_train_conversion = One_Hot_Encode(conversion_train, output_class_dict['conversion'])
    encoded_test_conversion = One_Hot_Encode(conversion_test, output_class_dict['conversion'])
    encoded_train_risk = One_Hot_Encode(risk_train, output_class_dict['risk'])
    encoded_test_risk = One_Hot_Encode(risk_test, output_class_dict['risk'])
    
    # Create tf.data train/test dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':mri_train, 'clinical_features':clinical_train},
    {'Conversion':encoded_train_conversion,'Risk':encoded_train_risk}))
    train_dataset = train_dataset.shuffle(encoded_train_risk.shape[0])
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(prefetch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':mri_test, 'clinical_features':clinical_test},
    {'Conversion':encoded_test_conversion,'Risk':encoded_test_risk}))
    test_dataset = test_dataset.shuffle(encoded_test_risk.shape[0])
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(prefetch_size)
    
    print("\n --- ITERATION " + str(i+1) + " ---")
    
    # Timer
    start = time.time()
    # Training
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=1, shuffle=True, use_multiprocessing=True, callbacks=callbacks_inital)
    end = time.time()  

    test_conversion.append(Record_Metrics.val_acc_conversion[-1])
    test_risk.append(Record_Metrics.val_acc_risk[-1])
    
    train_time = (end-start)/60
    print("Total training time: " + str(train_time) + " min")
    train_times.append(train_time)

print("\n --- FINAL TEST RESULTS ---")
print("Final avg. training time:", np.mean(np.asarray(train_times)))
print()
for i in range(len(test_conversion)):
  print("Conversion: " + str(test_conversion[i]) + "\tRisk: " + str(test_risk[i]))
print()
print("Conversion Accuracy:",np.mean(np.asarray(test_conversion)))
print("Risk Accuracy:",np.mean(np.asarray(test_risk)))
print()
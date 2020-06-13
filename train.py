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

from tensorflow.math import confusion_matrix

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
target_width = 197
target_height = 233
target_depth = 189
clinical_features = 14
features_shape_dict = {'mri':(target_width,target_height,target_depth,1), 'clinical':clinical_features}
output_class_dict = {'class':3, 'conversion':1, 'risk':3}
limit_size = None
test_size = 0.2

# Model parameters
epochs = 100
learning_rate = 0.05
batch_size = 20
prefetch_size = batch_size
dropout_rate = {'mri':0.5,'clinical':0.1}
regularizer = {'mri':0.005,'clinical':0.005,'fc':0.005}

# Load MRI data
mri_loader = MRI_Loader(target_shape=(target_width,target_height,target_depth), load_size=limit_size)
features, labels = mri_loader.Load_Data()

# Dataset Information
dataset_size = len(labels['conversion'])
print("\n--- DATASET INFORMATION ---")
print("DATASET SIZE: " + str(dataset_size))


# selective GPUs: devices=['/gpu:1','/gpu:2','/gpu:3']
strategy = tf.distribute.MirroredStrategy()

test_conversion = []
test_risk = []
train_times = []

test_conversion_auc = []
test_conversion_recall = []

test_risk_auc = []
test_risk_recall = []

with strategy.scope():
  # Training iterations
  for i in range(iterations):
    # Model definition
    model = MudNet(features_shape_dict, output_class_dict, regularizer, dropout_rate, learning_rate)
    # Display model info
    if (i == 0):
      print("\n--- MODEL INFORMATION ---")
      print(model.summary())
      #plot_model(model, to_file='MODEL.png', show_shapes=True, show_layer_names=False)
  
    # Generate callbacks
    Record_Metrics = Metrics()
    Plateau_Decay = LR_Plateau(factor=0.1, patience=2)
    callbacks_inital = [EarlyStopping(monitor='val_loss', patience=15), Plateau_Decay, Record_Metrics]
    
    # Create split training/test dataset
    mri_train, mri_test, clinical_train, clinical_test, conversion_train, conversion_test, risk_train, risk_test = Train_Test_Split(features, labels, test_size)
    
    # One-Hot Encoding
    encoded_train_risk = One_Hot_Encode(risk_train, output_class_dict['risk'])
    encoded_test_risk = One_Hot_Encode(risk_test, output_class_dict['risk'])
    
    # Create tf.data train/test dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':mri_train, 'clinical_features':clinical_train},
    {'Conversion':conversion_train,'Risk':encoded_train_risk}))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(prefetch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':mri_test, 'clinical_features':clinical_test},
    {'Conversion':conversion_test,'Risk':encoded_test_risk}))
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
    
    test_conversion_auc.append(Record_Metrics.val_auc_conversion[-1])
    test_conversion_recall.append(Record_Metrics.val_recall_conversion[-1])
    
    test_risk_auc.append(Record_Metrics.val_auc_risk[-1])
    test_risk_recall.append(Record_Metrics.val_recall_risk[-1])
    
    train_time = (end-start)/60
    print("Total training time: " + str(train_time) + " min")
    train_times.append(train_time)

print("\n --- FINAL TEST RESULTS ---")
print("Final avg. training time:", np.mean(np.asarray(train_times)))
print()
for i in range(len(test_conversion)):
  print("Conversion: " + str(test_conversion[i]) + "\tAUC: " + str(test_conversion_auc[i])+ "\tRecall: " + 
  str(test_conversion_recall[i]) + "\tRisk: " + str(test_risk[i]) + "\tAUC: " + str(test_risk_auc[i])+ "\tRecall: " + 
  str(test_risk_recall[i]))
print()
print("Conversion Accuracy:",np.mean(np.asarray(test_conversion)))
print("Risk Accuracy:",np.mean(np.asarray(test_risk)))
print()
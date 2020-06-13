#!/usr/bin/env python

import time

from utils.data_loader import MRI_Loader
from utils.callbacks import Metrics, LR_Plateau
from utils.preprocess import Stratified_KFolds_Generator, Train_Test_Split, One_Hot_Encode
from utils.model import MudNet
from utils.generate_figures import PlotConversion, PlotRisk

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

# Sort information
dataframe = pd.DataFrame(columns=['Conversion', 'Risk'])

# Save fold information
def RecordFoldInformation(Record_Metrics, fold):
  mean_conversion = np.mean(np.asarray(Record_Metrics.val_acc_conversion))
  mean_risk = np.mean(np.asarray(Record_Metrics.val_acc_risk))
  fold_scores_Conversion.append(np.mean(mean_conversion))
  fold_scores_Risk.append(np.mean(mean_risk))
  print("\n--- FOLD " + str(fold) + " RESULTS ---")
  print("Conversion avg. val_accuracy:", mean_conversion)
  print("Risk avg. val_accuracy:", mean_risk)

# Data parameters
target_width = 197
target_height = 233
target_depth = 189
clinical_features = 14
features_shape_dict = {'mri':(target_width,target_height,target_depth,1), 'clinical':clinical_features}
output_class_dict = {'class':3, 'conversion':1, 'risk':3}
limit_size = None
k_folds = 10

# Model parameters
epochs = 100
learning_rate = 0.05
batch_size = 24
prefetch_size = batch_size
dropout_rate = {'mri':0.5,'clinical':0.3}
regularizer = {'mri':0.0005,'clinical':0.0005,'fc':0.0005}

# Load MRI data
mri_loader = MRI_Loader(target_shape=(target_width,target_height,target_depth), load_size=limit_size)
features, labels = mri_loader.Load_Data()

# Dataset Information
dataset_size = len(labels['conversion'])
print("\n--- DATASET INFORMATION ---")
print("DATASET SIZE: " + str(dataset_size))

# Multi-GPU training
strategy = tf.distribute.MirroredStrategy(devices=['/gpu:1','/gpu:2','/gpu:3'])

fold_scores_Conversion = []
fold_scores_Risk = []

conv_scores = []
risk_scores = []
train_times = []

with strategy.scope():
  
  # Generate callbacks
  Record_Metrics = Metrics()  
  Plateau_Decay = LR_Plateau(factor=0.3, patience=3)
  callbacks_inital = [EarlyStopping(monitor='val_loss', patience=15), Plateau_Decay, Record_Metrics]
    
  # K-Fold Generator
  kGenerator = Stratified_KFolds_Generator(k_folds)
    
  fold = 1
    
  # K-Folds Iterator
  for train_index, test_index in kGenerator.split(features['mri'], labels['risk']):
    # Model definition
    model = MudNet(features_shape_dict, output_class_dict, regularizer, dropout_rate, learning_rate)
    model.save_weights('weights_inital0005.tf')
    # Display model info
    if fold == 1:
      print("\n--- MODEL INFORMATION ---")
      print(model.summary())
    #plot_model(model, 'MODEL.png', show_shapes=True)
    
    print("\n--- FOLD " + str(fold) + " TRAINING ---")
      
    # One-Hot Encoding
    encoded_risk = One_Hot_Encode(labels['risk'], output_class_dict['risk'])
    
    # Extract train/validation split from training data
    X_train_mri, X_val_mri = np.take(features['mri'], train_index, axis=0), np.take(features['mri'], test_index, axis=0)
    X_train_clinical, X_val_clinical = np.take(features['clinical'], train_index, axis=0), np.take(features['clinical'], test_index, axis=0)
    y_train_conversion, y_val_conversion = np.take(labels['conversion'], train_index, axis=0), np.take(labels['conversion'], test_index, axis=0)
    y_train_risk, y_val_risk = np.take(encoded_risk, train_index, axis=0), np.take(encoded_risk, test_index, axis=0)
    
    # Create tf.data train/test dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':X_train_mri, 'clinical_features':X_train_clinical},
    {'Conversion':y_train_conversion,'Risk':y_train_risk}))
    train_dataset = train_dataset.shuffle(y_train_conversion.shape[0])
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(prefetch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(({'mri_features':X_val_mri, 'clinical_features':X_val_clinical},
    {'Conversion':y_val_conversion,'Risk':y_val_risk}))
    val_dataset = val_dataset.shuffle(y_val_conversion.shape[0])
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(prefetch_size)
    
    model.load_weights('weights_inital0005.tf')
    
    # Timer
    start = time.time()
    # Train
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=1, shuffle=True, use_multiprocessing=True, callbacks=callbacks_inital)
    end = time.time()  
    
    # Record information
    #RecordFoldInformation(Record_Metrics, fold)
    fold += 1
      
    # Ouput K-Fold results
    #mean_conv = np.mean(np.asarray(fold_scores_Conversion))
    #mean_risk = np.mean(np.asarray(fold_scores_Risk))
    
    conv_acc = Record_Metrics.val_acc_conversion[-1]
    risk_acc = Record_Metrics.val_acc_risk[-1]
    
    train_time = (end-start)/60
    print("\n--- FOLD RESULTS ---")
    print("sMCI vs pMCI")
    print("Mean Fold Accuracy:", conv_acc)
    print("Risk")
    print("Mean Fold Accuracy:", risk_acc)
    print("Total training time: " + str(train_time) + " min")
    
    conv_scores.append(conv_acc)
    risk_scores.append(risk_acc)
    train_times.append(train_time)

# Output results
print("\n --- FINAL FOLD RESULTS ---")
for i in range(len(conv_scores)):
  print("Conversion: " + str(conv_scores[i]) + "\tRisk: " + str(risk_scores[i]) + "\tTime: " + str(train_times[i]) + " mins")
print()
print("Final avg. Conversion:", np.mean(np.asarray(conv_scores)))
print("Final avg. Risk:", np.mean(np.asarray(risk_scores)))
print("Final avg. training time:", np.mean(np.asarray(train_times)))
print(0.0005)
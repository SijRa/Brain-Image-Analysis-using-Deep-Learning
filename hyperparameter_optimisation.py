#!/usr/bin/env python

import optuna
from optuna.integration import KerasPruningCallback
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

import time

from utils.data_loader import MRI_Loader
from utils.callbacks import Metrics_Conversion_Risk, LR_Plateau
from utils.preprocess import Stratified_KFolds_Generator, Train_Test_Split, One_Hot_Encode
from utils.models import MudNet

from tensorflow.keras.models import load_model

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, Dropout, MaxPooling3D, concatenate, BatchNormalization, add, ELU
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, Recall, AUC, binary_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class Objective(object):
    def Conv_Layer(self, filters, kernel_size=(3, 4, 3), kernel_regularizer=l2(0.001), dropout_rate=0.3, strides=1):
      def f(_input):
        conv = Conv3D(filters, kernel_size=kernel_size, kernel_regularizer=kernel_regularizer, padding='same', strides=strides)(_input)
        norm = BatchNormalization()(conv)
        elu = ELU()(norm)
        dropped = Dropout(dropout_rate)(elu)
        return MaxPooling3D(pool_size=(3, 3, 3), strides=2)(dropped)
      return f
  
    def Conv_ResidualLayer(self, filters, kernel_size=(3, 4, 3), kernel_regularizer=l2(0.001), dropout_rate=0.3, strides=1, residual=None):
      def f(_input):
        conv = Conv3D(filters, kernel_size=kernel_size, kernel_regularizer=kernel_regularizer, padding='same', strides=strides)(_input)
        if (residual!=None):
          conv = add([conv, residual])
        norm = BatchNormalization()(conv)
        elu = ELU()(norm)
        return Dropout(dropout_rate)(elu)
      return f
    
    def Dense_Layer(self, units, kernel_regularizer=l2(0.001), dropout_rate=0.1):
      def f(_input):
        dense = Dense(units, kernel_regularizer=kernel_regularizer)(_input)
        norm = BatchNormalization()(dense)
        elu = ELU()(norm)
        return Dropout(dropout_rate)(elu)
      return f
  
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
 
    def __call__(self, trial):        
        
        dropout_mri = trial.suggest_uniform('dropout_mri', 0.1, 0.7)
        dropout_clinical = trial.suggest_uniform('dropout_clinical', 0.1, 0.5)
        kernel_regularizer = trial.suggest_uniform('kernel_regularizer', 0.01, 0.04)
        learning_rate_param = trial.suggest_uniform('learning_rate', 0.01, 0.05)             
        
        dict_params = {'dropout_mri':dropout_mri,
                       'dropout_clinical':dropout_clinical,
                       'kernel_regularizer':kernel_regularizer,
                       'learning_rate_param':learning_rate_param}
        
        # Input layers
        input_mri = Input(shape=(197,233,189,1), name='mri_features')
        input_clinical = Input(shape=(14), name='clinical_features')
        
        # pre-trained layers
        loaded_model = load_model("mudnet_auxiliary")
        x = loaded_model.layers[-2].output
      
        # prediction layers
        output_conversion = Dense(1, activation='sigmoid', name='Conversion')(x)
        output_risk = Dense(3, activation='softmax', name='Risk')(x)
      
        # Model compilation
        model = Model(inputs=loaded_model.input, outputs=[output_conversion, output_risk], name="MudNet")
        
        auc = AUC()
        recall = Recall()
        
        optimizer = Adam(dict_params['learning_rate_param'])
        
        model.compile(
          loss={
          'Conversion':binary_crossentropy,
          'Risk':categorical_crossentropy},
          optimizer=optimizer,
          metrics={
          'Conversion':[binary_accuracy, auc, recall],
          'Risk':[categorical_accuracy, auc, recall]})
          
        Plateau_Decay = LR_Plateau(factor=0.1, patience=2)
        callbacks = [EarlyStopping(monitor='val_loss', patience=15), Plateau_Decay, KerasPruningCallback(trial, 'val_loss')]
        
        hist = model.fit(self.train_set, epochs=100, validation_data=self.test_set, verbose=1, shuffle=True, use_multiprocessing=True, callbacks=callbacks)
        
        validation_loss = np.min(hist.history['val_loss'])
        return validation_loss

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
output_class_dict = {'conversion':1, 'risk':3}
limit_size = None
test_size = 0.2

# Model parameters
epochs = 100
learning_rate = 0.05
batch_size = 20
prefetch_size = batch_size
dropout_rate = {'mri':0.5,'clinical':0.1}
regularizer = {'mri':0.005,'clinical':0.005,'fc':0.005}

optimizer_direction = 'minimize'
number_of_random_points = 25  # random searches to start opt process
maximum_time = 4*60*60  # seconds

# Load MRI data
mri_loader = MRI_Loader(target_shape=(target_width,target_height,target_depth), load_size=limit_size)
features, labels = mri_loader.Load_Data()

# Dataset Information
dataset_size = len(labels['conversion'])
print("\n--- DATASET INFORMATION ---")
print("DATASET SIZE: " + str(dataset_size))

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  # Model definition
  model = MudNet(features_shape_dict, output_class_dict, regularizer, dropout_rate, learning_rate)
  # Display model info
  print("\n--- MODEL INFORMATION ---")
  print(model.summary())
    
  # Generate callbacks
  Record_Metrics = Metrics_Conversion_Risk()
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
  
  objective = Objective(train_dataset, test_dataset)
 
  optuna.logging.set_verbosity(optuna.logging.WARNING)
  study = optuna.create_study(direction=optimizer_direction,
          sampler=TPESampler(n_startup_trials=number_of_random_points), 
          pruner=SuccessiveHalvingPruner(min_resource='auto', 
                       reduction_factor=4, min_early_stopping_rate=0))
   
  study.optimize(objective, timeout=maximum_time)
   
  # save results
  df_results = study.trials_dataframe()
  df_results.to_csv('df_optuna_results.csv')
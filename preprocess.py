import numpy as np
import tensorflow as tf

def Train_Test_Split(data, test_size=0.3):
  dataset_size = len(data['labels'])
  features = np.asarray(data['features'])
  labels = np.asarray(data['labels'])
  training_samples = round(dataset_size - (dataset_size * test_size))
  train_features = features[:training_samples]
  test_features = features[training_samples:]
  train_labels = labels[:training_samples]
  test_labels = labels[training_samples:]
  return train_features, train_labels, test_features, test_labels
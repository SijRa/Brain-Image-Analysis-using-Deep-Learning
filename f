commit 30897961ba99aecaab7a177b2eac27d93a42d9ee
Author: 554282 <554282@c095.vc-main>
Date:   Tue Feb 25 17:37:17 2020 +0000

    current progress

diff --git a/__pycache__/callbacks.cpython-36.pyc b/__pycache__/callbacks.cpython-36.pyc
new file mode 100644
index 0000000..bc9a975
Binary files /dev/null and b/__pycache__/callbacks.cpython-36.pyc differ
diff --git a/__pycache__/data_loader.cpython-36.pyc b/__pycache__/data_loader.cpython-36.pyc
new file mode 100644
index 0000000..a7c9f1f
Binary files /dev/null and b/__pycache__/data_loader.cpython-36.pyc differ
diff --git a/__pycache__/model.cpython-36.pyc b/__pycache__/model.cpython-36.pyc
new file mode 100644
index 0000000..fbcfefd
Binary files /dev/null and b/__pycache__/model.cpython-36.pyc differ
diff --git a/__pycache__/preprocess.cpython-36.pyc b/__pycache__/preprocess.cpython-36.pyc
new file mode 100644
index 0000000..3573e04
Binary files /dev/null and b/__pycache__/preprocess.cpython-36.pyc differ
diff --git a/callbacks.py b/callbacks.py
new file mode 100644
index 0000000..5e6d97b
--- /dev/null
+++ b/callbacks.py
@@ -0,0 +1,25 @@
+import tensorflow as tf
+from tensorflow.keras.callbacks import EarlyStopping, Callback
+
+def Scheduler(epoch):
+  # This function keeps the learning rate at 0.001 for the first ten epochs and decreases it exponentially after that.
+  if epoch < 10:
+    return 0.001
+  else:
+    return 0.001 * tf.math.exp(0.1 * (10 - epoch))
+
+class History(Callback):
+  def on_train_begin(self, logs={}):
+    self.batch_iter = 0
+    self.epoch_iter = 0
+    self.epoch_losses = []
+    self.epoch_val_losses = []
+    self.epoch_acc = []
+    self.epoch_val_acc = []
+  
+  def on_epoch_end(self, epoch, logs={}):
+    self.epoch_losses.append(logs.get('loss'))
+    self.epoch_val_losses.append(logs.get('val_loss'))
+    self.epoch_acc.append(logs.get('categorical_accuracy'))
+    self.epoch_val_acc.append(logs.get('val_categorical_accuracy'))
+    self.epoch_iter += 1
\ No newline at end of file
diff --git a/data_loader.py b/data_loader.py
new file mode 100644
index 0000000..1232cb0
--- /dev/null
+++ b/data_loader.py
@@ -0,0 +1,76 @@
+#!/usr/bin/env python
+
+import random
+from os import listdir
+import numpy as np
+import nibabel as nib
+from SimpleITK import ReadImage
+from dltk.io.preprocessing import *
+
+class MRI_Loader:
+  
+  def __init__(self, target_shape, load_size=None):
+    self.mri_path = '../ADNI_volumes_customtemplate_float32/'
+    self.xls_path = '../ADNI_clinical_data'
+    self.target_shape = target_shape
+    self.load_size = None if load_size == None else load_size
+  
+  def Get_Class_Information(self, _list):
+    print("\n--- CLASS INFORMATION ---")
+    num_files = len(_list)
+    AD, NL, MCI = 0,0,0
+    for _file in _list:
+      filename_arr = _file.split('_')
+      if filename_arr[0] == 'stableAD':
+        AD += 1
+      elif filename_arr[0] == 'stableNL':
+        NL += 1
+      elif filename_arr[0] == 'stableMCI':
+        MCI += 1
+    print("AD: " + str(AD) + "\n" + str(AD*100/num_files) + "%\n")
+    print("Control: " + str(NL) + "\n" + str(NL*100/num_files)+ "%\n")
+    print("MCI: " + str(MCI) + "\n" + str(MCI*100/num_files)+ "%\n")
+  
+  def Shuffle_List(self, _list):
+    if any(_list):
+      random.shuffle(_list)
+    
+  def Get_Filenames(self):
+    file_names = sorted(listdir(self.mri_path))
+    self.Get_Class_Information(file_names)
+    self.Shuffle_List(file_names)
+    return file_names
+  
+  def Generate_Label(self, p_class):
+    label_dict = {"stableAD":np.array([0,0,1]),
+              "stableMCI":np.array([0,1,0]),
+              "stableNL":np.array([1,0,0])}
+    return label_dict[p_class]
+  
+  def Shape_Constraint(self, _file):
+    image_data = ReadImage(self.mri_path + _file)
+    return True if image_data.GetSize() == self.target_shape else False
+    
+  def Extract_MRI(self, filenames):
+    features, labels = [], []
+    print("\n--- COLLECTING MRI ---")
+    filenames = filenames[:self.load_size] if self.load_size != None else filenames
+    for _file in filenames:
+      print("LOADING MRI - " + _file)
+      image_data = nib.load(self.mri_path + _file).get_data().astype('float32')
+      if self.Shape_Constraint(_file):
+        normalised_data = whitening(image_data) # z-score normalisation
+        normalised_data = np.expand_dims(normalised_data, axis=-1)
+        filename_arr = _file.split('_')
+        label = self.Generate_Label(filename_arr[0])
+        features.append(normalised_data)
+        labels.append(label)
+    return features, labels
+      
+  def Load_Data(self):
+    data = {'features':[], 'labels':[]}
+    filenames = self.Get_Filenames()
+    features, labels = self.Extract_MRI(filenames)
+    data['features'] = features
+    data['labels'] = labels
+    return data
\ No newline at end of file
diff --git a/epoch-acc.png b/epoch-acc.png
new file mode 100644
index 0000000..e225f88
Binary files /dev/null and b/epoch-acc.png differ
diff --git a/epoch-loss.png b/epoch-loss.png
new file mode 100644
index 0000000..377058b
Binary files /dev/null and b/epoch-loss.png differ
diff --git a/model.py b/model.py
new file mode 100644
index 0000000..b2c96e4
--- /dev/null
+++ b/model.py
@@ -0,0 +1,28 @@
+import tensorflow as tf
+from tensorflow.keras.models import Sequential
+from tensorflow.keras.layers import Dense, Flatten, Conv3D, Dropout, MaxPooling3D
+from tensorflow.keras.metrics import categorical_crossentropy
+from tensorflow.keras import optimizers
+
+def Model(target_shape, classes, learning_rate=0.001):
+  model = Sequential()
+  model.add(Conv3D(24, kernel_size=(13, 11, 11), activation='relu', input_shape=target_shape, 
+  padding='same', strides=4))
+  model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2))
+  model.add(Dropout(0.1))
+  model.add(Conv3D(48, kernel_size=(6, 5, 5), activation='relu', padding='same'))
+  model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2))
+  model.add(Dropout(0.1))
+  model.add(Conv3D(24, kernel_size=(4, 3, 3), activation='relu'))
+  model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2))
+  model.add(Dropout(0.1))
+  model.add(Conv3D(8, kernel_size=(2, 2, 2), activation='relu'))
+  model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=2))
+  model.add(Dropout(0.1))
+  model.add(Flatten())
+  model.add(Dense(classes, activation='softmax'))
+  
+  model.compile(loss='categorical_crossentropy',
+                optimizer=optimizers.Adam(learning_rate=learning_rate),
+                metrics=['categorical_accuracy'])
+  return model
\ No newline at end of file
diff --git a/preprocess.py b/preprocess.py
new file mode 100644
index 0000000..9c2b3d2
--- /dev/null
+++ b/preprocess.py
@@ -0,0 +1,13 @@
+import numpy as np
+import tensorflow as tf
+
+def Train_Test_Split(data, test_size=0.3):
+  dataset_size = len(data['labels'])
+  features = np.asarray(data['features'])
+  labels = np.asarray(data['labels'])
+  training_samples = round(dataset_size - (dataset_size * test_size))
+  train_features = features[:training_samples]
+  test_features = features[training_samples:]
+  train_labels = labels[:training_samples]
+  test_labels = labels[training_samples:]
+  return train_features, train_labels, test_features, test_labels
\ No newline at end of file
diff --git a/train.py b/train.py
new file mode 100644
index 0000000..7400642
--- /dev/null
+++ b/train.py
@@ -0,0 +1,86 @@
+#!/usr/bin/env python
+from data_loader import MRI_Loader
+from callbacks import History, Scheduler
+from preprocess import Train_Test_Split
+from model import Model
+
+import pandas as pd
+import numpy as np
+import matplotlib.pyplot as plt
+import tensorflow as tf
+
+from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
+
+def PlotLoss():
+  plt.figure()
+  plt.title("Loss")
+  plt.xlabel("Epoch")
+  plt.ylabel("Loss")
+  epochAxis = np.arange(1, LossHistory.epoch_iter + 1,1)
+  plt.plot(epochAxis, LossHistory.epoch_losses, label='Train Loss')
+  plt.plot(epochAxis, LossHistory.epoch_val_losses, label='Test Loss')
+  plt.legend()
+  plt.savefig("epoch-loss.png")
+  
+def PlotAcc():
+  plt.figure()
+  plt.title("Accuracy")
+  plt.xlabel("Epoch")
+  plt.ylabel("Accuracy")
+  epochAxis = np.arange(1, LossHistory.epoch_iter + 1,1)
+  plt.plot(epochAxis, LossHistory.epoch_acc, label='Train Loss')
+  plt.plot(epochAxis, LossHistory.epoch_val_acc, label='Test Loss')
+  plt.legend()
+  plt.savefig("epoch-acc.png") 
+
+target_width = 256  #192 #256
+target_height = 256 #192 #256
+target_depth = 166  #160 #166
+
+learning_rate=0.001
+batch_size = 4
+epochs = 60
+num_classes = 3
+
+# Load MRI data
+mri_loader = MRI_Loader(target_shape=(target_width, target_height, target_depth), load_size=None)
+data = mri_loader.Load_Data()
+print("DATASET SIZE: " + str(len(data['labels'])))
+
+# Train test split
+X_train, y_train, X_test, y_test = Train_Test_Split(data, test_size=0.3)
+print("TRAIN SIZE: " + str(y_train.shape[0]))
+print("TEST SIZE: " + str(y_test.shape[0]))
+
+# Multi-GPU processing
+mirrored_strategy = tf.distribute.MirroredStrategy()
+batch_size *= mirrored_strategy.num_replicas_in_sync # update batch size for multi-gpu processing
+
+print("Mirrored Devices:", mirrored_strategy.num_replicas_in_sync)
+physical_devices = tf.config.list_physical_devices('GPU') 
+print("Num GPUs:", len(physical_devices)) 
+
+# Create tf.data dataset
+train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
+train_dataset = train_dataset.shuffle(X_train.shape[0])
+
+test_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
+test_dataset = test_dataset.shuffle(X_test.shape[0])
+
+train_dataset = train_dataset.batch(batch_size)
+train_dataset = train_dataset.prefetch(4)
+test_dataset = test_dataset.batch(batch_size)
+test_dataset = test_dataset.prefetch(4)
+
+# Generate callbacks    
+LossHistory = History()
+
+# Train model
+with mirrored_strategy.scope():
+  model = Model(target_shape=(target_width,target_height,target_depth,1), classes=num_classes, learning_rate=learning_rate)
+  model.fit(train_dataset, validation_data=(test_dataset), epochs=epochs, verbose=1, shuffle=True, use_multiprocessing=True,
+  callbacks=[LearningRateScheduler(Scheduler), EarlyStopping(monitor='val_categorical_accuracy', patience=30), LossHistory])
+
+# Generate graphs
+PlotLoss()
+PlotAcc()
\ No newline at end of file

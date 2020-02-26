import tensorflow-gpu as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, Dropout, MaxPooling3D
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import optimizers

def Model(target_shape, classes, learning_rate=0.001):
  model = Sequential()
  model.add(Conv3D(24, kernel_size=(13, 11, 11), activation='relu', input_shape=target_shape, 
  padding='same', strides=4))
  model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2))
  model.add(Dropout(0.1))
  model.add(Conv3D(48, kernel_size=(6, 5, 5), activation='relu', padding='same'))
  model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2))
  model.add(Dropout(0.1))
  model.add(Conv3D(24, kernel_size=(4, 3, 3), activation='relu'))
  model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2))
  model.add(Dropout(0.1))
  model.add(Conv3D(8, kernel_size=(2, 2, 2), activation='relu'))
  model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=2))
  model.add(Dropout(0.1))
  model.add(Flatten())
  model.add(Dense(classes, activation='softmax'))
  
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=learning_rate),
                metrics=['categorical_accuracy'])
  return model
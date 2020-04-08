import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, Dropout, MaxPooling3D
from lib.keras_DepthwiseConv3D import DepthwiseConv3D
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import optimizers
  
def CNN_Model(target_shape, classes, learning_rate=0.001, dropout_rate=0.1):
  
  input_mri = Input(shape=target_shape)
  
  convlayer_1 = Conv3D(18, kernel_size=(13, 11, 11), activation='elu', padding='same', strides=4)(input_mri)
  max_pool_1 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(convlayer_1)
  dropout_1 = Dropout(dropout_rate)(max_pool_1)
  
  convlayer_2 = Conv3D(36, kernel_size=(6, 5, 5), activation='elu', padding='same')(dropout_1)
  max_pool_2 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(convlayer_2)
  dropout_2 = Dropout(dropout_rate)(max_pool_2)
  
  convlayer_3 = Conv3D(18, kernel_size=(4, 3, 3), activation='elu')(dropout_2)
  max_pool_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(convlayer_3)
  dropout_3 = Dropout(dropout_rate)(max_pool_3)
  
  convlayer_4 = Conv3D(6, kernel_size=(2, 2, 2), activation='elu')(dropout_3)
  max_pool_4 = MaxPooling3D(pool_size=(1, 1, 1), strides=2)(convlayer_4)
  dropout_4 = Dropout(dropout_rate)(max_pool_4)
  
  mri_fc = Flatten()(dropout_4)
  
  output_mri = Dense(classes, activation='softmax')(mri_fc)
  
  model = keras.Model(input_mri, output_mri)
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=learning_rate),
                metrics=['categorical_accuracy'])
  return model
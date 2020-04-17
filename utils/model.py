import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, Dropout, MaxPooling3D, concatenate, BatchNormalization
from tensorflow.keras.losses import binary_crossentropy, MSE
from tensorflow.keras.metrics import binary_accuracy, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def CNN_Model(input_shapes, output_classes, learning_rate=0.001, dropout_rate=0.1, clinical_regularizer=0.01, risk_regularizer=0.01):
  
  input_mri = Input(shape=input_shapes['mri'], name='mri_features')
  input_clinical = Input(shape=input_shapes['clinical'], name='clinical_features')
  
  convlayer_1 = Conv3D(18, kernel_size=(13, 11, 11), activation='elu', padding='same', strides=4)(input_mri)
  normalised_batch1 = BatchNormalization()(convlayer_1)
  max_pool_1 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(normalised_batch1)
  dropout_1 = Dropout(dropout_rate)(max_pool_1)
  
  convlayer_2 = Conv3D(36, kernel_size=(6, 5, 5), activation='elu', padding='same')(dropout_1)
  normalised_batch2 = BatchNormalization()(convlayer_2)
  max_pool_2 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(normalised_batch2)
  dropout_2 = Dropout(dropout_rate)(max_pool_2)
  
  convlayer_3 = Conv3D(18, kernel_size=(4, 3, 3), activation='elu')(dropout_2)
  normalised_batch3 = BatchNormalization()(convlayer_3)
  max_pool_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(normalised_batch3)
  
  convlayer_4 = Conv3D(6, kernel_size=(2, 2, 2), activation='elu')(max_pool_3)
  normalised_batch4 = BatchNormalization()(convlayer_4)
  max_pool_4 = MaxPooling3D(pool_size=(1, 1, 1), strides=2)(normalised_batch4)
  
  mri_fc = Flatten()(max_pool_4)
  
  denselayer_1 = Dense(32, activation='elu', kernel_regularizer=l2(clinical_regularizer))(input_clinical)
  
  clinical_fc = Dense(10, activation='elu', kernel_regularizer=l2(clinical_regularizer))(denselayer_1)
  
  mixed_layer = concatenate([mri_fc, clinical_fc])
  
  output_pvs = Dense(output_classes['pvs'], activation='sigmoid', name='StableVsUnstable')(mixed_layer)
  
  dense8 = Dense(3, activation='elu', kernel_regularizer=l2(risk_regularizer))(output_pvs)
  dense9 = Dense(2, activation='elu', kernel_regularizer=l2(risk_regularizer))(dense8)
  
  output_risk = Dense(output_classes['risk'], activation='sigmoid', name='HighRiskVsLowRisk')(dense9)
  
  model = Model(inputs=[input_mri, input_clinical], outputs=[output_pvs, output_risk], name="MRI_CLINICAL_CNN")
  model.compile(
    loss={
    'StableVsUnstable':binary_crossentropy,
    'HighRiskVsLowRisk':MSE},
    optimizer=Adam(learning_rate=learning_rate),
    metrics={
    'StableVsUnstable':binary_accuracy,
    'HighRiskVsLowRisk':binary_accuracy})
  return model
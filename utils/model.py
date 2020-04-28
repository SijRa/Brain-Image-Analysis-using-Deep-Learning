import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, Dropout, MaxPooling3D, concatenate, BatchNormalization
from tensorflow.keras.losses import binary_crossentropy, MSE
from tensorflow.keras.metrics import binary_accuracy, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def MudNet(input_shapes, output_classes, regularizer, dropout_rate, learning_rate = 0.00001):
  
  # Input layers
  input_mri = Input(shape=input_shapes['mri'], name='mri_features')
  input_clinical = Input(shape=input_shapes['clinical'], name='clinical_features')
  
  # Convolutional layer 1 (MRI)
  convlayer_1 = Conv3D(24, kernel_size=(13, 11, 11), activation='elu', kernel_regularizer=l2(regularizer['mri']), padding='same', strides=4)(input_mri)
  normalised_batch1 = BatchNormalization()(convlayer_1)
  max_pool_1 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(normalised_batch1)
  dropout_1 = Dropout(dropout_rate['mri'])(max_pool_1)
  
  # Convolutional layer 2 (MRI)
  convlayer_2 = Conv3D(48, kernel_size=(6, 5, 5), activation='elu', kernel_regularizer=l2(regularizer['mri']), padding='same')(dropout_1)
  normalised_batch2 = BatchNormalization()(convlayer_2)
  max_pool_2 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(normalised_batch2)
  dropout_2 = Dropout(dropout_rate['mri'])(max_pool_2)
  
  # Convolutional layer 3 (MRI)
  convlayer_3 = Conv3D(24, kernel_size=(4, 3, 3), activation='elu', kernel_regularizer=l2(regularizer['mri']), padding='same')(dropout_2)
  normalised_batch3 = BatchNormalization()(convlayer_3)
  max_pool_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(normalised_batch3)
  dropout_3 = Dropout(dropout_rate['mri'])(max_pool_3)
  
  # Convolutional layer 4 (MRI)
  convlayer_4 = Conv3D(8, kernel_size=(4, 3, 3), activation='elu', kernel_regularizer=l2(regularizer['mri']), padding='same')(max_pool_3)
  normalised_batch4 = BatchNormalization()(convlayer_4)
  max_pool_4 = MaxPooling3D(pool_size=(1, 1, 1), strides=2)(normalised_batch4)
  dropout_4 = Dropout(dropout_rate['mri'])(max_pool_4)
  
  # Flattened layer
  mri_fc = Flatten()(dropout_4)
  
  # Dense layer 1 (Clinical)
  denselayer_1 = Dense(20, activation='elu', kernel_regularizer=l2(regularizer['clinical']))(input_clinical)
  normalised_clinical_1 = BatchNormalization()(denselayer_1)
  dropout_clinical_1 = Dropout(dropout_rate['clinical'])(normalised_clinical_1)
  # Dense layer 2 (Clinical)
  denselayer_2 = Dense(20, activation='elu', kernel_regularizer=l2(regularizer['clinical']))(dropout_clinical_1)
  normalised_clinical_2 = BatchNormalization()(denselayer_2)
  dropout_clinical_2 = Dropout(dropout_rate['clinical'])(normalised_clinical_2)
  # Dense layer 3 (Clinical)
  denselayer_3 = Dense(10, activation='elu', kernel_regularizer=l2(regularizer['clinical']))(dropout_clinical_2)
  normalised_clinical_3 = BatchNormalization()(denselayer_3)
  clinical_fc = Dropout(dropout_rate['clinical'])(normalised_clinical_3)
  
  # Mixed layer
  mixed_layer = concatenate([mri_fc, clinical_fc])
  
  # Fully connected layer
  final_dense = Dense(4, activation='elu', kernel_regularizer=l2(regularizer['final']))(mixed_layer)
  final_normalised = BatchNormalization()(final_dense)
  
  # Output layer
  output_pvs = Dense(output_classes['pvs'], activation='sigmoid', name='StableVsUnstable')(final_normalised)
  output_risk = Dense(output_classes['risk'], activation='sigmoid', name='HighRiskVsLowRisk')(final_normalised)
  
  # Model compilation
  model = Model(inputs=[input_mri, input_clinical], outputs=[output_pvs, output_risk], name="MRI_CLINICAL_CNN")
  model.compile(
    loss={
    'StableVsUnstable':binary_crossentropy,
    'HighRiskVsLowRisk':binary_crossentropy},
    optimizer=Adam(learning_rate),
    metrics={
    'StableVsUnstable':[binary_accuracy],
    'HighRiskVsLowRisk':[binary_accuracy]})
  return model
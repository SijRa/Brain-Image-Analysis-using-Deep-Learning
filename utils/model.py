from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, Dropout, MaxPooling3D, concatenate, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow_addons.optimizers import Lookahead

def MudNet(input_shapes, output_classes, regularizer, dropout_rate, learning_rate):
  
  # Input layers
  input_mri = Input(shape=input_shapes['mri'], name='mri_features')
  input_clinical = Input(shape=input_shapes['clinical'], name='clinical_features')
  
  # Convolutional layers (MRI)
  convlayer_1 = Conv3D(24, kernel_size=(11, 13, 11), activation='elu', kernel_regularizer=l2(regularizer['conv']), padding='same', strides=3)(input_mri)
  normalised_batch1 = BatchNormalization()(convlayer_1)
  max_pool_1 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(normalised_batch1)
  dropout_1 = Dropout(dropout_rate['conv'])(max_pool_1)
  convlayer_2 = Conv3D(48, kernel_size=(5, 6, 5), activation='elu', kernel_regularizer=l2(regularizer['conv']), padding='same')(dropout_1)
  normalised_batch2 = BatchNormalization()(convlayer_2)
  max_pool_2 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(normalised_batch2)
  dropout_2 = Dropout(dropout_rate['conv'])(max_pool_2)
  convlayer_3 = Conv3D(96, kernel_size=(3, 4, 3), activation='elu', kernel_regularizer=l2(regularizer['conv']), padding='same')(dropout_2)
  normalised_batch3 = BatchNormalization()(convlayer_3)
  max_pool_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=2)(normalised_batch3)
  dropout_3 = Dropout(dropout_rate['conv'])(max_pool_3)
  convlayer_4 = Conv3D(24, kernel_size=(3, 4, 3), activation='elu', kernel_regularizer=l2(regularizer['conv']), padding='same')(dropout_3)
  normalised_batch4 = BatchNormalization()(convlayer_4)
  max_pool_4 = MaxPooling3D(pool_size=(1, 1, 1), strides=2)(normalised_batch4)
  dropout_4 = Dropout(dropout_rate['conv'])(max_pool_4)
  convlayer_5 = Conv3D(8, kernel_size=(3, 4, 3), activation='elu', kernel_regularizer=l2(regularizer['conv']), padding='valid')(dropout_4)
  normalised_batch5 = BatchNormalization()(convlayer_5)
  max_pool_5 = MaxPooling3D(pool_size=(1, 1, 1), strides=2)(normalised_batch5)
  
  # Flattened layer
  mri_dense = Flatten()(max_pool_5)
  
  # Dense layers (Clinical) 
  denselayer_1 = Dense(20, activation='elu', kernel_regularizer=l2(regularizer['clinical']))(input_clinical)
  normaliseddense_1 = BatchNormalization()(denselayer_1) 
  dropoutdense_1 = Dropout(dropout_rate['fc'])(normaliseddense_1)
  denselayer_2 = Dense(20, activation='elu', kernel_regularizer=l2(regularizer['clinical']))(dropoutdense_1)
  normaliseddense_2 = BatchNormalization()(denselayer_2)
  dropoutdense_2 = Dropout(dropout_rate['fc'])(normaliseddense_2)
  denselayer_3 = Dense(10, activation='elu', kernel_regularizer=l2(regularizer['clinical']))(dropoutdense_2)
  normaliseddense_3 = BatchNormalization()(denselayer_3)
  clinical_dense = Dropout(dropout_rate['fc'])(normaliseddense_3)
  
  # Mixed layer
  mixed_layer = concatenate([mri_dense, clinical_dense])
  
  dense_fc = Dense(5, activation='elu', kernel_regularizer=l2(regularizer['fc']))(mixed_layer)
  final = BatchNormalization()(dense_fc)
  
  # Output layer
  output_conversion = Dense(output_classes['conversion'], activation='softmax', name='Conversion')(final)
  output_risk = Dense(output_classes['risk'], activation='softmax', name='Risk')(final)
  
  # Model compilation
  model = Model(inputs=[input_mri, input_clinical], outputs=[output_conversion, output_risk], name="MudNet")
  optimizer = Adam(learning_rate)
  optimizer = Lookahead(optimizer)
  model.compile(
    loss={
    'Conversion':categorical_crossentropy,
    'Risk':categorical_crossentropy},
    optimizer=optimizer,
    metrics={
    'Conversion':[categorical_accuracy],
    'Risk':[categorical_accuracy]})
  return model
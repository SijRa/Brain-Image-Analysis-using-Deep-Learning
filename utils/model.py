from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, Dropout, MaxPooling3D, concatenate, BatchNormalization, add, ELU
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, Recall, AUC, binary_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def Conv_Layer(filters, kernel_size=(3, 4, 3), kernel_regularizer=l2(0.001), dropout_rate=0.3, strides=1):
  def f(_input):
    conv = Conv3D(filters, kernel_size=kernel_size, kernel_regularizer=kernel_regularizer, padding='same', strides=strides)(_input)
    norm = BatchNormalization()(conv)
    elu = ELU()(norm)
    dropped = Dropout(dropout_rate)(elu)
    return MaxPooling3D(pool_size=(3, 3, 3), strides=2)(dropped)
  return f
  
def Conv_ResidualLayer(filters, kernel_size=(3, 4, 3), kernel_regularizer=l2(0.001), dropout_rate=0.3, strides=1, residual=None):
  def f(_input):
    conv = Conv3D(filters, kernel_size=kernel_size, kernel_regularizer=kernel_regularizer, padding='same', strides=strides)(_input)
    if (residual!=None):
      conv = add([conv, residual])
    norm = BatchNormalization()(conv)
    elu = ELU()(norm)
    return Dropout(dropout_rate)(elu)
  return f
  
def Dense_Layer(units, kernel_regularizer=l2(0.001), dropout_rate=0.1):
  def f(_input):
    dense = Dense(units, kernel_regularizer=kernel_regularizer)(_input)
    norm = BatchNormalization()(dense)
    elu = ELU()(norm)
    return Dropout(dropout_rate)(elu)
  return f


# sMCI vs pMCI
def MudNet(input_shapes, output_classes, regularizer, dropout_rate, learning_rate):
  
  # Input layers
  input_mri = Input(shape=input_shapes['mri'], name='mri_features')
  input_clinical = Input(shape=input_shapes['clinical'], name='clinical_features')
  
  # Convolutional Layers (MRI)
  x = Conv_Layer(24, kernel_size=(11, 13, 11), kernel_regularizer=l2(regularizer['mri']), dropout_rate=dropout_rate['mri'], strides=4)(input_mri)
  x = Conv_Layer(48, kernel_size=(3, 4, 3), kernel_regularizer=l2(regularizer['mri']), dropout_rate=dropout_rate['mri'])(x)

  # Pre-activation and normalisation residual  
  residual = Conv3D(96, kernel_size=(3, 4, 3), kernel_regularizer=l2(regularizer['mri']), padding='same')(x)
  x = BatchNormalization()(residual)
  x = ELU()(x)
  x = Dropout(dropout_rate['mri'])(x)
  x = Conv_ResidualLayer(96, kernel_size=(3, 4, 3), kernel_regularizer=l2(regularizer['mri']), dropout_rate=dropout_rate['mri'])(x)
  x = Conv_ResidualLayer(96, kernel_size=(3, 4, 3), kernel_regularizer=l2(regularizer['mri']), dropout_rate=dropout_rate['mri'])(x)
  x = Conv_ResidualLayer(96, kernel_size=(3, 4, 3), kernel_regularizer=l2(regularizer['mri']), dropout_rate=dropout_rate['mri'], residual=residual)(x)
  
  x = Conv_Layer(24, kernel_size=(3, 4, 3), kernel_regularizer=l2(regularizer['mri']), dropout_rate=dropout_rate['mri'])(x)
  x = Conv_Layer(8, kernel_size=(3, 4, 3), kernel_regularizer=l2(regularizer['mri']), dropout_rate=dropout_rate['mri'])(x)
  
  # Flattened layer
  mri_dense = Flatten()(x)
  
  # Dense layers (Clinical)
  x = Dense_Layer(20, kernel_regularizer=l2(regularizer['clinical']), dropout_rate=dropout_rate['clinical'])(input_clinical)
  x = Dense_Layer(20, kernel_regularizer=l2(regularizer['clinical']), dropout_rate=dropout_rate['clinical'])(x)
  clinical_dense = Dense_Layer(10, kernel_regularizer=l2(regularizer['clinical']), dropout_rate=dropout_rate['clinical'])(x) 
  
  # Mixed layer
  mixed_layer = concatenate([mri_dense, clinical_dense])
  output = Dense_Layer(4, kernel_regularizer=l2(regularizer['fc']), dropout_rate=dropout_rate['clinical'])(mixed_layer)
  
  # Output layers
  output_conversion = Dense(output_classes['conversion'], activation='sigmoid', name='Conversion')(output)
  output_risk = Dense(output_classes['risk'], activation='softmax', name='Risk')(output)
  
  # Model compilation
  model = Model(inputs=[input_mri, input_clinical], outputs=[output_conversion, output_risk], name="MudNet")
  optimizer = Adam(learning_rate)
  model.compile(
    loss={
    'Conversion':binary_crossentropy,
    'Risk':categorical_crossentropy},
    optimizer=optimizer,
    metrics={
    'Conversion':[binary_accuracy, AUC(), Recall()],
    'Risk':[categorical_accuracy, AUC(), Recall()]})
  return model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

default_folds = 5
default_test_size = 0.2

def Stratified_KFolds_Generator(folds=default_folds):
  return StratifiedKFold(n_splits=folds, shuffle=True)

def Train_Test_Split(features, labels, output_class_dict, test_size=default_test_size, stratify='risk'):
  mri_train, mri_test, clinical_train, clinical_test, conversion_train, conversion_test, risk_train, risk_test = train_test_split(features['mri'], features['clinical'],
    labels['conversion'], labels['risk'], test_size=test_size, stratify=labels[stratify], shuffle=True)
  return mri_train, mri_test, clinical_train, clinical_test, conversion_train, conversion_test, risk_train, risk_test

def Train_Test_Split_Auxiliary(features, labels, output_class_dict, test_size=default_test_size, stratify='class'):
  mri_train, mri_test, clinical_train, clinical_test, class_train, class_test = train_test_split(features['mri'], features['clinical'],
    labels['class'], test_size=test_size, stratify=labels[stratify], shuffle=True)
  return mri_train, mri_test, clinical_train, clinical_test, class_train, class_test

def One_Hot_Encode(y, num_classes):
  return to_categorical(y, num_classes)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

default_folds = 3
default_test_size = 0.3

def Stratified_KFolds_Generator(folds=default_folds):
  return StratifiedKFold(n_splits=folds, shuffle=True)

def Train_Test_Split(features, labels, test_size=default_test_size):
  mri_train, mri_test, clinical_train, clinical_test, pvs_train, pvs_test, risk_train, risk_test = train_test_split(features['mri'], features['clinical'],
    labels['pvs'], labels['risk'], test_size=test_size, stratify=labels['risk'], shuffle=True)
  return mri_train, mri_test, clinical_train, clinical_test, pvs_train, pvs_test, risk_train, risk_test
  

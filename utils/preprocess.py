from sklearn.model_selection import StratifiedKFold

default_folds = 3

def Stratified_KFolds_Generator(folds=default_folds):
  return StratifiedKFold(n_splits=folds, shuffle=True)
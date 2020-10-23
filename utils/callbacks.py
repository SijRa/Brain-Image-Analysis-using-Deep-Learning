from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau

def LR_Plateau(factor=0.2, patience=10):
  return ReduceLROnPlateau(moniter='val_loss', factor=factor, patience=patience, verbose=0, mode='min', min_delta=0.001, cooldown=1, min_lr=0)
    
class Metrics_Conversion_Risk(Callback):
  """
  Callbacks for the pMCI vs sMCI and Risk classifications
  """
  def on_train_begin(self, logs={}):
    self.batch_iter = 0
    self.epoch_iter = 0
    # Accuracy
    self.batch_acc_conversion, self.batch_acc_risk = [],[]
    self.val_acc_conversion, self.val_acc_risk = [],[]
    # AUC
    self.val_auc_conversion, self.val_auc_risk = [], []
    # Recall
    self.val_recall_conversion, self.val_recall_risk = [], []
    # Loss
    self.batch_loss_conversion, self.batch_loss_risk = [],[]
    self.val_loss_conversion, self.val_loss_risk = [],[]
  
  def on_batch_end(self, epoch, logs={}):
    # sMCI vs pMCI
    acc_conversion = logs.get('Conversion_binary_accuracy')
    loss_conversion = logs.get('Conversion_loss')
    # High Risk vs Low Risk
    acc_risk = logs.get('Risk_categorical_accuracy')
    loss_risk = logs.get('Risk_loss')
    # Append results
    self.batch_acc_conversion.append(acc_conversion)
    self.batch_loss_conversion.append(loss_conversion)
    self.batch_acc_risk.append(acc_risk)
    self.batch_loss_risk.append(loss_risk)
    self.batch_iter += 1
  
  def on_epoch_end(self, epoch, logs={}):
    # sMCI vs pMCI
    val_acc_conversion = logs.get('val_Conversion_binary_accuracy')
    val_loss_conversion = logs.get('val_Conversion_loss')
    val_auc_conversion = logs.get('val_Conversion_auc')
    val_recall_conversion = logs.get('val_Conversion_recall')
    # High Risk vs Low Risk
    val_acc_risk = logs.get('val_Risk_categorical_accuracy')
    val_loss_risk = logs.get('val_Risk_loss')
    val_auc_risk = logs.get('val_Risk_auc')
    val_recall_risk = logs.get('val_Risk_recall')
    # Append results
    self.val_acc_conversion.append(val_acc_conversion)
    self.val_loss_conversion.append(val_loss_conversion)
    self.val_auc_conversion.append(val_auc_conversion)
    self.val_recall_conversion.append(val_recall_conversion)
    self.val_acc_risk.append(val_acc_risk)
    self.val_loss_risk.append(val_loss_risk)
    self.val_auc_risk.append(val_auc_risk)
    self.val_recall_risk.append(val_recall_risk)
    self.epoch_iter += 1
    
class Metrics_Class(Callback):
  """
  Callbacks to record metrics for the AD vs CN problem
  """
  def on_train_begin(self, logs={}):    
    self.acc_score = []
    self.recall_score = []
    self.auc_score = []
    self.loss_score = []
    
  def on_epoch_end(self, epoch, logs={}):
    # class AD vs CN
    acc = logs.get('val_binary_accuracy')
    loss = logs.get('val_loss')
    auc = logs.get('val_auc')
    recall = logs.get('val_recall')
    # Append results
    self.acc_score.append(acc)
    self.loss_score.append(loss)
    self.auc_score.append(auc)
    self.recall_score.append(recall)
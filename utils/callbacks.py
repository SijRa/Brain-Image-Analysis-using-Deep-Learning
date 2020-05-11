from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau

def LR_Plateau(factor=0.2, patience=10):
  return ReduceLROnPlateau(moniter='val_loss', factor=factor, patience=patience, verbose=1, mode='min', min_delta=0.001, cooldown=0, min_lr=0)
    
class Metrics(Callback):
    
  def on_train_begin(self, logs={}):
    self.batch_iter = 0
    self.epoch_iter = 0
    # Accuracy
    self.batch_acc_conversion, self.batch_acc_risk = [],[]
    self.val_acc_conversion, self.val_acc_risk = [],[]
    # Loss
    self.batch_loss_conversion, self.batch_loss_risk = [],[]
    self.val_loss_conversion, self.val_loss_risk = [],[]
  
  def on_batch_end(self, epoch, logs={}):
    # sMCI vs pMCI
    acc_conversion = logs.get('Conversion_categorical_accuracy')
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
    val_acc_conversion = logs.get('val_Conversion_categorical_accuracy')
    val_loss_conversion = logs.get('val_Conversion_loss')
    # High Risk vs Low Risk
    val_acc_risk = logs.get('val_Risk_categorical_accuracy')
    val_loss_risk = logs.get('val_Risk_loss')
    # Append results
    self.val_acc_conversion.append(val_acc_conversion)
    self.val_loss_conversion.append(val_loss_conversion)
    self.val_acc_risk.append(val_acc_risk)
    self.val_loss_risk.append(val_loss_risk)
    self.epoch_iter += 1
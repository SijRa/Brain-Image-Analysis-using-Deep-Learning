import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LearningRateScheduler

def Schedule(epoch):
  # This function keeps the learning rate at 0.001 for the first ten epochs and decreases it exponentially after that.
  if epoch < 10:
    return 0.005
  else:
    return 0.005 * tf.math.exp(0.1 * (1 - epoch))
    
def Scheduler():
  return LearningRateScheduler(Schedule)
    
class History(Callback):
    
  def on_train_begin(self, logs={}):
    self.batch_iter = 0
    self.epoch_iter = 0
    # Accuracy
    self.batch_acc_pvs, self.batch_acc_risk = [],[]
    self.val_acc_pvs, self.val_acc_risk = [],[]
    # Loss
    self.batch_loss_pvs, self.batch_loss_risk = [],[]
    self.val_loss_pvs, self.val_loss_risk = [],[]
  
  def on_batch_end(self, epoch, logs={}):
    # sMCI vs pMCI
    acc_pvs = logs.get('StableVsUnstable_binary_accuracy')
    loss_pvs = logs.get('StableVsUnstable_loss')
    # High Risk vs Low Risk
    acc_risk = logs.get('HighRiskVsLowRisk_binary_accuracy')
    loss_risk = logs.get('HighRiskVsLowRisk_loss')
    # Append results
    self.batch_acc_pvs.append(acc_pvs)
    self.batch_loss_pvs.append(loss_pvs)
    self.batch_acc_risk.append(acc_risk)
    self.batch_loss_risk.append(loss_risk)
    self.batch_iter += 1
  
  def on_epoch_end(self, epoch, logs={}):
    # sMCI vs pMCI
    val_acc_pvs = logs.get('val_StableVsUnstable_binary_accuracy')
    val_loss_pvs = logs.get('val_StableVsUnstable_loss')
    # High Risk vs Low Risk
    val_acc_risk = logs.get('val_HighRiskVsLowRisk_binary_accuracy')
    val_loss_risk = logs.get('val_HighRiskVsLowRisk_loss')
    # Append results
    self.val_acc_pvs.append(val_acc_pvs)
    self.val_loss_pvs.append(val_loss_pvs)
    self.val_acc_risk.append(val_acc_risk)
    self.val_loss_risk.append(val_loss_risk)
    self.epoch_iter += 1
  
class Cross_Val_History(Callback):
    
  def on_train_begin(self, logs={}):
    self.epoch_iter = 0
    # Accuracy
    self.epoch_acc_pvs, self.epoch_val_acc_pvs = [],[]
    self.epoch_acc_risk, self.epoch_val_acc_risk = [],[]
  
  def on_epoch_end(self, epoch, logs={}):
    # sMCI vs pMCI
    acc_pvs = logs.get('StableVsUnstable_binary_accuracy')
    val_acc_pvs = logs.get('val_StableVsUnstable_binary_accuracy')
    # High Risk vs Low Risk
    acc_risk = logs.get('HighRiskVsLowRisk_binary_accuracy')
    val_acc_risk = logs.get('val_HighRiskVsLowRisk_binary_accuracy')
    # Append results
    self.epoch_acc_pvs.append(acc_pvs)
    self.epoch_val_acc_pvs.append(val_acc_pvs)
    self.epoch_acc_risk.append(acc_risk)
    self.epoch_val_acc_risk.append(val_acc_risk)
    self.epoch_iter += 1
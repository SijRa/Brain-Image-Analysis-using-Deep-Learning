import tensorflow as tf
from tensorflow.keras.callbacks import Callback

def Scheduler(epoch):
  # This function keeps the learning rate at 0.001 for the first ten epochs and decreases it exponentially after that.
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

class History(Callback):
    
  def on_train_begin(self, logs={}):
    self.batch_iter = 0
    self.epoch_iter = 0
    self.epoch_loss_pvs, self.epoch_loss_risk = [],[]
    self.epoch_val_loss_pvs, self.epoch_val_loss_risk = [],[]
    self.epoch_acc_pvs, self.epoch_acc_risk = [],[]
    self.epoch_val_acc_pvs, self.epoch_val_acc_risk = [],[]
  
  def on_epoch_end(self, epoch, logs={}):
    loss_pvs = logs.get('StableVsUnstable_loss')
    val_loss_pvs = logs.get('val_StableVsUnstable_loss')
    acc_pvs = logs.get('StableVsUnstable_binary_accuracy')
    val_acc_pvs = logs.get('val_StableVsUnstable_binary_accuracy')
    
    loss_risk = logs.get('HighRiskVsLowRisk_loss')
    val_loss_risk = logs.get('val_HighRiskVsLowRisk_loss')
    acc_risk = logs.get('HighRiskVsLowRisk_recall')
    val_acc_risk = logs.get('val_HighRiskVsLowRisk_recall')
    
    self.epoch_loss_pvs.append(loss_pvs)
    self.epoch_val_loss_pvs.append(val_loss_pvs)
    self.epoch_acc_pvs.append(acc_pvs)
    self.epoch_val_acc_pvs.append(val_acc_pvs)
    
    self.epoch_loss_risk.append(loss_risk)
    self.epoch_val_loss_risk.append(val_loss_risk)
    self.epoch_acc_risk.append(acc_risk)
    self.epoch_val_acc_risk.append(val_acc_risk)
    
    self.epoch_iter += 1
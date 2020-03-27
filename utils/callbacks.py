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
    self.epoch_losses = []
    self.epoch_val_losses = []
    self.epoch_acc = []
    self.epoch_val_acc = []
  
  def on_epoch_end(self, epoch, logs={}):
    loss = logs.get('loss')
    val_loss = logs.get('val_loss')
    acc = logs.get('categorical_accuracy')
    val_acc = logs.get('val_categorical_accuracy')
    self.epoch_losses.append(loss)
    self.epoch_val_losses.append(val_loss)
    self.epoch_acc.append(acc)
    self.epoch_val_acc.append(val_acc)
    self.epoch_iter += 1

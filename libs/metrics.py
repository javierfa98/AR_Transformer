#################################################################
# Metrics and Loss Functions used during Training and Testing
# Author: Javier Fañanás Anaya
# Email: javierfa@unizar.es
#################################################################

import tensorflow as tf

## Loss Functions:

#Root Mean Square Error
def rmse(y,y_):
  return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y,y_))),axis = [1,2])

#Normalized Root Mean Square Error
def nrmse(y_true, y_pred):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)), axis=[1, 2]))
    max_values = tf.reduce_max(y_true, axis=[1, 2])
    min_values = tf.reduce_min(y_true, axis=[1, 2])
    range_values = max_values - min_values
    nrmse = rmse / range_values
    return tf.reduce_mean(nrmse)

#Mean Square Error
def mse(y,y_):
  return tf.reduce_mean(tf.square(tf.subtract(y,y_)),axis = [1,2])

#Normalized Mean Square Error
def nmse(y,y_):
  mse = tf.reduce_mean(tf.square(tf.subtract(y,y_)),axis = [1,2])
  max_values = tf.reduce_max(y, axis=[1, 2])
  min_values = tf.reduce_min(y, axis=[1, 2])
  range_values = max_values - min_values
  nmse = mse / range_values
  return tf.reduce_mean(nmse)

#R2 Score
def r2(y,y_):
  y = tf.cast(y, tf.float32)  # Ensure y is of type float32
  y_ = tf.cast(y_, tf.float32)  # Ensure y_ is of type float32
  SS_res =  tf.reduce_sum(tf.square(tf.subtract(y,y_)), axis=1)
  SS_tot = tf.reduce_sum(tf.square(tf.subtract(y,tf.reduce_mean(y, axis=1, keepdims=True))), axis=1)
  r2 = tf.reduce_mean(tf.subtract(1.0,tf.divide(SS_res,SS_tot)),axis = 1)
  loss_r2 = -r2 #Negative R2 to use as loss function
  return loss_r2

## Metrics

#Mean Absolute Error
def mae_2dim(y,y_):
  return tf.reduce_mean(tf.abs(tf.subtract(y,y_)))

#Root Mean Square Error
def rmse_2dim(y,y_):
  return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y,y_))))

#Normalized Root Mean Square Error
def nrmse_2dim(y_true, y_pred):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))
    max_value = tf.reduce_max(y_true)
    min_value = tf.reduce_min(y_true)
    range_value = max_value - min_value
    nrmse = rmse / range_value
    return nrmse

#Mean Square Error
def mse_2dim(y,y_):
  return tf.reduce_mean(tf.square(tf.subtract(y,y_)))

#R2 Score
def r2_2dim(y,y_):
  y = tf.cast(y, tf.float32)  # Ensure y is of type float32
  y_ = tf.cast(y_, tf.float32)  # Ensure y_ is of type float32
  SS_res =  tf.reduce_sum(tf.square(tf.subtract(y,y_)))
  SS_tot = tf.reduce_sum(tf.square(tf.subtract(y,tf.reduce_mean(y))))
  r2 = tf.reduce_mean(tf.subtract(1.0,tf.divide(SS_res,SS_tot)))
  loss_r2 = -r2 #Negative R2 to use as loss function
  return loss_r2

#Mean Absolute Error
def mae_2dim(y,y_):
  return tf.reduce_mean(tf.abs(tf.subtract(y,y_)))
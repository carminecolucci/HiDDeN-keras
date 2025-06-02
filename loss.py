import keras
import tensorflow as tf

from const import *

EPS = 1e-8

# The image distortion loss L_i on the paper
@keras.saving.register_keras_serializable(package="HiDDeN")
def image_distortion_loss(y_true, y_pred):
    return tf.norm(y_true - y_pred, ord=2, keepdims=True) / (CHANNEL * IMAGE_SIZE * IMAGE_SIZE)

# The message distortion loss L_m on the paper
@keras.saving.register_keras_serializable(package="HiDDeN")
def message_distortion_loss(y_true, y_pred):
    return tf.norm(y_true - y_pred, ord=2, keepdims=True) / MESSAGE_LENGTH

# The adversarial loss L_g on the paper
@keras.saving.register_keras_serializable(package="HiDDeN")
def adversary_loss(y_true, y_pred):
    # From paper Lg = log(1 - y_pred)
    y_pred_clipped = tf.clip_by_value(y_pred, 0, 1.0 - EPS)
    return -tf.math.log(1.0 - y_pred_clipped)

# The discriminator loss L_a on the paper
@keras.saving.register_keras_serializable(package="HiDDeN")
def discriminator_loss(y_true, y_pred):
    # From paper La = log(1 - y_true) + log(y_pred)
    y_true_clipped = tf.clip_by_value(y_true, 0, 1.0 - EPS)
    y_pred_clipped = tf.clip_by_value(y_pred, EPS, 1.0)
    return -tf.math.log(1.0 - y_true_clipped) - tf.math.log(y_pred_clipped)

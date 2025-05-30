from utils import log10
import tensorflow as tf
import keras
from const import *


# The image distortion loss L_i on the paper
@keras.saving.register_keras_serializable(package="HiDDeN")
def image_distortion_loss(y_true, y_pred):
    return tf.norm(y_true - y_pred, ord=2, keepdims=True) /\
        (CHANNEL*IMAGE_SIZE*IMAGE_SIZE)

# The message distortion loss L_m on the paper
@keras.saving.register_keras_serializable(package="HiDDeN")
def message_distortion_loss(y_true, y_pred):
    return tf.norm(y_true - y_pred, ord=2, keepdims=True) /\
        (MESSAGE_LENGTH)

# The adversarial loss L_g on the paper
@keras.saving.register_keras_serializable(package="HiDDeN")
def adversary_loss(y_true, y_pred):
    return log10(y_pred + 1)

# The discriminator loss L_a on the paper
@keras.saving.register_keras_serializable(package="HiDDeN")
def discriminator_loss(y_true, y_pred):
    return log10(y_true + 1) + log10(y_pred + 1)

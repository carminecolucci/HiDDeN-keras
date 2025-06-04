import numpy as np
import random
import tensorflow as tf

from const import MESSAGE_LENGTH


def rgb2yuv(images):
    yuv = tf.image.rgb_to_yuv(images)
    yuv_offset = tf.stack([
        yuv[..., 0],
        yuv[..., 1] + 0.5,
        yuv[..., 2] + 0.5
    ], axis=-1)
    return yuv_offset

def yuv2rgb(images):
    yuv = tf.stack([
        images[..., 0],
        images[..., 1] - 0.5,
        images[..., 2] - 0.5
    ], axis=-1)
    rgb = tf.image.yuv_to_rgb(yuv)
    return rgb

# Generate a random binary string with size n
def generate_random_binary(n):
    return [np.float32(random.randint(0, 1)) for _ in range(n)]

# Generate a random array of binary messages
def generate_random_messages(N):
    messages_binary = []
    for _ in range(N):
        binary_message = generate_random_binary(MESSAGE_LENGTH)
        messages_binary.append(binary_message)
    return np.asarray(messages_binary, dtype="float32")

# Round every element to 0 or 1
def round_message_to_string(predicted_message):
    rounded_message = ""
    for num in predicted_message:
        if (float(num) > 0.5):
            rounded_message += "1"
        else:
            rounded_message += "0"
    return rounded_message

# Count bit error between the original binary message
# and the predicted one
def count_errors(original_message, predicted_message):
    original_message = round_message_to_string(original_message)
    count = 0
    for i in range(len(original_message)):
        if original_message[i] != predicted_message[i]:
            count += 1
    return count

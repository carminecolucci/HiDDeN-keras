import string
import random
from const import *
import numpy as np
import tensorflow as tf


# Generate a random binary string with size n
def generate_random_binary(n):
    return [np.float32(random.randint(0, 1)) for _ in range(n)]

# Generate a random array of binary messages
def generate_random_messages(N):
    messages_binary = []
    for _ in range(N):
        binary_message = generate_random_binary(MESSAGE_LENGTH)
        messages_binary.append(binary_message)
    return np.asarray(messages_binary, dtype='float32')

# Round every element to 0 or 1
def round_message_to_string(predicted_message):
    rounded_message = ''
    for num in predicted_message:
        if(float(num) > 0.5):
            rounded_message += '1'
        else:
            rounded_message += '0'
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

# Log in base 10 using Tensorflow

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

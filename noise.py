import numpy as np
import keras
import tensorflow as tf

from const import YUV
from utils import rgb2yuv, yuv2rgb

@keras.saving.register_keras_serializable(package="HiDDeN")
class Crop(keras.Layer):
    def __init__(self, p, pad_value=1.0):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in the range [0.0, 1.0], got {p}")
        self.p = p
        self.pad_value = pad_value

    def call(self, images):
        self.H = tf.shape(images)[1]
        self.W = tf.shape(images)[2]
        self.minx = np.random.randint(0, (1 - self.p) * self.W)
        self.miny = np.random.randint(0, (1 - self.p) * self.H)

        images = tf.map_fn(self.crop_and_pad, images)
        return images

    def crop_and_pad(self, image):
        crop_height = self.p * self.H
        crop_width = self.p * self.W
        crop = tf.slice(image, [self.miny, self.minx, 0], [crop_height, crop_width, -1])

        pad_top = self.miny
        pad_bottom = self.H - crop_height - pad_top
        pad_left = self.minx
        pad_right = self.W - crop_width - pad_left

        paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

        padded = tf.pad(crop, paddings, mode="CONSTANT", constant_values=self.pad_value)
        return padded


class JpegCompression(keras.Layer):
    def __init__(self, q=50):
        super().__init__()
        if not 0 <= q <= 100:
            raise ValueError(f"q must be in the range [0, 100], got {q}")
        self.q = q

    def call(self, images):
        if YUV:
            images = yuv2rgb(images)

        rgb = tf.cast(images * 255, tf.uint8)
        rgb = tf.map_fn(self.jpeg_compress, rgb)
        images = tf.cast(images, tf.float32) / 255.0

        if YUV:
            images = rgb2yuv(images)
        return images

    def jpeg_compress(self, image):
        jpeg = tf.image.encode_jpeg(image, format="rgb", quality=self.q)
        decoded = tf.image.decode_jpeg(jpeg)
        return decoded

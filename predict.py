import keras
import numpy as np
import skimage.io as io
from skimage.color import rgb2yuv, yuv2rgb
from skimage.transform import resize
import matplotlib.pyplot as plt

from const import *
from hidden import KConvertToTensor, KExpandDims, KTile, KConcat, HIDDEN
from data_loader import load_data
from utils import count_errors, generate_random_messages, round_message_to_string


if __name__ == "__main__":
    train_generator, test_generator, input_shape = load_data()
    N, H, W, C = NUM_IMAGES, *input_shape
    test_messages = generate_random_messages(SIZE_TEST)
    L = test_messages.shape[1]
    network = HIDDEN("HiDDeN_COCO2017.keras", H, W, C, L)
    decoded_img, decoded_msg = network.predict(test_generator, test_messages)
    errors = []
    i = 0
    for msg in decoded_msg:
        rpm = round_message_to_string(msg)
        tpm = round_message_to_string(test_messages[i])
        err = count_errors(tpm, rpm)
        errors.append(err)
        i += 1
    print(f"{sum(errors) / SIZE_TEST}/{MESSAGE_LENGTH}")

    test_messages = generate_random_messages(1)
    image = np.float32(io.imread("dataset/test/000000002934.jpg")) / 255
    image = resize(image, (H, W, C))
    if YUV:
        image = rgb2yuv(image)

    image = np.reshape(image, (1, H, W, C))
    decoded_img, decoded_msg, discriminator_output = network.predict([image, test_messages])
    image = np.reshape(image, (H, W, C))
    decoded_img = np.reshape(decoded_img, (H, W, C))
    if YUV:
        image = yuv2rgb(image)
        decoded_img = yuv2rgb(decoded_img)

    decoded_img = (decoded_img - decoded_img.min()) / (decoded_img.max() - decoded_img.min())
    plt.figure()
    plt.subplot(1, 2, 1); plt.imshow(image); plt.title("Input Image"); plt.colorbar()
    plt.subplot(1, 2, 2); plt.imshow(decoded_img); plt.title("Decoded Image"); plt.colorbar()
    plt.show()
    diff_image = (decoded_img - image)
    diff_image = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())
    plt.figure()
    plt.imshow(diff_image, clim=None); plt.title("Difference Image (FSHS)"); plt.colorbar()
    plt.show()

    colors = ("red", "green", "blue")

    image = np.uint8(image * 256)
    fig, ax = plt.subplots()
    ax.set_xlim([0, 256])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id], bins=256, range=(0, 256)
        )
        ax.plot(bin_edges[0: -1], histogram, color=color)

    ax.set_title("Image histogram")
    plt.show()

    decoded_img = np.uint8(decoded_img * 256)
    fig, ax = plt.subplots()
    ax.set_xlim([0, 256])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            decoded_img[:, :, channel_id], bins=256, range=(0, 256)
        )
        ax.plot(bin_edges[0: -1], histogram, color=color)

    ax.set_title("Decoded image histogram")
    plt.show()

    print(f"Original message: {round_message_to_string(test_messages[0])}")
    print(f"Decoded message: {round_message_to_string(decoded_msg[0])}")
    print(f"Discriminator output: {discriminator_output}")

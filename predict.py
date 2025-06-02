import utils
from aae import *
from data_loader import load_data
import skimage.io as io
from skimage.color import rgb2yuv, yuv2rgb
from skimage.transform import resize
import matplotlib.pyplot as plt


def predict(network, prediction_images, prediction_messages):
    print("Starting Prediction")
    decoded_img = []
    original_msg = []
    decoded_msg = []
    x = prediction_images
    for i, batch in enumerate(x):
        print(f"Batch {i + 1}/{len(x)}")
        batch_size = len(batch)
        index = np.random.randint(0, len(x), batch_size)
        pred_messages = prediction_messages[index]
        (imgs, msgs, _) = network.predict_on_batch([batch, pred_messages])
        decoded_img.extend(imgs)
        original_msg.extend(pred_messages)
        decoded_msg.extend(msgs)

    for i in range(len(decoded_msg)):
        decoded_msg[i] = round_message_to_string(decoded_msg[i])
        original_msg[i] = round_message_to_string(original_msg[i])
    print(f"Accuracy: {np.sum(np.array(original_msg) == np.array(decoded_msg))}/{len(decoded_msg)}")
    return decoded_img, decoded_msg


if __name__ == "__main__":
    network = keras.models.load_model('HiDDeN_COCO2017.keras')
    network.summary()
    (train_generator, test_generator, input_shape) = load_data()
    (N, H, W, C) = (NUM_IMAGES, input_shape[0], input_shape[1], input_shape[2])
    test_messages = generate_random_messages(SIZE_TEST)
    decoded_img, decoded_msg = predict(network, test_generator, test_messages)
    errors = []
    i = 0
    for msg in decoded_msg:
        rpm = round_message_to_string(msg)
        tpm = round_message_to_string(test_messages[i])
        err = count_errors(tpm, rpm)
        errors.append(err)
        i += 1
    print(f'{sum(errors) / SIZE_TEST}/{MESSAGE_LENGTH}')

    H, W, C = 128, 128, 3
    test_messages = generate_random_messages(1)
    image = np.float32(io.imread("dataset/test/000000000809.jpg"))/255
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
    plt.figure()
    plt.subplot(1, 2, 1); plt.imshow(image); plt.title('Input Image'); plt.colorbar()
    plt.subplot(1, 2, 2); plt.imshow(decoded_img); plt.title('Decoded Image'); plt.colorbar()
    plt.show()
    diff_image = (decoded_img - image) * 50
    diff_image = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())
    plt.figure()
    plt.imshow(diff_image, clim=None); plt.title('Difference Image (FSHS)'); plt.colorbar()
    plt.show()

    colors = ("red", "green", "blue")

    image = np.uint8(image * 256)
    fig, ax = plt.subplots()
    ax.set_xlim([0, 256])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id], bins=256, range=(0, 256)
        )
        ax.plot(bin_edges[0:-1], histogram, color=color)

    ax.set_title("Image histogram")
    plt.show()

    decoded_img = np.uint8(decoded_img * 256)
    fig, ax = plt.subplots()
    ax.set_xlim([0, 256])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            decoded_img[:, :, channel_id], bins=256, range=(0, 256)
        )
        ax.plot(bin_edges[0:-1], histogram, color=color)

    ax.set_title("Decoded image histogram")
    plt.show()

    print(f"Original message: {utils.round_message_to_string(test_messages[0])}")
    print(f"Decoded message: {utils.round_message_to_string(decoded_msg[0])}")
    print(f"Discriminator output: {discriminator_output}")
import keras
import matplotlib.pyplot as plt
from const import *
from utils import *
from aae import *
from data_loader import load_data

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
    network = keras.models.load_model('HiDDeN_COCO2017_10k.keras')
    network.summary()
    (train_generator, test_generator, input_shape) = load_data()
    (N, H, W, C) = (NUM_IMAGES,
                    input_shape[0], input_shape[1], input_shape[2])
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

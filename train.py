from utils import *
from aae import HIDDEN
from data_loader import load_data


if __name__ == "__main__":
    # Take some infos from the dataset
    (train_generator, test_generator, input_shape) = load_data()
    (N, H, W, C) = (NUM_IMAGES,
                    input_shape[0], input_shape[1], input_shape[2])
    # Generate random messages as input of the encoder
    messages = generate_random_messages(N)
    L = messages.shape[1]
    epochs = EPOCHS
    print(f'{N} images, {H} x {W} x {C}')
    print(f"Message length: {L}")
    # Create the network
    network = HIDDEN(H, W, C, L, "adam")

    # Train the network
    network.train(epochs, train_generator, messages)
    network.save("HiDDeN_COCO2017_10k.keras")
    
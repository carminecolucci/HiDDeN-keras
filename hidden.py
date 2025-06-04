import numpy as np
import keras
from keras.layers import Activation, Dense, BatchNormalization, \
    Conv2D, Dropout, Identity, Input, GaussianNoise, GlobalAveragePooling2D
from keras.models import Model
import tensorflow as tf

from const import KERNEL_SIZE
from utils import round_message_to_string


@keras.saving.register_keras_serializable(package="HiDDeN")
class KExpandDims(keras.Layer):
    def call(self, x, axis):
        return tf.expand_dims(x, axis=axis)

@keras.saving.register_keras_serializable(package="HiDDeN")
class KConvertToTensor(keras.Layer):
    def call(self, x, dtype):
        return tf.convert_to_tensor(x, dtype=dtype)

@keras.saving.register_keras_serializable(package="HiDDeN")
class KTile(keras.Layer):
    def call(self, x, multiples):
        return tf.tile(x, multiples=multiples)

@keras.saving.register_keras_serializable(package="HiDDeN")
class KConcat(keras.Layer):
    def call(self, x, axis):
        return tf.concat(x, axis=axis)

class HIDDEN():
    # This is the class of the entire network
    def __init__(self, height, width, channel, message_length, optimizer):
        self.message_length = message_length  # L on the paper
        self.H = height  # H on the paper
        self.W = width  # W on the paper
        self.C = channel  # C on the paper
        self.image_shape = (self.H, self.W, self.C)
        print("Build models...")
        self._build_encoder_model()
        self._build_noise_layer_model("identity")
        self._build_decoder_model()
        self._build_discriminator_model()
        self._build_and_compile_network(optimizer)

    def __init__(self, model, height, width, channel, message_length):
        self.message_length = message_length
        self.H = height
        self.W = width
        self.C = channel
        self.image_shape = (self.H, self.W, self.C)
        self.network = keras.models.load_model(model)
        self.encoder_model = self.network.layers[2]
        self.noise_layer_model = self.network.layers[3]
        self.decoder_model = self.network.layers[4]
        self.discriminator_model = self.network.layers[5]

    def _build_encoder_model(self):
        # Build the encoder
        print("Building Encoder...")
        input_images = Input(shape=self.image_shape, name="encoder_input")
        input_messages = Input(shape=(self.message_length,), name="input_messages")
        # Phase 1
        # The encoder convolves with a filter of dimension 3, and then three times with dimension 64
        x = input_images
        # Applying 4 Conv-BN-ReLU blocks with 64 output filters
        for filters in [3, 64, 64, 64]:
            x = Conv2D(filters=filters, kernel_size=KERNEL_SIZE, strides=1, padding="same", use_bias=False)(x)
            x = BatchNormalization(-1)(x)
            x = Activation("relu")(x)

        # Phase 2
        expanded_message = KExpandDims()(input_messages, axis=1)
        expanded_message = KExpandDims()(expanded_message, axis=1)
        a = tf.constant([1, self.H, self.W, 1], tf.int32)
        expanded_message = KConvertToTensor()(expanded_message, dtype=tf.float32)
        # Replicating the message H*W times
        expanded_message = KTile()(expanded_message, multiples=a)
        # Concatenate messages and images channel-wise
        x2 = KConcat()([expanded_message, x, input_images], axis=-1)

        # Phase 3
        # Latest Conv-BN-ReLU block with 64+3+30 output filters
        encoded_images = Conv2D(64, kernel_size=KERNEL_SIZE, strides=1, padding="same", use_bias=False)(x2)
        encoded_images = BatchNormalization(-1)(encoded_images)
        encoded_images = Activation("relu")(encoded_images)

        # Final Convolutonial Layer with 1 x 1 kernel and C output filters
        encoded_images = Conv2D(self.C, 1, padding="same",
                                strides=1)(encoded_images)

        self.encoder_model = Model(
            [input_images, input_messages], encoded_images, name="encoder")

    # TODO: noise layer wants as input also the cover image in case of "Crop" noise.
    def _build_noise_layer_model(self, name):
        # Function that applies the noise layer to the image
        print("Building Noise Layer...")
        input_images = Input(shape=self.image_shape, name="noise_input")
        if name == "identity":
            x = Identity()(input_images)
            self.noise_layer_model = Model(inputs=input_images, outputs=x, name="noise")
        elif name == "gaussian":
            x = GaussianNoise(2)(input_images)
            self.noise_layer_model = Model(inputs=input_images, outputs=x, name="noise")
        elif name == "dropout":
            x = Dropout(0.3)(input_images)
            self.noise_layer_model = Model(inputs=input_images, outputs=x, name="noise")

    def _build_decoder_model(self):
        # Build the decoder
        print("Building Decoder Generator...")
        input_images = Input(shape=self.image_shape, name="decoder_input")
        x = input_images
        # Applying 7 Conv-BN-ReLU blocks with 64 output filters
        for filters in [64, 64, 64, 64, 64, 64, 64]:
            x = Conv2D(filters, kernel_size=KERNEL_SIZE, strides=1, padding="same", use_bias=False)(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)

        # Last ConvBNReLU with L filters
        x = Conv2D(self.message_length, kernel_size=KERNEL_SIZE, padding="same", use_bias=False)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        # Average Pooling over all spatial dimensions
        x = GlobalAveragePooling2D()(x)
        # Final linear layer with L units
        x = Dense(self.message_length)(x)
        self.decoder_model = Model(input_images, x, name="decoder")

    def _build_discriminator_model(self):
        # build the adversary
        input_images = Input(shape=self.image_shape, name="adversary_input")
        x = input_images
        # Applying 3 Conv-BN-ReLU blocks with 64 output filters
        for filters in [64, 64, 64]:
            x = Conv2D(filters, kernel_size=KERNEL_SIZE, strides=1, padding="same")(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
        # Average Pooling over all spatial dimensions
        x = GlobalAveragePooling2D()(x)
        # Final linear layer to classify the image
        adversary_output = Dense(1, activation="sigmoid")(x)
        self.discriminator_model = Model(input_images, adversary_output, name="discriminator")

    def _build_and_compile_network(self, optimizer):
        self.discriminator_model.compile(
            loss=keras.losses.BinaryCrossentropy(), optimizer="adam")
        print("Connecting models...")

        images = Input(shape=self.image_shape, name="input")
        messages = Input(shape=(self.message_length,), name="messages")
        encoder_output = self.encoder_model([images, messages])
        noise_output = self.noise_layer_model(encoder_output)
        decoder_output = self.decoder_model(noise_output)
        discriminator_output = self.discriminator_model(encoder_output)
        # The final network: Encoder + Noise + Decoder + Adversary
        self.network = Model([images, messages], [encoder_output, decoder_output, discriminator_output], name="HiDDeN_ESM_2025_group_5")

        # Compile all the network
        # Encoder output -> MSE Loss * 0.7 (L_I)
        # decoder_output -> MSE Loss * 1 (L_M)
        # discriminator_output -> Adv_loss * 0.001 (L_G)
        # The last loss (L_A) is defined in the discriminator model 
        self.network.compile(loss=["mse", "mse", keras.losses.BinaryCrossentropy()],
                             # The relative weights of the losses, lambda_i and lambda_g
                             loss_weights=[0.7, 1, 0.001],
                             optimizer=optimizer)
        self.network.summary()

    # Train on batch the entire network
    def train(self, epochs, train_images, train_messages):
        for epoch in range(epochs):
            for i, batch in enumerate(train_images):
                print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_images)}")

                batch_size = len(batch)
                index = np.random.randint(0, len(train_images), batch_size)
                real = np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))
                batch_messages = train_messages[index]
                cover_images = batch
                encoded_images = self.encoder_model.predict([batch, batch_messages])
                # Train the adversary
                loss_real = self.discriminator_model.train_on_batch(cover_images, real)
                loss_fake = self.discriminator_model.train_on_batch(encoded_images, fake)
                #  Train all the network
                autoencoder_loss = self.network.train_on_batch([batch, batch_messages], [batch, batch_messages, real])
                print(f"Autoencoder loss: {autoencoder_loss[0]}\tImage loss: {autoencoder_loss[1]}\t"
                      f"Message loss: {autoencoder_loss[2]}\tAdversary loss: {autoencoder_loss[3]}")

    def predict(self, prediction_images, prediction_messages):
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
            imgs, msgs, _ = self.network.predict_on_batch([batch, pred_messages])
            decoded_img.extend(imgs)
            original_msg.extend(pred_messages)
            decoded_msg.extend(msgs)

        for i in range(len(decoded_msg)):
            decoded_msg[i] = round_message_to_string(decoded_msg[i])
            original_msg[i] = round_message_to_string(original_msg[i])
        print(f"Accuracy: {np.sum(np.array(original_msg) == np.array(decoded_msg))}/{len(decoded_msg)}")
        return decoded_img, decoded_msg

    def save(self, path):
        self.network.save(path)

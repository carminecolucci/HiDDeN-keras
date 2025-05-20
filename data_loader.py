from keras.utils import image_dataset_from_directory
from keras.layers import Rescaling
from const import *

def load_data():
    # Get and process the COCO dataset
    datadir = 'dataset'
    trainingset = datadir + '/train/'
    testset = datadir + '/tester/'

    normalization_layer = Rescaling(1.0 / 255)

    # Normalizing the train images to the range of [0., 1.]
    train_generator = image_dataset_from_directory(
        directory=trainingset,
        color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        labels="inferred",
        class_names="input",
        shuffle=True
    )
    train_generator = train_generator.map(lambda image, label: (normalization_layer(image), label))

    # Normalizing the test images to the range of [0., 1.]
    test_generator = image_dataset_from_directory(
        directory=testset,
        color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        labels="inferred",
        class_names="input",
        shuffle=True
    )
    test_generator = test_generator.map(lambda image, label: (normalization_layer(image), label))

    num_samples = train_generator.n
    input_shape = train_generator.image_shape

    print(f"Image input {input_shape}")
    print(f'Loaded {num_samples} training samples')
    print(f'Loaded {test_generator.n} test samples')
    return (train_generator, test_generator, input_shape)

from keras.utils import image_dataset_from_directory
from keras.layers import Rescaling
from const import *

def load_data():
    # Get and process the COCO dataset
    datadir = 'dataset'
    trainingset = datadir + '/train2017/'
    testset = datadir + '/test2017/'

    normalization_layer = Rescaling(1.0 / 255)

    # Normalizing the train images to the range of [0., 1.]
    train_generator = image_dataset_from_directory(
        directory=trainingset,
        color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        labels=None,
        shuffle=True
    )
    train_generator = train_generator.map(lambda image: normalization_layer(image))

    # Normalizing the test images to the range of [0., 1.]
    test_generator = image_dataset_from_directory(
        directory=testset,
        color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        labels=None,
        shuffle=True
    )
    test_generator = test_generator.map(lambda image: normalization_layer(image))
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    print(f"Image input {input_shape}, batch size {BATCH_SIZE}.")
    print(f'Training set has {len(train_generator)} batches.')
    print(f'Test set has {len(test_generator)} batches.')
    return (train_generator, test_generator, input_shape)

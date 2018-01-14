import datetime
import sys
import pickle

import matplotlib.pyplot as plt
from keras import metrics
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

from groundwater_timenet.learn.settings import *
from groundwater_timenet.learn.generator import ConvolutionalAtrousGenerator
from groundwater_timenet.utils import setup_logging, LEARN_LOG


logger = setup_logging(__name__, LEARN_LOG, "INFO")


def create_model(
        input_size=INPUT_SIZE, temporal_size=TEMPORAL_SIZE,
        meta_size=META_SIZE, layer_dilation=CONVOLUTIONAL_LAYER_DILATION,
        defaults=CONVOLUTIONAL_LAYER_DEFAULTS):
    # TODO: perhaps a merge layer will also be a good idea instead of repeating
    # the metadata.
    model = Sequential()
    # first layer collects all metadata
    layer = Conv1D(
        filters=input_size,
        kernel_size=temporal_size + meta_size,
        input_shape=(input_size, temporal_size + meta_size),
        dilation_rate=layer_dilation[0],
        **defaults
    )
    model.add(layer)
    for dilation_rate in layer_dilation[1:]:
        layer = Conv1D(
            filters=input_size,
            kernel_size=temporal_size + meta_size,
            dilation_rate=dilation_rate,
            **defaults
        )
        model.add(layer)
    # last layer outputs only the input shape
    layer = MaxPooling1D(
        pool_size=input_size,
    )
    model.add(layer)
    return model


def plot_history(history):
    """taken from:  https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/"""
    logger.debug(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def main(directory=None, epochs=EPOCHS):
    start = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    try:
        os.makedirs(CONVOLUTIONAL_MODEL_FILEPATH)
    except FileExistsError:
        pass
    try:
        os.makedirs(os.path.join(TENSORBOARD_FILEPATH, start))
    except FileExistsError:
        pass

    model = create_model()
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto')
    tensor_board = TensorBoard(
        log_dir=os.path.join('.', 'var', 'log', 'tensorboard'))

    model.compile(
        loss='mean_squared_error',
        optimizer='rmsprop',
        metrics=['accuracy', metrics.mae]
    )
    train_generator = ConvolutionalAtrousGenerator(directory=os.path.join(directory, 'train'))
    validation_generator = ConvolutionalAtrousGenerator(directory=os.path.join(directory, 'validation'))

    # If you have installed TensorFlow with pip, you should be able to
    # launch TensorBoard from the command line:
    # tensorboard --logdir=/full_path_to_your_logs

    history = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, tensor_board, ReduceLROnPlateau()]
    )

    plot_history(history)
    with open(os.path.join(TENSORBOARD_FILEPATH, start, 'history.csv'), 'w') as p:
        for metric in history.params['metrics']:
            p.write(','.join([metric] + [str(h) for h in history.history[metric]]) + '\n')

    end = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    model.save(CONVOLUTIONAL_MODEL_FILEPATH.format(
        datetime_start=start, datetime_end=end))


if __name__ == "__main__":
    try:
        directory = sys.argv[0]
        main(directory)
    except IndexError:
        main()
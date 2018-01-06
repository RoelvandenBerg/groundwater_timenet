import datetime

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, TensorBoard

from groundwater_timenet.learn.settings import *
from groundwater_timenet.learn.generator import ConvolutionalAtrousGenerator


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


def main():
    start = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    model = create_model()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto')
    tensor_board = TensorBoard(
        log_dir='./var/log/tensorboard',
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None)

    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    generator = ConvolutionalAtrousGenerator()

    # If you have installed TensorFlow with pip, you should be able to
    # launch TensorBoard from the command line:
    # tensorboard --logdir=/full_path_to_your_logs

    model.fit_generator(
        generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, tensor_board]
    )
    end = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    model.save(CONVOLUTIONAL_MODEL_FILEPATH.format(
        datetime_start=start, datetime_end=end))

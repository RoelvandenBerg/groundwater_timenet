from keras.models import Sequential
from keras.layers import Input, Conv1D, Dense, Add, Multiply
from keras.callbacks import EarlyStopping, TensorBoard
from groundwater_timenet.learn.constants import *

# If you have installed TensorFlow with pip, you should be able to launch TensorBoard from the command line:
# tensorboard --logdir=/full_path_to_your_logs


def create_model(filters, kernel_size, layer_dilation=CONVOLUTIONAL_LAYER_DILATION,
                 defaults=CONVOLUTIONAL_LAYER_DEFAULTS, input_shape=()):
    # todo: how to choose the filters and kernel_size?
    # Probably: 365 or 366 (a year of data / a year of data + 1)
    # kernel_size should be a day I guess? (1, ..) SEE: http://sergeiturukin.com/2017/03/02/wavenet.html
    # TODO: apart from this model also add a metadata layer and merge (Keras merge layer) these with the same shape on the first (and/or last?) conv layer.
    model = Sequential()
    defaults['input_shape'] = input_shape

    temporal_input = Input(shape=(TEMPORAL_SIZE, INPUT_SIZE))
    metadata_input = Input(shape=(1, META_SIZE))
    temporal_layer = Conv1D(
        filters,
        kernel_size,
        dilation_rate=1,
        **defaults
    )(temporal_input)
    metadata_layer = Dense(META_SIZE, activation='relu')(metadata_input)

    model.
    for dilation_rate in layer_dilation:
        layer = Conv1D(
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            **defaults
        )
        if 'input_shape' in defaults:
            del defaults['input_shape']
        model.add(layer)
    # todo: what about the input and last layer?
    return model


def main():
    filters = None
    kernel_size = None
    input_shape = None
    model = create_model(filters, kernel_size, input_shape=input_shape)

    EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=0,
                                  verbose=0, mode='auto')
    TensorBoard(log_dir='./logs', histogram_freq=0,
                                batch_size=32, write_graph=True,
                                write_grads=False, write_images=False,
                                embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)

    model.compile(loss='rmsprop', optimizer='sgd')
    #TODO: https://keras.io/models/sequential/#fit_generator
    model.fit_generator()


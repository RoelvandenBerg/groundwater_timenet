from keras.models import Sequential
from keras.layers import Conv1D

# # These are the defaults we keep for now but we might want to tweak later on.
# THINK_ABOUT_DEFAULTS = {
#     "kernel_initializer": 'glorot_uniform',  # Perhaps also try lecun_uniform?
#     "bias_initializer": 'zeros',  # think about initializing with a small constant (0.1) when using ReLus to avoid dead neurons.
#     "kernel_regularizer": None,  # probably good idea to use a weight regularizer to prevent exploding or vanishing gradients in case the problem arises.
#     "kernel_constraint": None,  # for now: don't use, use a kernel_regularizer if the exploding / vanishing gradients seems to pop up first. Try this out if that doesn't give the required results.
#     "bias_constraint": None  # see kernel constraint
# }
#
# # These defaults we keep for pretty sure and do not tweak:
# KEEP_DEFAULT = {
#     "use_bias": True,
#     "bias_regularizer": None,
#     "activity_regularizer": None,
# }

DEFAULTS = {
    "padding": 'causal',
    "activation": 'tanh',  # we could also try sigmoid (we chose tanh because wavenet uses it and it has a stronger gradient). And I'm curious about ReLu activation.
}

LAYER_DILATION = (1, 2, 6, 12, 24)  # half-a-month (non-telemetric systems usually have this frequency), monthly, quarterly, halfyearly and yearly.


def create_model(filters, kernel_size, layer_dilation=LAYER_DILATION,
                 defaults=DEFAULTS, input_shape=()):
    # todo: how to choose the filters and kernel_size?
    # Probably: 365 or 366 (a year of data / a year of data + 1)
    # kernel_size should be a day I guess? (1, ..) SEE: http://sergeiturukin.com/2017/03/02/wavenet.html
    # TODO: apart from this model also add a metadata layer and merge (Keras merge layer) these with the same shape on the first (and/or last?) conv layer.
    model = Sequential()
    defaults['input_shape'] = input_shape
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
    model.compile(loss='rmsprop', optimizer='sgd')
    #TODO: https://keras.io/models/sequential/#fit_generator
    model.fit_generator()


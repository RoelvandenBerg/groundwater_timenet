# TODO: these sizes can be determined, for now we assume them constant:

# These are the sizes of the network
INPUT_SIZE = 24
OUTPUT_SIZE = 1
META_SIZE = 212
TEMPORAL_SIZE = 7

# Convolutional layer settings.
CONVOLUTIONAL_LAYER_DILATION = (1, 2, 6, 12, 24)  # half-a-month (non-telemetric systems usually have this frequency), monthly, quarterly, halfyearly and yearly.
CONVOLUTIONAL_LAYER_DEFAULTS = {
    "padding": 'causal',
    "activation": 'tanh',  # we could also try sigmoid (we chose tanh because wavenet uses it and it has a stronger gradient). And I'm curious about ReLu activation.
}

# # These are the defaults we keep for now but we might want to tweak later on.
# CONVOLUTIONAL_LAYER_THINK_ABOUT_DEFAULTS = {
#     "kernel_initializer": 'glorot_uniform',  # Perhaps also try lecun_uniform?
#     "bias_initializer": 'zeros',  # think about initializing with a small constant (0.1) when using ReLus to avoid dead neurons.
#     "kernel_regularizer": None,  # probably good idea to use a weight regularizer to prevent exploding or vanishing gradients in case the problem arises.
#     "kernel_constraint": None,  # for now: don't use, use a kernel_regularizer if the exploding / vanishing gradients seems to pop up first. Try this out if that doesn't give the required results.
#     "bias_constraint": None  # see kernel constraint
# }
#
# # These defaults we keep for pretty sure and do not tweak:
# CONVOLUTIONAL_LAYER_KEEP_DEFAULT = {
#     "use_bias": True,
#     "bias_regularizer": None,
#     "activity_regularizer": None,
# }



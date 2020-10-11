# encoding: UTF-8
import os

SEED = 0

LIST_NN_PARAMS = [
    # {"filters": 8, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": False},
    # {"filters": 8, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": False},
    {"filters": 8, "kernel_size": (5, 5), "activation": "tanh", "strides": (1, 1), "padding": "same", "pooling": True},
    # {"filters": 16, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": False},
    # {"filters": 16, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": False},
    {"filters": 16, "kernel_size": (5, 5), "activation": "tanh", "strides": (1, 1), "padding": "same", "pooling": True},
    {"filters": 32, "kernel_size": (3, 3), "activation": "tanh", "strides": (1, 1), "padding": "same", "pooling": False}
]

PARAMS_AE_MODEL = {
    i: params for i, params in enumerate(LIST_NN_PARAMS)
}

DIR_DST = os.path.join("..", "dist")

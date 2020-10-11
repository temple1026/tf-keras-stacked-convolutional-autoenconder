# encoding: UTF-8

from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization,  # UpSampling2D,
                                     Conv2D, Conv2DTranspose, Input,
                                     MaxPooling2D, Reshape)


def getConvBlock(layer_input, upconv=False, name="", **kwargs):
    """
    Return a convolutional block.
    """
    params_layer = kwargs["params_layer"].copy()

    check_pooling = params_layer.pop("pooling")

    if upconv:
        # layers_hidden = UpSampling2D(size=(2, 2), name=f"{name}_upsampling")(layer_input) if check_pooling else layer_input
        # layer_output = Conv2D(name=f"{name}_conv", **params_layer)(layers_hidden)

        # layers_hidden = Conv2D(name=f"{name}_conv", **params_layer)(layer_input)
        # layer_output = UpSampling2D(size=(2, 2), name=f"{name}_upsampling")(layers_hidden) if check_pooling else layers_hidden
        params_layer["strides"] = (2, 2) if check_pooling else (1, 1)
        layer_output = Conv2DTranspose(name=f"{name}_conv", **params_layer)(layer_input)

    else:
        layers_hidden = Conv2D(name=f"{name}_conv", **params_layer)(layer_input)
        layer_output = MaxPooling2D((2, 2), name=f"{name}_pool", padding="same")(layers_hidden) if check_pooling else layers_hidden

    return layer_output


def loadStackedModel(width, height, depth, params_model) -> Model:
    """
    Return the convolutional autoencoder model depending on the depth.
    """

    # Add common layers.
    layer_input = Input(shape=(width, height), name="input0")
    layers_hidden = Reshape(target_shape=(width, height, 1), name="reshape0")(layer_input)
    layers_hidden = BatchNormalization(name="bn0")(layers_hidden)

    # Add convolutional and pooling layers.
    for d in range(depth):
        layers_hidden = getConvBlock(layer_input=layers_hidden, name=f"enc{d}", params_layer=params_model[d])

    # Add deconvolutional layers
    for d in reversed(range(depth - 1)):
        layers_hidden = getConvBlock(layer_input=layers_hidden, upconv=True, name=f"dec{d}", params_layer=params_model[d])

    # if mismatch the output shape of hidden_layers and input, choose stride (2, 2).
    strides = (1, 1) if layers_hidden.shape[1:3] == (width, height) else (2, 2)
    layer_output = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=strides, padding="same", activation="sigmoid", name="output")(layers_hidden)

    return Model(inputs=layer_input, outputs=layer_output)

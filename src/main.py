# encoding: UTF-8
import os
import random
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose,
                                     Input, MaxPooling2D, Reshape, UpSampling2D,
                                     BatchNormalization)
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

SEED = 0
LIST_NN_PARAMS = [
    # {"filters": 8, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": False},
    # {"filters": 8, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": False},
    {"filters": 8, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": True},
    # {"filters": 16, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": False},
    # {"filters": 16, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": False},
    {"filters": 16, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": True},
    {"filters": 1, "kernel_size": (3, 3), "activation": "relu", "strides": (1, 1), "padding": "same", "pooling": False}
]

PARAMS_AE_MODEL = {
    i: params for i, params in enumerate(LIST_NN_PARAMS)
}


def setSeed(seed, mode_ops):
    os.environ['PYTHONHASHSEED'] = '0'

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if mode_ops:

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    else:
        os.environ['TF_DETERMINISTIC_OPS'] = '0'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '0'

        max_workers = 4
        tf.config.threading.set_inter_op_parallelism_threads(max_workers)
        tf.config.threading.set_intra_op_parallelism_threads(max_workers)


def getConvBlock(layer_input, upconv=False, name="", **kwargs):
    params_layer = kwargs["params_layer"].copy()

    check_pooling = params_layer.pop("pooling")

    if upconv:
        layers_hidden = Conv2D(name=f"{name}_conv", **params_layer)(layer_input)
        layer_output = UpSampling2D(size=(2, 2))(layers_hidden) if check_pooling else layers_hidden

    else:
        layers_hidden = Conv2D(name=f"{name}_conv", **params_layer)(layer_input)
        layer_output = MaxPooling2D((2, 2), name=f"{name}_pool", padding="same")(layers_hidden) if check_pooling else layers_hidden

    return layer_output


def loadStackedModel(width, height, depth, params_model=PARAMS_AE_MODEL) -> Model:
    """

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

    # if the last layer is pooled, choose stride (2, 2).
    strides = (2, 2) if params_model[depth - 1]["pooling"] else (1, 1)

    # Add a layer for reconstruction
    layer_output = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=strides, padding="same", activation="sigmoid", name="output")(layers_hidden)

    return Model(inputs=layer_input, outputs=layer_output)


def calPower(data):
    power = np.sum(data**2, keepdims=True)
    return power


def addNoise(imgs, noises, snr):
    p_imgs = []
    p_noises = []
    for img, noise in zip(imgs, noises):
        p_imgs.append(calPower(img))
        p_noises.append(calPower(noise))

    a = np.sqrt(np.array(p_imgs)/(np.array(p_noises) * np.power(10.0, snr/10.0)))

    imgs_out = imgs + a*noises

    return imgs_out


def main():
    setSeed(0, 1)

    # VRAMを無駄に確保しないように設定
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=20, type=int)
    parser.add_argument("-b", "--batch_size", default=100, type=int)
    parser.add_argument("-s", "--stacked", default=1, type=int)
    parser.add_argument("-n", "--snr", default=-40, type=int)

    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    stacked = args.stacked
    snr = args.snr

    mnist = tf.keras.datasets.mnist
    (trainX, trainY), (testX, testY) = mnist.load_data()

    width, height = trainX.shape[1:]

    trainX_corrupted = addNoise(trainX, np.random.random(size=(trainX.shape)), snr=snr)
    testX_corrupted = addNoise(testX, np.random.random(size=(testX.shape)), snr=snr)

    trainX_corrupted = trainX_corrupted / 255.0
    testX_corrupted = testX_corrupted / 255.0

    trainX = trainX / 255.0

    depth = len(PARAMS_AE_MODEL.keys())

    model_trained = None

    dist_path = os.path.join("dist", "models")
    os.makedirs(dist_path, exist_ok=True)

    if not stacked:
        # Training for non-stacked convolutional denoising autoencoder
        model = loadStackedModel(width, height, depth)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse")
        model.summary()
        dist_save = os.path.join(dist_path, "best_no_stacked.hdf5")
        mc = ModelCheckpoint(dist_save, save_best_only=True, mode='min')
        model.fit(x=trainX_corrupted, y=trainX, batch_size=batch_size, shuffle=True, validation_split=0.1, epochs=epochs, callbacks=[mc])

    else:
        # Train depth + 1 times. At depth time is training for all layers.
        for d in range(depth + 1):
            model = loadStackedModel(width, height, d+1 if d+1 <= depth else depth)
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse")

            if model_trained is not None:
                names_trained_layer = {layer.name: i for i, layer in enumerate(model_trained.layers)}

                for idx, layer in enumerate(model.layers):
                    # print(idx, layer)
                    if layer.name in names_trained_layer:
                        try:
                            idx_trained_layer = names_trained_layer[layer.name]
                            weights_trained = model_trained.layers[idx_trained_layer].get_weights()

                            model.layers[idx].set_weights(weights_trained)

                            model.layers[idx].trainable = False if d < depth else True
                            print(f"Trained weights were set into {layer.name}.")
                        except ValueError:
                            print(f"Mismatch layer shapes: {layer.name}")

                    else:
                        print(f"{layer.name} is trainable.")

            if d == depth:
                """
                他の学習をする場合にはここで層の入れ替えをする
                """

            model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse")
            model.summary()

            dist_save = os.path.join(dist_path, f"best_stacked_{d}.hdf5")
            mc = ModelCheckpoint(dist_save, save_best_only=True, mode='min')
            model.fit(x=trainX_corrupted, y=trainX, batch_size=batch_size, shuffle=True, validation_split=0.1, epochs=epochs, callbacks=[mc])

            model_trained = model

            del model

    # evaluate
    model = tf.keras.models.load_model(dist_save)
    results = model.predict(x=testX_corrupted)

    idx_target = 0
    dir_img = "img"
    os.makedirs(dir_img, exist_ok=True)

    plt.cla()
    plt.imshow(np.reshape(results[idx_target], newshape=(width, height)), cmap="gray")
    plt.savefig(os.path.join(dir_img, f"pred_{idx_target}.png"))

    plt.cla()
    plt.imshow(np.reshape(testX_corrupted[idx_target] * 255.0, newshape=(width, height)), cmap="gray")
    plt.savefig(os.path.join(dir_img, f"testX_corrupted_{idx_target}.png"))

    plt.cla()
    plt.imshow(np.reshape(testX[idx_target], newshape=(width, height)), cmap="gray")
    plt.savefig(os.path.join(dir_img, f"testX_{idx_target}.png"))

    plt.cla()
    plt.imshow(trainX_corrupted[idx_target] * 255.0, cmap="gray")
    plt.savefig(os.path.join(dir_img, f"trainX_corrupted_{idx_target}.png"))

    plt.cla()
    plt.imshow(trainX[idx_target] * 255.0, cmap="gray")
    plt.savefig(os.path.join(dir_img, f"trainX_{idx_target}.png"))


if __name__ == "__main__":
    main()

# encoding: UTF-8
import os
import random
import argparse

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from libs.models import loadStackedModel
from libs.utilities import addNoise
from config import SEED, PARAMS_AE_MODEL, DIR_DST


def setSeed(seed, keep_repro=True):
    os.environ['PYTHONHASHSEED'] = '0'

    # Set seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Settings for the reproducibility in Tensorflow 2.3.
    if keep_repro:

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


def replaceLayers(model):
    """
    Not implemented yet.
    """
    return model


def main():
    # Set seeds
    setSeed(SEED)

    # Set the limitation of VRAM
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Define arguments
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

    # Load MNIST
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

    # Get the size of image
    width, height = trainX.shape[1:]

    # Add the random noise into inputs and ground truth.
    trainX_corrupted = addNoise(trainX, np.random.random(size=(trainX.shape)), snr=snr)
    testX_corrupted = addNoise(testX, np.random.random(size=(testX.shape)), snr=snr)

    # 0-1 Normalize
    trainX_corrupted = trainX_corrupted / 255.0
    testX_corrupted = testX_corrupted / 255.0
    trainX = trainX / 255.0
    testX = testX / 255.0

    # Define the number of layers
    depth = len(PARAMS_AE_MODEL.keys())

    # define the path to dst
    dst_path_base = os.path.join(DIR_DST, "stacked" if stacked else "no-stacked") + f"_snr{snr}"
    dst_path_model = os.path.join(dst_path_base, "models")
    os.makedirs(dst_path_model, exist_ok=True)

    if not stacked:
        # Training for non-stacked convolutional denoising autoencoder

        # Load model
        model = loadStackedModel(width, height, depth, PARAMS_AE_MODEL)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse")
        model.summary()
        dst_save = os.path.join(dst_path_model, "best_no_stacked.hdf5")
        mc = ModelCheckpoint(dst_save, save_best_only=True, mode='min')
        model.fit(x=trainX_corrupted, y=trainX, batch_size=batch_size, shuffle=True, validation_split=0.1, epochs=epochs, callbacks=[mc])

    else:
        # Train depth + 1 times. At depth time is training for all layers.

        model_trained = None

        for d in range(depth + 1):
            model = loadStackedModel(width, height, d+1 if d+1 <= depth else depth, PARAMS_AE_MODEL)
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse")

            if model_trained is not None:
                names_trained_layer = {layer.name: i for i, layer in enumerate(model_trained.layers)}

                for idx, layer in enumerate(model.layers):
                    # print(idx, layer)
                    if layer.name in names_trained_layer:
                        # if the layer.name of model exist in trained model
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
                Replace layers in advance if you run as another task. e.g) classification
                """
                # replaceLayers(model)
                y = trainX
            else:
                y = trainX

            model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse")
            model.summary()

            # Define the path to model
            dst_save = os.path.join(dst_path_model, f"best_stacked_{d}.hdf5")

            # Degine callbacks
            mc = ModelCheckpoint(dst_save, save_best_only=True, mode='min')

            # Train
            model.fit(x=trainX_corrupted, y=y, batch_size=batch_size, shuffle=True, validation_split=0.1, epochs=epochs, callbacks=[mc])

            # Save the model as trained model.
            model_trained = model

            del model

    # evaluate
    model = tf.keras.models.load_model(dst_save)
    model.evaluate(x=testX_corrupted, y=testX)
    results = model.predict(x=testX_corrupted)

    idx_target = 0
    dir_img = os.path.join(dst_path_base, "img")
    os.makedirs(dir_img, exist_ok=True)

    plt.cla()
    plt.imshow(np.reshape(results[idx_target] * 255.0, newshape=(width, height)), cmap="gray")
    plt.savefig(os.path.join(dir_img, f"pred_{idx_target}.png"))

    plt.cla()
    plt.imshow(np.reshape(testX_corrupted[idx_target] * 255.0, newshape=(width, height)), cmap="gray")
    plt.savefig(os.path.join(dir_img, f"testX_corrupted_{idx_target}.png"))

    plt.cla()
    plt.imshow(np.reshape(testX[idx_target] * 255.0, newshape=(width, height)), cmap="gray")
    plt.savefig(os.path.join(dir_img, f"testX_{idx_target}.png"))

    plt.cla()
    plt.imshow(trainX_corrupted[idx_target] * 255.0, cmap="gray")
    plt.savefig(os.path.join(dir_img, f"trainX_corrupted_{idx_target}.png"))

    plt.cla()
    plt.imshow(trainX[idx_target] * 255.0, cmap="gray")
    plt.savefig(os.path.join(dir_img, f"trainX_{idx_target}.png"))


if __name__ == "__main__":
    main()

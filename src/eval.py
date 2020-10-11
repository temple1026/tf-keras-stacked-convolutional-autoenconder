# encoding: UTF-8
import os
import random
import argparse

import numpy as np
import tensorflow as tf

# import matplotlib.pyplot as plt

from libs.utilities import addNoise, calSNR
from config import SEED


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
    parser.add_argument("-n", "--snr", default=-40, type=int)
    parser.add_argument(
        "-d", "--dir_model",
        default=os.path.join("..", "dist", "stacked_snr-40", "models", "best_stacked_3.hdf5"),
        type=str)

    args = parser.parse_args()
    snr = args.snr
    dir_model = args.dir_model

    if not os.path.exists(dir_model):
        print(f"{dir_model} doesn't exist.")
        return

    # Load MNIST
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

    # Get the size of image
    width, height = trainX.shape[1:]

    # Add the random noise into inputs and ground truth.
    # trainX_corrupted = addNoise(trainX, np.random.random(size=(trainX.shape)), snr=snr)
    testX_corrupted = addNoise(testX, np.random.random(size=(testX.shape)), snr=snr)

    # 0-1 Normalize
    # trainX_corrupted = trainX_corrupted / 255.0
    testX_corrupted = testX_corrupted / 255.0
    # trainX = trainX / 255.0
    testX = testX / 255.0

    # evaluate
    model = tf.keras.models.load_model(dir_model)

    model.evaluate(x=testX_corrupted, y=testX)
    results = model.predict(x=testX_corrupted)

    testX = np.reshape(testX, newshape=(testX.shape[0], width, height, 1))
    snrs = calSNR(y_true=testX*255.0, y_pred=results*255.0)

    print(f"SNR: {float(np.mean(snrs, axis=0)):.4}")

    # idx_target = 0
    # dir_img = os.path.join(dst_path_base, "img")
    # os.makedirs(dir_img, exist_ok=True)

    # plt.cla()
    # plt.imshow(np.reshape(results[idx_target] * 255.0, newshape=(width, height)), cmap="gray")
    # plt.savefig(os.path.join(dir_img, f"pred_{idx_target}.png"))

    # plt.cla()
    # plt.imshow(np.reshape(testX_corrupted[idx_target] * 255.0, newshape=(width, height)), cmap="gray")
    # plt.savefig(os.path.join(dir_img, f"testX_corrupted_{idx_target}.png"))

    # plt.cla()
    # plt.imshow(np.reshape(testX[idx_target] * 255.0, newshape=(width, height)), cmap="gray")
    # plt.savefig(os.path.join(dir_img, f"testX_{idx_target}.png"))

    # plt.cla()
    # plt.imshow(trainX_corrupted[idx_target] * 255.0, cmap="gray")
    # plt.savefig(os.path.join(dir_img, f"trainX_corrupted_{idx_target}.png"))

    # plt.cla()
    # plt.imshow(trainX[idx_target] * 255.0, cmap="gray")
    # plt.savefig(os.path.join(dir_img, f"trainX_{idx_target}.png"))


if __name__ == "__main__":
    main()

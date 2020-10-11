# encoding: UTF-8
import numpy as np


def calPower(img):
    """
    Calculate the total power of image.
    """
    power = np.sum(img**2, keepdims=True)
    return power


def addNoise(imgs, noises, snr):
    """
    Add the noise into images depending on the snr value.
    """

    p_imgs = np.zeros(shape=(imgs.shape[0], 1, 1))
    p_noises = np.zeros(shape=(noises.shape[0], 1, 1))

    # Calculate the power both images and noises.
    for idx, (img, noise) in enumerate(zip(imgs, noises)):
        p_imgs[idx] = calPower(img)
        p_noises[idx] = calPower(noise)

    # Calculate the ratio of noise
    a = np.sqrt(p_imgs/(p_noises * np.power(10.0, snr/10.0)))

    # Add the noise
    imgs_out = imgs + a*noises

    return imgs_out

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gen_rl_example.py
"""
A short script to generate and output example data for the RL algorithm

Copyright (c) 2016, David Hoffman
"""
import time
import pickle
import pyfftw
import numpy as np
from pyDecon.utils import _ensure_positive
from scipy.ndimage import convolve
from skimage.external import tifffile as tif
from pyDecon.decon import richardson_lucy as rl

if __name__ == "__main__":
    print("Generating the data ...")
    x = np.linspace(-2.5, 2.5, 64, True)
    kernel = np.exp(-x**2)
    kernel = kernel[np.newaxis] * kernel[:, np.newaxis]
    # normalize kernel
    k_norm = kernel / kernel.sum()
    # make signal
    x = np.linspace(-10, 10, 512)
    signal = 5.0 * np.logical_and(x < 3, x > -3)
    signal = signal[np.newaxis] * signal[:, np.newaxis]
    blurred = convolve(signal, k_norm, mode="reflect")
    blurred = _ensure_positive(blurred)
    kernel = _ensure_positive(kernel)
    # save ground truth images
    print("Saving the data ...")
    tif.imsave("ground truth.tif", signal.astype(np.float32))
    tif.imsave("blurred.tif", blurred.astype(np.float32))
    tif.imsave("psf.tif", k_norm.astype(np.float32))
    # add some noise to both
    print("Add noise ...")
    np.random.seed(12345)
    blurred[blurred < 0] = 0
    blurred_noisy = np.random.poisson(blurred) + np.random.randn(*blurred.shape) ** 2
    kernel[kernel < 0] = 0
    psf = np.random.poisson(kernel * 100) + np.random.randn(*kernel.shape) ** 2
    psf /= psf.sum()
    print("Save noisy data ...")
    tif.imsave("image.tif", blurred_noisy.astype(np.float32))
    tif.imsave("noisy psf.tif", psf.astype(np.float32))
    # deconvolve data
    print("Deconvolving the data ...", end="")
    # see if you can load wisdom
    try:
        pyfftw.import_wisdom(pickle.load(open("wisdom.p", "rb")))
    except FileNotFoundError:
        pass
    start = time.time()
    decon = rl(blurred_noisy, psf, 10, 1)
    print("{:.3f} seconds".format(time.time() - start))
    print("Saving deconvolved data ...")
    # export wisdom
    pickle.dump(pyfftw.export_wisdom(), open("wisdom.p", "wb"))
    tif.imsave("decon image.tif", decon.astype(np.float32))
    print("finished!")

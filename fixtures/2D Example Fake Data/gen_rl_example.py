#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gen_rl_example.py
"""
A short script to generate and output example data for the RL algorithm.

Copyright (c) 2016, David Hoffman
"""
import pathlib
import time
from collections import OrderedDict
from itertools import product

import numpy as np
import tifffile as tif
from matplotlib import pyplot as plt
from pydecon.decon import richardson_lucy as rl
from pydecon.utils import _ensure_positive
from scipy.ndimage import convolve
from skimage.restoration import richardson_lucy as rl_skimage


def gen_data(parent):
    """Generate data."""
    print("Generating the data ...")
    x = np.linspace(-2.5, 2.5, 64, True)
    kernel = np.exp(-(x**2))
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
    tif.imsave(parent / "ground truth.tif", signal.astype(np.float32))
    tif.imsave(parent / "blurred.tif", blurred.astype(np.float32))
    tif.imsave(parent / "psf.tif", k_norm.astype(np.float32))

    # add some noise to both
    print("Add noise ...")
    np.random.seed(12345)
    blurred[blurred < 0] = 0
    blurred_noisy = np.random.poisson(blurred) + np.random.randn(*blurred.shape) ** 2
    kernel[kernel < 0] = 0
    psf = np.random.poisson(kernel * 100) + np.random.randn(*kernel.shape) ** 2
    psf /= psf.sum()

    print("Save noisy data ...")
    tif.imsave(parent / "image.tif", blurred_noisy.astype(np.float32))
    tif.imsave(parent / "noisy psf.tif", psf.astype(np.float32))

    return signal, blurred, kernel, blurred_noisy, psf


def plot(data):
    """Make the plot."""
    num = len(data)
    fig, axs = plt.subplots(2, num, figsize=(num * 6, 12))
    for (k, d), (ax1, ax2) in zip(data.items(), axs.T):
        ax1.matshow(d, cmap="Greys_r")
        ax1.set_title(k)
        ax2.plot(d[d.shape[0] // 2])
    for ax in axs.ravel():
        ax.axis("off")
    return fig, axs


if __name__ == "__main__":

    thispath = pathlib.Path(__file__).parent

    data = OrderedDict()

    # generate data
    signal, blurred, kernel, blurred_noisy, psf = gen_data(thispath)
    data["Ground Truth"] = signal
    data["Blurred"] = blurred
    data["Noisy"] = blurred_noisy

    # deconvolve data
    print("Deconvolving the data ...", end="")
    for iters, core_type, init in product((10, 20), ("matlab", "fast"), ("matlab", "mean")):
        type_str = "iters={} core_type={} init={}".format(iters, core_type, init)
        print(type_str, end=" ...")
        start = time.time()
        decon = rl(blurred_noisy, psf, iters, 1, core_type, init)
        print("{:.3f} seconds".format(time.time() - start))
        print("Saving deconvolved data ...")
        tif.imsave(thispath / "decon image {}.tif".format(type_str), decon.astype(np.float32))
        data[type_str] = decon

    fig, axs = plot(data)
    fig.savefig(thispath / "Comparison.png", dpi=300, bbox_inches="tight")

    # decon with skimage
    print("Deconvolving the data with skimage...", end="")
    start = time.time()
    decon_skimage = rl_skimage(blurred_noisy, psf, 10, False)
    print("{:.3f} seconds".format(time.time() - start))

    tif.imsave(thispath / "decon skimage.tif", decon.astype(np.float32))
    print("finished!")

    plt.show()

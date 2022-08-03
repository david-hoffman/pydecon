#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gen_rl_example3d.py
"""
A short script to generate and output example data for the RL algorithm.

Copyright (c) 2016, David Hoffman
"""

import pathlib
import time

import numpy as np
import tifffile as tif
from skimage.restoration import richardson_lucy as rl_skimage

from pydecon.decon import richardson_lucy as rl

if __name__ == "__main__":
    thispath = pathlib.Path(__file__).parent

    print("Read data ...")
    data = tif.imread(thispath / "Real 3D Data.tif")
    psf = tif.imread(thispath / "Real 3D PSF.tif")

    print("Deconvolving the data ... ", end="")
    start = time.time()
    decon = rl(data, psf, 10, 1)
    print("{:.3f} seconds".format(time.time() - start))

    print("Saving deconvolved data ... ", end="")
    tif.imsave(thispath / "decon image.tif", decon.astype(np.float32))
    print("finished!")

    print("Deconvolving the data with skimage...", end="")
    start = time.time()
    decon_skimage = rl_skimage(data, psf, 10, False)
    print("{:.3f} seconds".format(time.time() - start))
    tif.imsave(thispath / "decon skimage.tif", decon_skimage.astype(np.float32))
    print("finished!")

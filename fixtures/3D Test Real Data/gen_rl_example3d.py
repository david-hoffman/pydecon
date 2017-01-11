#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gen_rl_example3d.py
"""
A short script to generate and output example data for the RL algorithm

Copyright (c) 2016, David Hoffman
"""
import time
import pickle
import pyfftw
import numpy as np
from skimage.external import tifffile as tif
from skimage.restoration import richardson_lucy as rl_skimage
from pyDecon.decon import richardson_lucy as rl

if __name__ == "__main__":
    print("Read data ...")
    data = tif.imread("Real 3D Data.tif")
    psf = tif.imread("Real 3D PSF.tif")
    # deconvolve data
    print("Deconvolving the data ...", end="")
    # see if you can load wisdom
    try:
        pyfftw.import_wisdom(pickle.load(open("wisdom.p", "rb")))
    except FileNotFoundError:
        pass
    start = time.time()
    decon = rl(data, psf, 10, 1, threads=24)
    print("{:.3f} seconds".format(time.time() - start))
    print("Saving deconvolved data ...")
    # export wisdom
    pickle.dump(pyfftw.export_wisdom(), open("wisdom.p", "wb"))
    tif.imsave("decon image.tif", decon.astype(np.float32))
    # decon with skimage
    print("Deconvolving the data with skimage...", end="")
    start = time.time()
    decon_skimage = rl_skimage(data, psf, 10, False)
    print("{:.3f} seconds".format(time.time() - start))
    tif.imsave("decon skimage.tif", decon_skimage.astype(np.float32))
    print("finished!")

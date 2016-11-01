#!/usr/bin/env python
# -*- coding: utf-8 -*-
# utils.py
"""
Utility functions for deconvolution

Copyright (c) 2016, David Hoffman
"""

import numpy as np


def set_pyfftw_threads(threads=1):
    """A utility to set the number of threads to use in pyfftw"""
    raise NotImplementedError


def _ensure_positive(data):
    """Make sure data is positive and has no zeros

    For numerical stability

    If we realize that mutating data is not a problem
    and that changing in place could lead to signifcant
    speed ups we can lose the data.copy() line"""
    # make a copy of the data
    data = data.copy()
    data[data <= 0] = np.finfo(data.dtype).resolution
    return data


def _prep_img_and_psf(image, psf):
    """Do basic data checking, convert data to float, normalize psf and make
    sure data are positive"""
    assert psf.ndim == image.ndim, ("image and psf do not have the same number"
                                    " of dimensions")
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    # normalize the kernel
    psf /= psf.sum()
    # need to make sure both image and PSF are totally positive.
    image = _ensure_positive(image)
    psf = _ensure_positive(psf)
    return image, psf

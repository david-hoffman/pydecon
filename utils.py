#!/usr/bin/env python
# -*- coding: utf-8 -*-
# utils.py
"""
Utility functions for deconvolution

Copyright (c) 2016, David Hoffman
"""

import numpy as np
from dphutils import radial_profile


def set_pyfftw_threads(threads=1):
    """A utility to set the number of threads to use in pyfftw"""
    raise NotImplementedError


def _ensure_positive(data):
    """Make sure data is positive"""
    return np.fmax(data, 0)


def _zero2eps(data):
    """Replace zeros and negative numbers with machine precision"""
    return np.fmax(data, np.finfo(data.dtype).eps)


def _prep_img_and_psf(image, psf):
    """Do basic data checking, convert data to float, normalize psf and make
    sure data are positive"""
    assert psf.ndim == image.ndim, ("image and psf do not have the same number"
                                    " of dimensions")
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    # need to make sure both image and PSF are totally positive.
    image = _ensure_positive(image)
    psf = _ensure_positive(psf)
    # normalize the kernel
    psf /= psf.sum()
    return image, psf


def radialavg(data):
    """Radially average psf/otf"""
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(
            "Data has wrong number of dimensions, ndim = {}".format(data.ndim))
    # find data maximum, then we use this as the center
    center = np.unravel_index(data.argmax(), data.shape)
    yxcenter = center[-2:]
    # figure out maxsize of data that is reasonable
    maxsize = max(*yxcenter, *(np.array(data.shape[-2:]) - np.array(yxcenter)))
    # maxsize should be odd
    maxsize += 1 - maxsize % 2
    if data.ndim == 2:
        return radial_profile(data, yxcenter)[0][:maxsize]
    elif data.ndim == 3:
        # return the radial profile for each z slice
        return np.array([radial_profile(d, yxcenter)[0][:maxsize] for d in data])
    else:
        raise RuntimeError("Something has gone wrong!")


def expand_radialavg(data):
    """Expand a radially averaged data set to a full 2D or 3D psf/otf

    Data will have maximum at center

    Assumes standard numpy ordering of axes (i.e. zyx)"""
    ndim = data.ndim
    if ndim < 1 or ndim > 2:
        raise ValueError(
            "Data has wrong number of dimensions, ndim = {}".format(data.ndim))
    if ndim == 1:
        # we know tha the above makes the data odd
        yxsize = data.size * 2 - 1
        # define the new datashape
    elif ndim == 2:
        yxsize = data.shape[-1] * 2 - 1
    else:
        raise RuntimeError("Something has gone wrong!")

    datashape = (yxsize, yxsize)
    # start building the coordinate system
    idx = np.indices((datashape))
    center = np.array(datashape) // 2
    # calculate the radius from center
    idx2 = idx - center[[Ellipsis] + [np.newaxis] * len(datashape)]
    r = np.sqrt(np.sum([i**2 for i in idx2], 0))
    # figure out old r for the averaged data
    oldr = np.arange((yxsize + 1) // 2)
    if ndim == 1:
        return np.interp(r, oldr, data)
    else:
        return np.array([np.interp(r, oldr, d) for d in data])

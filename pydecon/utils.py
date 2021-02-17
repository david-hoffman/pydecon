#!/usr/bin/env python
# -*- coding: utf-8 -*-
# utils.py
"""
Utility functions for deconvolution

Copyright (c) 2016, David Hoffman
"""

import numpy as np
from scipy.fftpack.helper import next_fast_len


def radial_profile(data, center=None, binsize=1.0):
    """Take the radial average of a 2D data array

    Adapted from http://stackoverflow.com/a/21242776/5030014

    Parameters
    ----------
    data : ndarray (2D)
        the 2D array for which you want to calculate the radial average
    center : sequence
        the center about which you want to calculate the radial average
    binsize : sequence
        Size of radial bins, numbers less than one have questionable utility

    Returns
    -------
    radial_mean : ndarray
        a 1D radial average of data
    radial_std : ndarray
        a 1D radial standard deviation of data

    Examples
    --------
    >>> radial_profile(np.ones((11, 11)))
    (array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]))
    """
    # test if the data is complex
    if np.iscomplexobj(data):
        # if it is complex, call this function on the real and
        # imaginary parts and return the complex sum.
        real_prof, real_std = radial_profile(np.real(data), center, binsize)
        imag_prof, imag_std = radial_profile(np.imag(data), center, binsize)
        return real_prof + imag_prof * 1j, np.sqrt(real_std ** 2 + imag_std ** 2)
        # or do mag and phase
        # mag_prof, mag_std = radial_profile(np.abs(data), center, binsize)
        # phase_prof, phase_std = radial_profile(np.angle(data), center, binsize)
        # return mag_prof * np.exp(phase_prof * 1j), mag_std * np.exp(phase_std * 1j)
    # pull the data shape
    idx = np.indices((data.shape))
    if center is None:
        # find the center
        center = np.array(data.shape) // 2
    else:
        # make sure center is an array.
        center = np.asarray(center)
    # calculate the radius from center
    idx2 = idx - center[(Ellipsis,) + (np.newaxis,) * (data.ndim)]
    r = np.sqrt(np.sum([i ** 2 for i in idx2], 0))
    # convert to int
    r = np.round(r / binsize).astype(np.int)
    # sum the values at equal r
    tbin = np.bincount(r.ravel(), data.ravel())
    # sum the squares at equal r
    tbin2 = np.bincount(r.ravel(), (data ** 2).ravel())
    # find how many equal r's there are
    nr = np.bincount(r.ravel())
    # calculate the radial mean
    # NOTE: because nr could be zero (for missing bins) the results will
    # have NaN for binsize != 1
    radial_mean = tbin / nr
    # calculate the radial std
    radial_std = np.sqrt(tbin2 / nr - radial_mean ** 2)
    # return them
    return radial_mean, radial_std


def _fft_pad(array, newshape=None, mode="median", **kwargs):
    """Pad an array to prep it for fft"""
    # pull the old shape
    oldshape = array.shape
    if newshape is None:
        # update each dimension to a 5-smooth hamming number
        newshape = tuple(next_fast_len(n) for n in oldshape)
    else:
        if isinstance(newshape, int):
            newshape = tuple(newshape for n in oldshape)
        else:
            newshape = tuple(newshape)
    # generate padding and slices
    padding, slices = _padding_slices(oldshape, newshape)
    return np.pad(array[slices], padding, mode=mode, **kwargs)


def _padding_slices(oldshape, newshape):
    """This function takes the old shape and the new shape and calculates
    the required padding or cropping.newshape

    Can be used to generate the slices needed to undo fft_pad above"""
    # generate pad widths from new shape
    padding = tuple(
        _calc_pad(o, n) if n is not None else _calc_pad(o, o) for o, n in zip(oldshape, newshape)
    )
    # Make a crop list, if any of the padding is negative
    slices = tuple(_calc_crop(s1, s2) for s1, s2 in padding)
    # leave 0 pad width where it was cropped
    padding = [(max(s1, 0), max(s2, 0)) for s1, s2 in padding]
    return padding, slices


def _calc_crop(s1, s2):
    """Calc the cropping from the padding"""
    a1 = abs(s1) if s1 < 0 else None
    a2 = s2 if s2 < 0 else None
    return slice(a1, a2, None)


def _calc_pad(oldnum, newnum):
    """ Calculate the proper padding for fft_pad

    We have three cases:
    old number even new number even
    >>> _calc_pad(10, 16)
    (3, 3)

    old number odd new number even
    >>> _calc_pad(11, 16)
    (2, 3)

    old number odd new number odd
    >>> _calc_pad(11, 17)
    (3, 3)

    old number even new number odd
    >>> _calc_pad(10, 17)
    (4, 3)

    same numbers
    >>> _calc_pad(17, 17)
    (0, 0)

    from larger to smaller.
    >>> _calc_pad(17, 10)
    (-4, -3)
    """
    # how much do we need to add?
    width = newnum - oldnum
    # calculate one side, smaller
    pad_s = width // 2
    # calculate the other, bigger
    pad_b = width - pad_s
    # if oldnum is odd and newnum is even
    # we want to pull things backward
    if oldnum % 2:
        pad1, pad2 = pad_s, pad_b
    else:
        pad1, pad2 = pad_b, pad_s
    return pad1, pad2


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
    assert psf.ndim == image.ndim, "image and psf do not have the same number" " of dimensions"
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    # need to make sure both image and PSF are totally positive.
    image = _ensure_positive(image)
    # I'm not actually sure if this step is necessary or a good idea.
    psf = _ensure_positive(psf)
    # normalize the kernel
    psf /= psf.sum()
    return image, psf


def radialavg(data):
    """Radially average psf/otf

    Note: it only really makes sense to radially average the OTF"""
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError("Data has wrong number of dimensions, ndim = {}".format(data.ndim))
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


# fixes fft issue
def expand_radialavg(data):
    """Expand a radially averaged data set to a full 2D or 3D psf/otf

    Data will have maximum at center

    Assumes standard numpy ordering of axes (i.e. zyx)"""
    ndim = data.ndim
    if ndim < 1 or ndim > 2:
        raise ValueError("Data has wrong number of dimensions, ndim = {}".format(data.ndim))
    half_yxsize = data.shape[-1]
    quadsize = half_yxsize + 1
    datashape = (quadsize, quadsize)
    # start building the coordinate system
    idx = np.indices((datashape))
    # calculate the radius from center
    r = np.sqrt(np.sum([i ** 2 for i in idx], 0))
    # figure out old r for the averaged data
    oldr = np.arange(half_yxsize)
    # final shape
    final_shape = (2 * half_yxsize,) * 2
    if ndim == 1:
        lrquad = np.interp(r, oldr, data)
    else:
        final_shape = (data.shape[0],) + final_shape
        lrquad = np.array([np.interp(r, oldr, d) for d in data])
    # make final array to fill
    final_ar = np.empty(final_shape, dtype=lrquad.dtype)
    # fill each quadrant
    final_ar[..., half_yxsize:, half_yxsize:] = lrquad[..., :-1, :-1]
    final_ar[..., :half_yxsize, half_yxsize:] = lrquad[..., :0:-1, :-1]
    final_ar[..., half_yxsize:, :half_yxsize] = lrquad[..., :-1, :0:-1]
    final_ar[..., :half_yxsize, :half_yxsize] = lrquad[..., :0:-1, :0:-1]
    return final_ar

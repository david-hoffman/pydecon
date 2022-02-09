"""LR deconvolution using cupy (or numpy).

based on code by David Hoffman
https://github.com/david-hoffman/pyDecon
which is
Copyright (c) 2016, David Hoffman


modified to work with cupy Aug/Sep/Oct 2020, Volker Hilsenstein

In contrast to David Hoffman's code, all util functions for padding,
psf prep etc. are included in this file, so that this file 
is a self-contained 

Changelog:
27.Oct.2020:
Removed all cupy imports, importing 
numpy is sufficient because cupy arrays adhere to NEP-18.
Passing in a cupy array will result in dispatching GPU-accelerated
cupy functions, which I learned about here
 https://github.com/dask/dask-blog/pull/77

"""

import scipy.fft
import numpy as np

# Note: all this is using numpy functions, but due to cupy implementing
# the NEP-18 dispatch mechanism
# https://numpy.org/neps/nep-0018-array-function-protocol.html
# this code works out of the box with cupy arrays with full GPU support.


# Note that cupy > 8.0.0 implements fft plan caching, which
# gives an additional speedup

from numpy import pad, fmax, clip, finfo, conj, roll, square, zeros_like
from numpy import float as floattype
from numpy.fft import *


def _prep_img_and_psf(image, psf):
    """Do basic data checking, convert data to float, normalize psf and make sure data are positive."""
    assert psf.ndim == image.ndim, "image and psf do not have the same number" " of dimensions"
    image = image.astype(floattype)
    psf = psf.astype(floattype)
    # need to make sure both image and PSF are totally positive.
    image = _ensure_positive(image)
    # I'm not actually sure if this step is necessary or a good idea.
    psf = _ensure_positive(psf)
    # normalize the kernel
    psf /= psf.sum()
    return image, psf


def _calc_crop(s1, s2):
    """Calc the cropping from the padding."""
    a1 = abs(s1) if s1 < 0 else None
    a2 = s2 if s2 < 0 else None
    return slice(a1, a2, None)


def _calc_pad(oldnum, newnum):
    """Calculate the proper padding for fft_pad.

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


def fft_pad(array, newshape=None, mode="median", **kwargs):
    """Pad an array to prep it for fft."""
    # pull the old shape
    oldshape = array.shape
    if newshape is None:
        # update each dimension to a 5-smooth hamming number
        newshape = tuple(scipy.fft.helper.next_fast_len(n) for n in oldshape)
    else:
        if isinstance(newshape, int):
            newshape = tuple(newshape for n in oldshape)
        else:
            newshape = tuple(newshape)
    # generate padding and slices
    padding, slices = padding_slices(oldshape, newshape)
    return pad(array[slices], padding, mode=mode, **kwargs)


def padding_slices(oldshape, newshape):
    """Pad slices."""
    # generate pad widths from new shape
    padding = tuple(
        _calc_pad(o, n) if n is not None else _calc_pad(o, o) for o, n in zip(oldshape, newshape)
    )
    # Make a crop list, if any of the padding is negative
    slices = tuple(_calc_crop(s1, s2) for s1, s2 in padding)
    # leave 0 pad width where it was cropped
    padding = [(max(s1, 0), max(s2, 0)) for s1, s2 in padding]
    return padding, slices


def _ensure_positive(data):
    """Make sure data is positive."""
    return fmax(data, 0)


def _zero2eps(data):
    """Replace zeros and negative numbers with machine precision."""
    return fmax(data, finfo(data.dtype).eps)


def _prep(img, psf):
    img, psf = _prep_img_and_psf(img, psf)
    if psf.shape != img.shape:
        # its been assumed that the background of the psf has already been
        # removed and that the psf has already been centered
        psf = fft_pad(psf, img.shape, mode="constant")
    otf = rfftn(ifftshift(psf))
    return img, psf, otf


def _richardson_lucy(img, psf, otf, iterations=10, accelerate=True, **kwargs):
    """Perform actual RL deconvolution.

    assumes that img, psf, and otf have been prepared.

    Typically, you would call richardson_lucy (see docstring there)
    """
    eps = finfo(img.dtype).eps

    t = img
    tm1 = zeros_like(t)
    g_tm1 = zeros_like(t)
    g_tm2 = zeros_like(t)

    for i in range(iterations):
        if accelerate:
            # first order andrew biggs acceleration
            alpha = (g_tm1 * g_tm2).sum() / (square(g_tm2).sum() + eps)
            alpha = clip(alpha, 0, 1)
            h1_t = t - tm1
            y = t + alpha * h1_t
        else:
            y = t
        t = _ensure_positive(y)
        # update
        reblur = irfftn(otf * rfftn(t, t.shape, **kwargs), t.shape, **kwargs)
        reblur = _zero2eps(reblur)
        im_ratio = img / reblur
        estimate = irfftn(
            conj(otf) * rfftn(im_ratio, im_ratio.shape, **kwargs), im_ratio.shape, **kwargs
        )

        # The below is to compensate for the slight shift that using np.conj
        # can introduce verus actually reversing the PSF. See notebooks for
        # details.
        for i, (s, p) in enumerate(zip(img.shape, psf.shape)):
            if s % 2 and not p % 2:
                estimate = roll(estimate, 1, i)
        estimate = _ensure_positive(estimate)
        tp1 = t * estimate
        # update g's
        g_tm2 = g_tm1
        # this is where the magic is, we need to compute from previous step
        # which may have been augmented by acceleration
        g_tm1 = tp1 - y
        t, tm1 = tp1, t
    return t


def richardson_lucy(img, psf, iterations=10, accelerate=True, **kwargs):
    """Richardson-Lucy Deconvolution.

    Perform RL deconvolution with (optional) Andrew-Biggs acceleration
    on array img with point spread function psf.

    If numpy arrays are passed in as img, psf the deconvolution is performed
    using numpy (CPU computation), if cupy arrays are passed in, the deconvolution
    is performed using cupy (GPU computation using CUDA). Use cupy version >= 8.0
    as it implements FFT plan chaching which significantly speeds up the computation.

    img: numpy or cupy array
    psf: numpy or cupy array
    iterations: number of RL-iterations
    accelerate: boolean, use first order Andrew-Biggs acceleration if True
    **kwargs: these kwargs will be passed on to the rfftn/irfftn functions
    """
    # set up
    img, psf, otf = _prep(img, psf)
    # perform actual iterations
    return _richardson_lucy(img, psf, otf, iterations, accelerate, **kwargs)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# decon.py
"""
Functions that actually perform the deconvolution.

Copyright (c) 2016, David Hoffman
"""

import numpy as np
try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import (fftshift, ifftshift, fftn, ifftn,
                                             rfftn, irfftn)
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import (fftshift, ifftshift, fftn, ifftn,
                           rfftn, irfftn)
from dphutils import fft_pad
from .utils import _prep_img_and_psf, _ensure_positive
import scipy.signal.signaltools as sig
from scipy.signal import fftconvolve
from scipy.ndimage import convolve


def _get_fshape_slice(image, psf):
    """This is necessary for the fast Richardson-Lucy Algorithm"""
    s1 = np.array(image.shape)
    s2 = np.array(psf.shape)
    assert (s1 >= s2).all()
    shape = s1 + s2 - 1
    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [sig.fftpack.helper.next_fast_len(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    return fshape, fslice


def _rl_core_direct(image, psf, u_t):
    """The core update step of the RL algorithm

    An exact version that uses direct convolution"""
    reblur = convolve(u_t, psf, mode="reflect")
    reblur = _ensure_positive(reblur)
    im_ratio = image / reblur
    # reverse slicing
    s = [slice(None, None, -1)] * psf.ndim
    estimate = convolve(im_ratio, psf[s], mode="reflect")
    return u_t * estimate


def _rl_core_accurate(image, psf, u_t):
    """The core update step of the RL algorithm

    An accurate version that """
    reblur = fftconvolve(u_t, psf, "same")
    reblur = _ensure_positive(reblur)
    im_ratio = image / reblur
    # reverse slicing
    s = slice(None, None, -1)
    estimate = fftconvolve(im_ratio, psf[(s, ) * psf.ndim], "same")
    return u_t * estimate


def _rl_core_matlab(image, otf, u_t):
    """The core update step of the RL algorithm

    This is a fast but inaccurate version modeled on matlab's version"""
    reblur = irfftn(otf * rfftn(u_t), u_t.shape)
    reblur[reblur <= 0] = np.finfo(reblur.dtype).resolution
    im_ratio = image / reblur
    estimate = irfftn(np.conj(otf) * rfftn(im_ratio), im_ratio.shape)
    # need to figure out a way to pass the psf shape
    for i, (s, p) in enumerate(zip(image.shape, psf.shape)):
        if s % 2 and not p % 2:
            estimate = np.roll(estimate, 1, i)
    return u_t * estimate


def _rl_accelerate(g_tm1, g_tm2, u_t, u_tm1, u_tm2, prediction_order):
    """Biggs-Andrews Acceleration

    .. [2] Biggs, D. S. C.; Andrews, M. Acceleration of Iterative Image
    Restoration Algorithms. Applied Optics 1997, 36 (8), 1766."""
    alpha = (g_tm1 * g_tm2).sum() / (g_tm2**2).sum()
    alpha = max(min(alpha, 1), 0)
    # if alpha is positive calculate predicted step
    if alpha:
        # first order correction
        h1_t = u_t - u_tm1
        if prediction_order > 1:
            # second order correction
            h2_t = (u_t - 2 * u_tm1 + u_tm2)
        else:
            h2_t = 0
        u_tp1 = u_t + alpha * h1_t + alpha**2 / 2 * h2_t
        return u_tp1
    else:
        return u_t


def _rl_core_fast_accurate(image, otf, iotf, u_t, fshape, fslice, **kwargs):
    """The core update step of the RL algorithm

    This one is fast and accurate. It does proper fft convolution
    steps (based on fftconvolve from scipy.signal) but is optimized
    to minimize the actual number of ffts performed. For oddly
    shaped data it is the fasted algorithm.


    We may be able to speed up the processing a little by avoiding all the
    copies but it doesn't seem to be the bottle neck."""
    # reblur the estimate
    reblur = irfftn(rfftn(u_t, fshape, **kwargs) * otf,
                    fshape, **kwargs)[fslice]
    reblur = sig._centered(reblur, image.shape)
    reblur = _ensure_positive(reblur)
    # calculate the difference with the image
    im_ratio = image / reblur
    # convolve the difference ratio with the inverse psf
    estimate = irfftn(rfftn(im_ratio, fshape, **kwargs) * iotf,
                      fshape, **kwargs)[fslice]
    estimate = sig._centered(estimate, image.shape)
    # multiply with the previous estimate to get the new estimate
    return u_t * estimate


def richardson_lucy(image, psf, iterations=10, prediction_order=1,
                    core=_rl_core_fast_accurate, init="mean", **kwargs):
    """
    Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function. Assumes that it has no background
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    prediction_order : int (0, 1 or 2)
        Use Biggs-Andrews to accelerate the algorithm [2]

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.

    Examples
    --------

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] Biggs, D. S. C.; Andrews, M. Acceleration of Iterative Image
    Restoration Algorithms. Applied Optics 1997, 36 (8), 1766.

    """
    # Stolen from the dev branch of skimage because stable branch is slow
    # checked against matlab on 20160805 and agrees to within machine precision
    image, psf = _prep_img_and_psf(image, psf)
    if core is _rl_core_fast_accurate:
        fshape, fslice = _get_fshape_slice(image, psf)
        otf = rfftn(psf, fshape, **kwargs)
        rev_slice = [slice(None, None, -1)] * psf.ndim
        iotf = rfftn(psf[rev_slice], fshape, **kwargs)
        core_dict = dict(
            image=image, otf=otf, iotf=iotf, fshape=fshape, fslice=fslice
        )
    elif core is _rl_core_accurate or core is _rl_core_direct:
        core_dict = dict(image=image, psf=psf)
    elif core is _rl_core_matlab:
        image, psf = _prep_img_and_psf(image, psf)
        if psf.shape != image.shape:
            # its been assumed that the background of the psf has already been
            # removed and that the psf has already been centered
            psf = fft_pad(psf, image.shape, mode='constant')
        otf = rfftn(ifftshift(psf))
        core_dict = dict(image=image, otf=otf)
    else:
        raise TypeError("{} is not a valid core".format(core))
    # initialize variable for iterations
    # previous estimate
    u_tm1 = None
    if init == "matlab":
        core_dict["u_t"] = u_t = image
    else:
        # current estimate, for the initial estimate we use the mean of the
        # data this promotes a smooth solution and helps to reduce noise.
        core_dict["u_t"] = u_t = np.ones_like(image) * image.mean()
    # previous difference
    g_tm1 = None
    for i in range(iterations):
        # call the update function
        u_tp1 = core(**core_dict, **kwargs)
        # update
        # update g's
        g_tm2 = g_tm1
        g_tm1 = u_tp1 - u_t
        # now move u's along
        u_tm2 = u_tm1
        u_tm1 = u_t
        u_t = u_tp1
        if prediction_order:
            if i > 2:
                u_t = _rl_accelerate(g_tm1, g_tm2, u_t, u_tm1, u_tm2,
                                     prediction_order)
            elif i > 1:
                u_t = _rl_accelerate(g_tm1, g_tm2, u_t, u_tm1, u_tm2,
                                     1)
        # update estimate and ensure positive
        core_dict["u_t"] = u_t = _ensure_positive(u_t)
    # return final estimate
    return u_tp1


def wiener_filter(image, psf, reg, **kwargs):
    """Wiener Deconvolution

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    reg : float
        The regularization parameter, this is in place of the SNR
        which in most cases isn't known as a function of frequency

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.

    Notes
    -----
    Even though the SNR is not usually known the sharp drop of OTF's
    at the band limit usually mean that simply estimating the regularization
    is sufficient.
    """
    image, psf = _prep_img_and_psf(image, psf)
    if psf.shape != image.shape:
        # its been assumed that the background of the psf has already been
        # removed and that the psf has already been centered
        psf = fft_pad(psf, image.shape, mode='constant')
    otf = rfftn(ifftshift(psf), **kwargs)
    filt = np.conj(otf) / (abs(otf)**2 + reg**2)
    im_deconv = np.irfftn(filt * rfftn(ifftshift(image), **kwargs),
                          image.shape, **kwargs)
    return im_deconv


if __name__ == '__main__':
    from skimage.data import hubble_deep_field
    from skimage.color import rgb2grey
    from matplotlib import pyplot as plt
    plt.rcParams["image.cmap"] = "Greys_r"
    np.random.seed(12345)
    image = rgb2grey(hubble_deep_field())
    image = image / image.max()
    image *= 25.0
    x = np.linspace(-4, 4, 64)
    xx, yy = np.meshgrid(x, x)
    psf = np.exp(-(xx**2 + yy**2)) * 100
    blur_image = convolve(image, psf / psf.sum(), mode="reflect")
    blur_image_noisy = np.random.poisson(blur_image)
    psf_noisy = np.random.poisson(psf)
    deblurred_image = richardson_lucy(blur_image_noisy,
                                      psf_noisy / psf_noisy.sum(), 10, 1)
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    titles = ("Image" ,"PSF", "Blurred Image", "Noisy PSF", "Noisy Image", "Deconvolved")
    datas = (image, psf, blur_image, psf_noisy, blur_image_noisy, deblurred_image)
    for ax, t, d in zip(axs.ravel(), titles, datas):
        ax.matshow(d)
        ax.set_title(t)

    plt.show()

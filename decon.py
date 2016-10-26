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


def _prep_img_and_psf(image, psf):
    """Ensure that image and psf have the same size"""
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    assert psf.ndim == image.ndim, ("image and psf do not have the same number"
                                    " of dimensions")
    psf_norm = psf / psf.sum()
    if psf_norm.shape != image.shape:
        # its been assumed that the background of the psf has already been removed
        psf_norm = fft_pad(psf_norm, image.shape, mode='constant')
    assert psf_norm.shape == image.shape
    return image, psf_norm


def wiener_filter(image, psf, reg):
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
    otf = rfftn(ifftshift(psf_norm))
    filt = np.conj(otf) / (abs(otf)**2 + reg**2)
    im_deconv = np.irfftn(filt * rfftn(ifftshift(image)), image.shape)
    return im_deconv


def richardson_lucy(image, psf, iterations=10, prediction_order=2):
    """
    Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
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
    .. [2] Biggs, D. S. C.; Andrews, M. Acceleration of Iterative Image Restoration
    Algorithms. Applied Optics 1997, 36 (8), 1766.

    """
    # Stolen from the dev branch of skimage because stable branch is slow
    # checked against matlab on 20160805 and agrees to within machine precision
    image, psf = _prep_img_and_psf(image, psf)
    otf = rfftn(psf)
    # initialize variable for iterations
    # previous estimate
    u_tm1 = None
    # current estimate
    u_t = image
    # previous difference
    g_tm1 = None
    for i in range(iterations):
        # call the update function
        u_tp1 = rl_core(image, otf, u_t)
        # update
        if prediction_order:
            u_tm2 = u_tm1
            u_tm1 = u_t
            g_tm2 = g_tm1
            g_tm1 = u_tp1 - u_t
            if i > 2:
                temp = rl_accelerate(g_tm1, g_tm2, u_t, u_tm1, u_tm2,
                                  prediction_order)
                if temp is not None:
                    u_tp1 = temp
        # enure positivity
        u_tp1[u_tp1 < 0] = 0
        # update estimate
        u_t = u_tp1
    # return final estimate
    return u_tp1


def rl_core(image, otf, y_t):
    """The core update step of the RL algorithm"""
    reblur = irfftn(otf * rfftn(y_t), y_t.shape)
    im_ratio = image / reblur
    estimate = irfftn(np.conj(otf) * rfftn(im_ratio), im_ratio.shape)
    return y_t * estimate


def rl_accelerate(g_tm1, g_tm2, u_t, u_tm1, u_tm2, prediction_order):
    """Biggs-Andrews Acceleration"""
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
        return None


if __name__ == '__main__':
    from skimage.data import moon
    from skimage.color import rgb2grey
    from scipy.signal import fftconvolve
    from matplotlib import pyplot as plt
    plt.rcParams["image.cmap"] = "Greys_r"
    image = rgb2grey(moon()).astype(float)
    x = np.linspace(-4, 4, 64)
    xx, yy = np.meshgrid(x, x)
    psf = np.exp(-(xx**2 + yy**2))
    blur_image = fftconvolve(image, psf, "same")
    deblurred_image = richardson_lucy(blur_image, psf, 10, False)
    fig, axs = plt.subplots(2, 2)
    axs.shape = (-1, )
    axs[0].matshow(image)
    axs[1].matshow(psf)
    axs[2].matshow(blur_image)
    axs[3].matshow(deblurred_image)
    plt.show()

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
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    assert psf.ndim == image.ndim, ("image and psf do not have the same number"
                                    " of dimensions")
    psf_norm = psf / psf.sum()
    if psf_norm.shape != image.shape:
        psf_norm = fft_pad(psf_norm, image.shape, mode='constant')
    assert psf_norm.shape == image.shape
    # initialize variable for iterations
    # previous estimate
    u_tm1 = None
    # current estimate
    u_t = image
    # previous difference
    g_tm1 = None
    # below needs to be normalized.
    otf = rfftn(ifftshift(psf_norm))
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
                u_tp1 = rl_accelerate(g_tm1, g_tm2, u_t, u_tm1, u_tm2,
                                  prediction_order)
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


if __name__ == '__main__':
    from skimage.data import moon
    from dphutils import richardson_lucy as rl2
    from skimage.color import rgb2grey
    from scipy.ndimage import convolve
    from matplotlib import pyplot as plt
    image = rgb2grey(moon()).astype(float)
    x = np.linspace(-4, 4, 33)
    xx, yy = np.meshgrid(x, x)
    psf = np.exp(-(xx**2 + yy**2))
    blur_image = convolve(image, psf)
    deblurred_image = richardson_lucy(blur_image, psf, 10, prediction_order=False)
    deblurred_image2 = rl2(blur_image, psf, 10, prediction_order=False)
    fig, axs = plt.subplots(2, 3)
    axs.shape = (-1, )
    axs[0].matshow(image)
    axs[1].matshow(psf)
    axs[2].matshow(blur_image)
    axs[3].matshow(deblurred_image)
    axs[4].matshow(deblurred_image2)
    axs[5].hist((deblurred_image- deblurred_image2).ravel())
    plt.show()

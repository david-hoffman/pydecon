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
    # Build the dictionary to pass around and update
    psf_norm = psf / psf.sum()
    if psf_norm.shape != image.shape:
        psf_norm = fft_pad(psf_norm, image.shape, mode='constant')
    u_tm2 = None
    u_tm1 = None
    g_tm2 = None
    g_tm1 = None
    u_t = None
    y_t = image
    # below needs to be normalized.
    otf = rfftn(ifftshift(psf_norm))

    for i in range(iterations):
        # call the update function
        # make mirror psf
        # calculate RL iteration using the predicted step (y_t)
        reblur = np.real(irfftn(otf * rfftn(y_t)))
        # assert (reblur > eps).all(), 'Reblur 0 or negative'
        im_ratio = image / reblur
        # assert (im_ratio > eps).all(), 'im_ratio 0 or negative'
        estimate = np.real(irfftn(np.conj(otf) * rfftn(im_ratio)))
        # assert (estimate > eps).all(), 'im_ratio 0 or negative'
        u_tp1 = y_t * estimate

        # enforce non-negativity
        u_tp1[u_tp1 < 0] = 0

        # update
        u_tm2 = u_tm1
        u_tm1 = u_t
        u_t = u_tp1
        g_tm2 = g_tm1
        g_tm1 = u_tp1 - y_t
        # initialize alpha to zero
        alpha = 0
        # run through the specified iterations
        if i > 1:
            # calculate alpha according to 2
            alpha = (g_tm1 * g_tm2).sum() / (g_tm2**2).sum()

            alpha = max(min(alpha, 1), 0)
            if not np.isfinite(alpha):
                print(alpha)
                alpha = 0
            assert alpha >= 0, alpha
            assert alpha <= 1, alpha

        # if alpha is positive calculate predicted step
        if alpha != 0:
            if prediction_order > 0:
                # first order correction
                h1_t = u_t - u_tm1
                if prediction_order > 1:
                    # second order correction
                    h2_t = (u_t - 2 * u_tm1 + u_tm2)
                else:
                    h2_t = 0
            else:
                h1_t = 0
        else:
            h2_t = 0
            h1_t = 0

        y_t = u_t + alpha * h1_t + alpha**2 / 2 * h2_t
        # enure positivity
        y_t[y_t < 0] = 0

    im_deconv = u_t

    return im_deconv


def rl_core(image, otf, y_t):
    """The core update step of the RL algorithm"""
    reblur = irfftn(otf * rfftn(y_t))
    # assert (reblur > eps).all(), 'Reblur 0 or negative'
    im_ratio = image / reblur
    # assert (im_ratio > eps).all(), 'im_ratio 0 or negative'
    estimate = np.real(irfftn(np.conj(otf) * rfftn(im_ratio)))
    # assert (estimate > eps).all(), 'im_ratio 0 or negative'
    return y_t * estimate

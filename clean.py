#!/usr/bin/env python
# -*- coding: utf-8 -*-
# clean.py
"""
Functions designed for cleaning PSFs/OTFs based on theoretical considerations

Copyright (c) 2016, David Hoffman
"""

import numpy as np
try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import (fftshift, ifftshift, fftn, ifftn,
                                             rfftn, irfftn, fftfreq)
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import (fftshift, ifftshift, fftn, ifftn,
                           rfftn, irfftn, fftfreq)
from pyOTF.utils import remove_bg


def _make_kspace(data, res):
    """Make coordinates in kspace"""
    assert data.ndim == len(res), "Resolution doesn't match shape"
    ks = tuple(fftshift(fftfreq(s, r)) for s, r in zip(data.shape, res))
    ktot = np.meshgrid(*ks, indexing="ij")
    kres = tuple(k[1] - k[0] for k in ks)
    return ktot, kres


def _make_xspace(data, res):
    """Make coordinates in x space"""
    assert data.ndim == len(res), "Resolution doesn't match shape"
    xs = tuple((np.arange(s) - s / 2) * r for s, r in zip(data.shape, res))
    xtot = np.meshgrid(*xs, indexing="ij")
    return xtot, res


def psf2dclean(psf, exp_kwargs, ns=4):
    """A function to clean up a 2d psf in both real space and frequency space

    This function makes a few assumptions about the psf, it has 

    Parameters
    ----------
    psf : ndarray (2 dimensional)
        The measured point spread function of the microscope.
    """
    # break out kwargs
    na, wl, res = exp_kwargs["na"], exp_kwargs["wl"], exp_kwargs["res"]
    # psf can be approximated by a gaussian with sigma = 0.45 * wl / (2 * na)
    # https://en.wikipedia.org/wiki/Airy_disk
    # filter in x-space by assuming that there is no data at ns sigma
    cleaned_psf = psf.copy()
    xtot, xres = _make_xspace(psf, (res, res))
    r = np.hypot(*xtot)
    mask = r > ns * 0.45 * wl / (2 * na)
    cleaned_psf[mask] = 0
    # TODO: switch fft's to rfft's
    otf = fftshift(fftn(ifftshift(cleaned_psf)))
    # make the coordinate system, assume data is centered
    ktot, kres = _make_kspace(psf, (res, res))
    # we care about radius only
    kr = np.hypot(*ktot)
    mask = kr > (2 * na / wl)
    otf[mask] = 0
    # ifft
    cleaned_psf = np.real(fftshift(ifftn(ifftshift(otf))))
    return cleaned_psf

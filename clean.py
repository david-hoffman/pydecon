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


def psf3dclean(psf, exp_kwargs, ns=4):
    """A function to clean up a 3D PSF in both real and reciprocal space

    Parameters
    ----------
    """

    # break out kwargs
    na, ni, wl, res, zres = (exp_kwargs["na"], exp_kwargs["ni"],
                             exp_kwargs["wl"], exp_kwargs["res"],
                             exp_kwargs["zres"])
    # psf can be approximated by a gaussian with sigma = 0.45 * wl / (2 * na)
    # https://en.wikipedia.org/wiki/Airy_disk
    # filter in x-space by assuming that there is no data at ns sigma
    cleaned_psf = psf.copy()
    xtot, xres = _make_xspace(psf, (zres, res, res))
    r = np.hypot(*xtot[1:])
    z = xtot[0]
    theta = np.arcsin(na / ni)
    # make a double sided cone
    mask = r - abs(z) * np.tan(theta) > ns * 0.45 * wl / (2 * na)
    cleaned_psf[mask] = 0
    # TODO: switch fft's to rfft's
    otf = fftshift(fftn(ifftshift(cleaned_psf)))
    # make the coordinate system, assume data is centered
    ktot, kres = _make_kspace(cleaned_psf, (zres, res, res))
    # we care about radius only
    kr = np.hypot(*ktot[1:])
    kz = ktot[0]
    kr_max = ni / wl  # the radius of the spherical shell
    kr_0 = na / wl  # the offset of the circle from the origin
    # z displacement of circle's center
    z0 = np.sqrt(kr_max ** 2 - kr_0 ** 2)
    cent_kr = kr - kr_0
    # calculate top half
    # onehalf = np.hypot(cent_kr, kz - z0 - z_offset * dkz) <= kr_max
    onehalf = np.hypot(cent_kr, kz - z0) > kr_max
    # calculate bottom half
    # otherhalf = np.hypot(cent_kr, kz + z0 - z_offset * dkz) <= kr_max
    otherhalf = np.hypot(cent_kr, kz + z0) > kr_max
    mask = np.logical_or(otherhalf, onehalf)
    # mask = # more stuff here
    otf[mask] = 0
    # ifft
    cleaned_psf = np.real(fftshift(ifftn(ifftshift(otf))))
    return cleaned_psf

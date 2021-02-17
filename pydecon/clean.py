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
    from pyfftw.interfaces.numpy_fft import (
        fftshift,
        ifftshift,
        fftn,
        ifftn,
        fftfreq,
    )

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import fftshift, ifftshift, fftn, ifftn, fftfreq


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


def calc_infocus_psf(psf):
    """Calculate the infocus psf using the projection-slice theorem.

    https://en.wikipedia.org/wiki/Projection-slice_theorem"""
    otf = fftshift(fftn(ifftshift(psf))).mean(0)
    infocus_psf = np.real(fftshift(ifftn(ifftshift(otf))))
    return infocus_psf


# TODO: for both cleaning functions there should be an option to limit
# the window size to the realspace mask.
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
    na, ni, wl, res, zres = (
        exp_kwargs["na"],
        exp_kwargs["ni"],
        exp_kwargs["wl"],
        exp_kwargs["res"],
        exp_kwargs["zres"],
    )
    # psf can be approximated by a gaussian with sigma = 0.45 * wl / (2 * na)
    # https://en.wikipedia.org/wiki/Airy_disk
    # filter in x-space by assuming that there is no data at ns sigma
    cleaned_psf = psf.copy()
    xtot, xres = _make_xspace(psf, (zres, res, res))
    r = np.hypot(*xtot[1:])
    z = xtot[0]
    theta = np.arcsin(na / ni)
    # make a double sided cone
    psf_mask = r - abs(z) * np.tan(theta) > ns * 0.45 * wl / (2 * na)
    # zmax = zres * mask.shape[0] / 2
    # width = ns * 0.45 * wl / (2 * na) + zmax * np.tan(theta)
    # window = slice_maker(*(np.array(mask.shape[1:]) // 2), int(2 * width / res))
    # print(window)
    # return mask, window
    bg = np.median(cleaned_psf[psf_mask])
    cleaned_psf = np.fmax(cleaned_psf - bg, 0)
    cleaned_psf[psf_mask] = 0
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
    otf_mask = np.logical_or(otherhalf, onehalf)
    # mask = # more stuff here
    otf[otf_mask] = 0
    # ifft
    cleaned_psf = np.real(ifftshift(ifftn(fftshift(otf))))
    return cleaned_psf


### GRAVEYARD

# class PSFFinder(PeakFinder):
#     """Object to find and analyze subdiffractive emmitters"""

#     def __init__(self, data, psfwidth=1.3, window_width=20):
#         """Analyze a z-stack of subdiffractive emmitters

#         This object will find sub-diffractive emmitters so that they
#         can be cleaned up and analyzed

#         Parameters
#         ----------
#         stack : ndarray

#         Kwargs
#         ------
#         psfwidth : float
#         window_width : int"""
#         super().__init__(data, psfwidth)
#         self.find_blobs()
#         self.window_width = window_width
#         self.find_psfs(2 * psfwidth)

#     def find_psfs(self, max_s=2.1):
#         """Function to find and fit blobs in the max intensity image

#         Blobs with the appropriate parameters are saved for further fitting.

#         Parameters
#         ----------
#         max_s: float
#             Reject all peaks with a fit width greater than this
#         num_peaks: int
#             The number of peaks to analyze further"""
#         window_width = self.window_width
#         # find blobs
#         self.find_blobs()
#         # prune blobs
#         self.remove_edge_blobs(window_width)
#         self.prune_blobs(window_width)
#         # fit blobs in max intensity
#         blobs_df = self.fit_blobs(window_width)
#         # round to make sorting a little more meaningfull
#         blobs_df.SNR = blobs_df.dropna().SNR.round().astype(int)
#         # sort by SNR then sigma_x after filtering for unreasonably
#         # large blobs and reindex data frame here
#         new_blobs_df = (
#             blobs_df[blobs_df.sigma_x < max_s]
#             .sort_values(["SNR", "sigma_x"], ascending=[False, True])
#             .reset_index(drop=True)
#         )
#         # set the internal state to the selected blobs
#         self.blobs = new_blobs_df[["y0", "x0", "sigma_x", "amp"]].values.astype(int)
#         self.fits = new_blobs_df

#     def find_window(self, blob_num=0):
#         """Finds the biggest window distance."""
#         # pull all blobs
#         blobs = self.blobs
#         # three different cases
#         if not len(blobs):
#             # no blobs in window, raise hell
#             raise RuntimeError("No blobs found, can't find window")
#         else:
#             # TODO: this should be refactored to use KDTrees
#             # more than one blob find
#             best = np.round(
#                 self.fits.iloc[blob_num][["y0", "x0", "sigma_x", "amp"]].values
#             ).astype(int)

#             def calc_r(blob1, blob2):
#                 """Calc euclidean distance between blob1 and blob2"""
#                 y1, x1, s1, a1 = blob1
#                 y2, x2, s2, a2 = blob2
#                 return np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)

#             # calc distances
#             r = np.array([calc_r(best, blob) for blob in blobs])
#             # find min distances
#             # remember that best is in blobs so 0 will be in the list
#             # find the next value
#             r.sort()
#             try:
#                 r_min = r[1]
#             except IndexError:
#                 # make r_min the size of the image
#                 r_min = min(np.concatenate((np.array(self.data.shape) - best[:2], best[:2])))
#             # now window size equals sqrt or this
#             win_size = int(round(2 * (r_min / np.sqrt(2) - best[2] * 3)))

#         window = slice_maker(best, win_size)

#         return window

#     def plot_all_windows(self):
#         """Plot all the windows so that user can choose favorite"""
#         windows = [self.find_window(i) for i in range(len(self.fits))]
#         fig, axs = display_grid({i: self.data[win] for i, win in enumerate(windows)})
#         return fig, axs


# class PSFProcMethods(object):
#     """A class specifically designed to mixin to add methods"""

#     def _inspect(self, psf):
#         """"""
#         otf = fftshift(fftn(ifftshift(psf)))
#         mip(psf)
#         slice_plot(abs(otf))
#         slice_plot(np.angle(otf))

#     def inspect_psf(self, blob_num=0):
#         """Inspect the psf assoicated with blob_num"""
#         psf = self.get_psf(blob_num)
#         self._inspect(psf)

#     def inspect_psf_clean(self, blob_num=0):
#         """Inspect the psf assoicated with blob_num"""
#         psf = self.clean_psf(blob_num)
#         self._inspect(psf)


# class PSF2DProcessor(PSFProcMethods, PSFFinder):
#     """An object for processing 2D PSFs and OTFs from 3D stacks"""

#     def __init__(self, stack, wl=585, na=0.85, res=130, **kwargs):
#         """Find PSFs and turn them into OTFs

#         Parameters
#         ----------
#         stack : ndarray
#         na : float
#         pixsize : float
#         det_wl : float
#         """
#         assert stack.ndim == 3, "Stack is expected to be 3D"
#         psfwidth = wl / 4 / na / res
#         # use the max projection for peak finding
#         super().__init__(stack.max(0), psfwidth, **kwargs)
#         self.na = na
#         self.res = res
#         self.wl = wl

#     def get_psf(self, blob_num=0):
#         """make a 2d psf"""
#         psf3d = self.stack[[Ellipsis] + self.find_window(blob_num)]
#         psf3d_corr = center_data(remove_bg(psf3d, 1.0))
#         psf2d = calc_infocus_psf(psf3d_corr)
#         return psf2d

#     def clean_psf(self, blob_num, **kwargs):
#         """"""
#         exp_kwargs = dict(na=self.na, wl=self.wl, res=self.res)
#         psf = psf2dclean(self.get_psf(blob_num), exp_kwargs, **kwargs)
#         return psf


# class PSF3DProcessor(PSFProcMethods, PSFFinder):
#     """An object for processing 2D PSFs and OTFs from 3D stacks"""

#     def __init__(self, stack, wl=585, na=0.85, ni=1.0, res=130, zres=250, **kwargs):
#         """Find PSFs and turn them into OTFs

#         Parameters
#         ----------
#         stack : ndarray
#         na : float
#         pixsize : float
#         det_wl : float
#         """
#         assert stack.ndim == 3, "Stack is expected to be 3D"
#         psfwidth = wl / 4 / na / res
#         # use the max projection for peak finding
#         super().__init__(stack.max(0), psfwidth, **kwargs)
#         self.na = na
#         self.res = res
#         self.wl = wl
#         self.ni = ni
#         self.zres = zres

#     def get_psf(self, blob_num=0):
#         """make a 2d psf"""
#         psf3d = self.stack[[Ellipsis] + self.find_window(blob_num)]
#         psf3d_corr = center_data(remove_bg(psf3d, 1.0))
#         return psf3d_corr

#     def clean_psf(self, blob_num, **kwargs):
#         """"""
#         exp_kwargs = dict(na=self.na, wl=self.wl, res=self.res, ni=self.ni, zres=self.zres)
#         psf = psf3dclean(self.get_psf(blob_num), exp_kwargs, **kwargs)
#         return psf

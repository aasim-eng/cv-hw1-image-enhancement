"""
metrics.py
Quality numbers that compare a result image with the original
(reference) image. Both images must be uint8 and the same size.

  PSNR: higher is better (in dB)
  SSIM: higher is better (1.0 = identical structure)
  MAE : lower is better  (0 = identical)
"""

import numpy as np
from skimage.metrics import structural_similarity


def mean_absolute_error(reference, result):
    """MAE = average of |reference - result| over all pixels and channels.

    We change to float first. With uint8, 10 - 20 would wrap around
    to 246 instead of -10.
    """
    difference = reference.astype(np.float64) - result.astype(np.float64)
    return float(np.mean(np.abs(difference)))


def psnr(reference, result):
    """Peak Signal-to-Noise Ratio in dB.

    MSE  = average of (reference - result)^2
    PSNR = 10 * log10(255^2 / MSE)

    255 is the largest possible pixel value for uint8 images.
    """
    difference = reference.astype(np.float64) - result.astype(np.float64)
    mse = np.mean(difference ** 2)
    if mse == 0:
        return float("inf")  # images are identical
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def ssim(reference, result):
    """Structural Similarity from scikit-image.

    channel_axis=2 tells scikit-image that the color channels are in
    the last axis (height, width, channel). SSIM is computed per
    channel and then averaged. data_range=255 because images are uint8.
    """
    value = structural_similarity(reference, result, channel_axis=2, data_range=255)
    return float(value)

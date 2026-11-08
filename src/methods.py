"""
methods.py
The 4 enhancement methods. Every method takes a BGR uint8 image
and returns a BGR uint8 image of the same size.

Why we work on the L channel of CIE-Lab:
  Lab splits an image into L (lightness) and a, b (color).
  If we change only L, we change brightness/contrast but the
  colors (a, b) stay the same. If we changed R, G, B separately,
  each channel would change by a different amount and the colors
  could shift.

OpenCV note: for uint8 images, cv2.COLOR_BGR2LAB scales L to 0-255
(not 0-100 like in the textbook). So we can treat L like a normal
gray image.
"""

import cv2
import numpy as np


# ---------- helper functions ----------

def split_lab(image_bgr):
    """Change BGR to Lab and return the three channels separately."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    return lightness, channel_a, channel_b


def merge_lab(lightness, channel_a, channel_b):
    """Put the Lab channels back together and change Lab to BGR."""
    lab = cv2.merge([lightness, channel_a, channel_b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------- method 1: baseline ----------

def baseline(image_bgr):
    """No processing. We return a copy of the degraded image so we
    can measure how bad the damage is before any enhancement."""
    return image_bgr.copy()


# ---------- method 2: linear contrast stretching ----------

def contrast_stretch(image_bgr, low_percentile, high_percentile):
    """Linear contrast stretching on LUMINANCE (the L channel of Lab).

    CHOICE: we stretch only L, not each B, G, R channel separately.
    Reason: stretching each channel separately can change the colors.

    We use percentiles instead of the true min and max.
    Reason: the image has noise. One very dark or very bright noisy
    pixel would decide the whole range and the stretch would do almost
    nothing. With percentiles (for example 1% and 99%), we ignore those
    few extreme pixels. Pixels outside the range are clipped to 0 or 255.

    Formula: new_L = (L - low) / (high - low) * 255
    """
    lightness, channel_a, channel_b = split_lab(image_bgr)

    low_value = np.percentile(lightness, low_percentile)
    high_value = np.percentile(lightness, high_percentile)

    # Safety check: if the image is one flat color we cannot stretch it.
    if high_value - low_value < 1:
        return image_bgr.copy()

    # Use float so the division does not lose precision.
    lightness_float = lightness.astype(np.float32)
    stretched = (lightness_float - low_value) / (high_value - low_value) * 255.0
    stretched = np.clip(stretched, 0, 255)
    stretched = np.round(stretched).astype(np.uint8)

    return merge_lab(stretched, channel_a, channel_b)


# ---------- method 3: histogram equalization on L ----------

def equalize_lightness(image_bgr):
    """Histogram equalization on the L channel of Lab.

    Equalization spreads the brightness values so that all levels
    from 0 to 255 are used about equally. Dark, low-contrast images
    become brighter with more contrast. Colors (a, b) are not touched.
    """
    lightness, channel_a, channel_b = split_lab(image_bgr)
    # equalizeHist needs a single-channel uint8 image. L is exactly that.
    equalized = cv2.equalizeHist(lightness)
    return merge_lab(equalized, channel_a, channel_b)


# ---------- method 4: non-local means denoising + equalization ----------

def denoise_then_equalize(image_bgr, nlm_strength, nlm_color_strength,
                          nlm_template_size, nlm_search_size):
    """First remove noise with non-local means (NLM), then equalize L.

    Why this order: histogram equalization also makes the noise
    stronger. If we clean the noise first, the result is less grainy.

    How NLM works (short): for each pixel, it searches for similar
    small patches in the area around it and averages them. Noise is
    random so it averages away, but real edges stay.

    Possible problem: if nlm_strength is too high, small details
    (grass, leaves, text) are also smoothed away.

    fastNlMeansDenoisingColored works on a uint8 BGR image.
    Parameters:
      nlm_strength       = h, filter strength for lightness
      nlm_color_strength = hColor, filter strength for color
      nlm_template_size  = size of the small patch (odd number, e.g. 7)
      nlm_search_size    = size of the search area (odd number, e.g. 21)
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        nlm_strength,
        nlm_color_strength,
        nlm_template_size,
        nlm_search_size,
    )
    return equalize_lightness(denoised)

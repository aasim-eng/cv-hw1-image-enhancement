"""
degrade.py
Functions to load a photo, make it smaller, and make a "damaged" copy
(lower contrast + Gaussian noise).
"""

import cv2
import numpy as np


def load_image(path):
    """Read a color image from disk.

    Note: OpenCV reads images in BGR order (Blue, Green, Red),
    not RGB. We keep BGR inside the whole pipeline and only change
    to RGB when we show images with matplotlib.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Could not read image: " + str(path))
    return image


def resize_longest_side(image, max_side):
    """Make the image smaller so its longest side is max_side pixels.

    Why: phone photos are very big (3024 x 4032). Non-local means
    denoising would be very slow on them. A smaller image is enough
    to compare the methods.
    """
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image.copy()

    scale = max_side / longest_side
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    # INTER_AREA is the recommended OpenCV method for making images smaller.
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def lower_contrast(image, contrast_factor):
    """Pull every pixel value toward middle gray (128).

    new_value = 128 + contrast_factor * (old_value - 128)

    contrast_factor = 1.0 -> no change
    contrast_factor = 0.5 -> half the contrast (dark gets lighter,
                             bright gets darker)

    Returns a float32 image (not yet clipped), because we still add
    noise after this step.
    """
    # Change uint8 to float32 first. With uint8, numbers below 0 or
    # above 255 would "wrap around" (for example 250 + 10 = 4).
    image_float = image.astype(np.float32)
    return 128.0 + contrast_factor * (image_float - 128.0)


def add_gaussian_noise(image_float, noise_sigma, random_seed):
    """Add random Gaussian noise with mean 0 and std = noise_sigma.

    We make a new random generator with the SAME seed for every image.
    So the noise is always the same when we run the program again.
    """
    random_generator = np.random.default_rng(random_seed)
    noise = random_generator.normal(0.0, noise_sigma, size=image_float.shape)
    return image_float + noise.astype(np.float32)


def to_uint8(image_float):
    """Clip values to the range 0-255 and change back to uint8.

    Clipping is needed because after our math some values can be
    below 0 or above 255, and a normal image cannot store them.
    """
    clipped = np.clip(image_float, 0, 255)
    return np.round(clipped).astype(np.uint8)


def degrade_image(image, contrast_factor, noise_sigma, random_seed):
    """Full degradation: lower the contrast, then add noise."""
    low_contrast = lower_contrast(image, contrast_factor)
    noisy = add_gaussian_noise(low_contrast, noise_sigma, random_seed)
    return to_uint8(noisy)

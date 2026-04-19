"""
minicv.filtering.filters — Spatial filters built on the convolution engine.

All linear filters delegate to :func:`minicv.filtering.convolution.spatial_filter`
(and therefore to :func:`convolve2d`) so that kernel validation, boundary
handling, and vectorized computation are centralised in one place.

Public API
----------
mean_filter         : Box / averaging filter.
gaussian_kernel     : Generate an M×M Gaussian kernel.
gaussian_filter     : Gaussian low-pass blur.
median_filter       : Non-linear median filter (controlled loops — see notes).
sobel               : Sobel gradient magnitude and optional direction maps.
sobel_x / sobel_y   : Individual directional Sobel responses.
"""

import numpy as np

from minicv.utils.validation import (
    validate_image_array,
    is_grayscale,
    ensure_uint8,
)
from minicv.utils.padding import pad as _pad
from minicv.filtering.convolution import spatial_filter, _validate_kernel


# =====================================================================
#  1.  Mean / Box Filter
# =====================================================================

def mean_filter(image, ksize=3, *, pad_mode="zero"):
    """Apply an averaging (box) filter.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    ksize : int, optional
        Kernel side length (must be a positive odd integer).  Default 3.
    pad_mode : str, optional
        Boundary handling mode (default ``"zero"``).

    Returns
    -------
    numpy.ndarray
        Filtered image, dtype ``float64``, same spatial shape as input.

    Raises
    ------
    TypeError
        If *ksize* is not an ``int``.
    ValueError
        If *ksize* is even or less than 1.

    Notes
    -----
    The kernel is a ``ksize × ksize`` matrix filled with
    ``1 / ksize²``.  The filter is applied via
    :func:`~minicv.filtering.convolution.spatial_filter`.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> blurred = mean_filter(img, ksize=5)
    >>> blurred.shape
    (100, 100)
    """
    _CALLER = "mean_filter"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    _validate_ksize(ksize, caller=_CALLER)

    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    return spatial_filter(image, kernel, pad_mode=pad_mode)


# =====================================================================
#  2.  Gaussian Filter
# =====================================================================

def gaussian_kernel(ksize, sigma):
    """Generate a 2-D Gaussian kernel.

    Parameters
    ----------
    ksize : int
        Side length of the square kernel (must be positive and odd).
    sigma : float
        Standard deviation of the Gaussian distribution.  Must be > 0.

    Returns
    -------
    numpy.ndarray
        Normalised ``(ksize, ksize)`` kernel with dtype ``float64``
        whose elements sum to 1.0.

    Raises
    ------
    TypeError
        If *ksize* is not ``int`` or *sigma* is not numeric.
    ValueError
        If *ksize* is even / < 1 or *sigma* ≤ 0.

    Notes
    -----
    The kernel is computed analytically from the 2-D Gaussian PDF:

    .. math::

        G(x, y) = \\frac{1}{2\\pi\\sigma^2}
                   \\exp\\!\\left(-\\frac{x^2 + y^2}{2\\sigma^2}\\right)

    where (x, y) are offsets from the kernel centre.  The result is
    normalised so the sum of all elements equals exactly 1.
    """
    _CALLER = "gaussian_kernel"
    _validate_ksize(ksize, caller=_CALLER)

    if not isinstance(sigma, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"[{_CALLER}] 'sigma' must be numeric, got "
            f"{type(sigma).__name__}."
        )
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"[{_CALLER}] 'sigma' must be > 0, got {sigma}.")

    half = ksize // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_filter(image, ksize=3, sigma=1.0, *, pad_mode="zero"):
    """Apply a Gaussian low-pass blur.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    ksize : int, optional
        Kernel side length (positive, odd).  Default 3.
    sigma : float, optional
        Standard deviation of the Gaussian.  Default 1.0.
    pad_mode : str, optional
        Boundary handling mode (default ``"zero"``).

    Returns
    -------
    numpy.ndarray
        Blurred image, dtype ``float64``.

    Raises
    ------
    TypeError / ValueError
        See :func:`gaussian_kernel` and
        :func:`~minicv.filtering.convolution.spatial_filter`.

    Examples
    --------
    >>> blurred = gaussian_filter(img, ksize=5, sigma=1.4)
    """
    _CALLER = "gaussian_filter"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    k = gaussian_kernel(ksize, sigma)
    return spatial_filter(image, k, pad_mode=pad_mode)


# =====================================================================
#  3.  Median Filter  (non-linear — controlled loops, documented)
# =====================================================================

def median_filter(image, ksize=3, *, pad_mode="edge"):
    """Apply a non-linear median filter.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image (uint8 or float).
    ksize : int, optional
        Window side length (positive, odd).  Default 3.
    pad_mode : str, optional
        Boundary mode (default ``"edge"`` — typically best for median).

    Returns
    -------
    numpy.ndarray
        Filtered image with the **same dtype** as input.

    Raises
    ------
    TypeError / ValueError
        On bad *image* or *ksize*.

    Notes
    -----
    **Why loops are necessary:**  The median is a rank-order statistic
    and cannot be expressed as a linear weighted sum of neighbours.
    There is no way to compute it via convolution or ``einsum``.
    However, the implementation minimises Python overhead by:

    1. Extracting **all** overlapping windows at once via
       ``numpy.lib.stride_tricks.as_strided`` (zero-copy 4-D view).
    2. Reshaping the 4-D view to ``(H*W, ksize²)`` and calling
       ``numpy.median`` along ``axis=1`` in a **single vectorized
       call** — so there are **zero explicit Python loops over
       pixels** for grayscale images.
    3. For RGB images a loop over 3 channels is used (3 iterations
       only).

    This is the same strategy NumPy-based median implementations use
    and is dramatically faster than a naïve double for-loop.

    Examples
    --------
    >>> cleaned = median_filter(noisy_img, ksize=5)
    """
    _CALLER = "median_filter"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    _validate_ksize(ksize, caller=_CALLER)

    # ── helper: single-channel median via stride_tricks ──────────────
    def _median_2d(plane):
        H, W = plane.shape
        half = ksize // 2
        padded = _pad(plane, half, half, mode=pad_mode).astype(np.float64)
        s = padded.strides
        windows = np.lib.stride_tricks.as_strided(
            padded,
            shape=(H, W, ksize, ksize),
            strides=(s[0], s[1], s[0], s[1]),
        )
        # Flatten each window and take the median in one vectorized call
        flat = windows.reshape(H, W, -1)       # (H, W, ksize²)
        med = np.median(flat, axis=2)           # (H, W)
        return med

    # ── grayscale ────────────────────────────────────────────────────
    img = image
    squeeze_hw1 = False
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
        squeeze_hw1 = True

    if img.ndim == 2:
        result = _median_2d(img)
        if image.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = result.astype(image.dtype)
        return result

    # ── RGB: per-channel (3 iterations only) ─────────────────────────
    if img.ndim == 3 and img.shape[2] == 3:
        channels = []
        for c in range(3):
            channels.append(_median_2d(img[:, :, c]))
        result = np.stack(channels, axis=-1)
        if image.dtype == np.uint8:
            return np.clip(result, 0, 255).astype(np.uint8)
        return result.astype(image.dtype)

    raise ValueError(
        f"[{_CALLER}] Unsupported shape {img.shape}."
    )


# =====================================================================
#  5.  Sobel Gradients
# =====================================================================

# Pre-defined Sobel 3×3 kernels
_SOBEL_X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float64)

_SOBEL_Y = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=np.float64)


def sobel_x(image, *, pad_mode="zero"):
    """Compute the horizontal Sobel gradient (∂I/∂x approximation).

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    pad_mode : str, optional
        Boundary handling mode (default ``"zero"``).

    Returns
    -------
    numpy.ndarray
        Gradient image, dtype ``float64``.  Values may be negative.

    Notes
    -----
    The 3×3 Sobel-X kernel is::

        [[-1,  0,  1],
         [-2,  0,  2],
         [-1,  0,  1]]
    """
    validate_image_array(image, allow_float=True, caller="sobel_x")
    return spatial_filter(image, _SOBEL_X, pad_mode=pad_mode)


def sobel_y(image, *, pad_mode="zero"):
    """Compute the vertical Sobel gradient (∂I/∂y approximation).

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    pad_mode : str, optional
        Boundary handling mode (default ``"zero"``).

    Returns
    -------
    numpy.ndarray
        Gradient image, dtype ``float64``.  Values may be negative.

    Notes
    -----
    The 3×3 Sobel-Y kernel is::

        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
    """
    validate_image_array(image, allow_float=True, caller="sobel_y")
    return spatial_filter(image, _SOBEL_Y, pad_mode=pad_mode)


def sobel(image, *, pad_mode="zero", return_direction=False):
    """Compute the Sobel gradient magnitude (and optionally direction).

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` image.  For RGB inputs, convert to
        grayscale first via :func:`minicv.io.color.convert_color`.
    pad_mode : str, optional
        Boundary handling mode (default ``"zero"``).
    return_direction : bool, optional
        If ``True``, also return the gradient angle in radians.

    Returns
    -------
    magnitude : numpy.ndarray
        ``(H, W)`` gradient magnitude: ``sqrt(Gx² + Gy²)``, dtype
        ``float64``.
    direction : numpy.ndarray, optional
        ``(H, W)`` gradient angle in radians ``[-π, π]`` via
        ``arctan2(Gy, Gx)``.  Only returned when *return_direction*
        is ``True``.

    Raises
    ------
    ValueError
        If *image* is not 2-D grayscale.

    Notes
    -----
    .. math::

        G = \\sqrt{G_x^2 + G_y^2}, \\qquad
        \\theta = \\operatorname{atan2}(G_y,\\, G_x)
    """
    _CALLER = "sobel"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    if image.ndim != 2:
        raise ValueError(
            f"[{_CALLER}] Expected 2-D grayscale, got shape {image.shape}. "
            f"Convert to grayscale first."
        )

    gx = sobel_x(image, pad_mode=pad_mode)
    gy = sobel_y(image, pad_mode=pad_mode)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    if return_direction:
        direction = np.arctan2(gy, gx)
        return magnitude, direction
    return magnitude


# =====================================================================
#  Internal helpers
# =====================================================================

def _validate_ksize(ksize, *, caller=""):
    """Validate a square kernel size parameter."""
    prefix = f"[{caller}] " if caller else ""
    if not isinstance(ksize, (int, np.integer)):
        raise TypeError(
            f"{prefix}'ksize' must be an int, got {type(ksize).__name__}."
        )
    ksize = int(ksize)
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(
            f"{prefix}'ksize' must be a positive odd integer, got {ksize}."
        )
    return ksize

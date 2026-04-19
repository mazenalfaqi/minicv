"""
minicv.filtering.processing — Image processing techniques.

All functions operate on NumPy arrays and rely exclusively on NumPy,
Matplotlib, and the MiniCV convolution/filter pipeline.  No external
image-processing libraries are used.

Public API
----------
threshold_global      : Fixed-value binary thresholding.
threshold_otsu        : Automatic Otsu thresholding.
threshold_adaptive    : Local adaptive thresholding (mean or gaussian).
bit_plane_slice       : Extract a single bit-plane from a uint8 image.
histogram             : Compute intensity histogram (256 bins).
histogram_equalization: Global histogram equalization (grayscale).
unsharp_mask          : Sharpen via unsharp masking (bonus technique 1).
laplacian             : Laplacian edge detector (bonus technique 2).
"""

import numpy as np

from minicv.utils.validation import (
    validate_image_array,
    is_grayscale,
    ensure_uint8,
)
from minicv.filtering.convolution import spatial_filter, convolve2d
from minicv.filtering.filters import gaussian_kernel, gaussian_filter


# =====================================================================
#  4.  Thresholding
# =====================================================================

# ── 4a. Global Thresholding ──────────────────────────────────────────

def threshold_global(image, thresh, maxval=255):
    """Apply fixed (global) binary thresholding.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image ``(H, W)`` — uint8 or float.
    thresh : int or float
        Threshold value.  Pixels **> thresh** are set to *maxval*;
        others are set to 0.
    maxval : int or float, optional
        Value assigned to pixels above the threshold.  Default 255.

    Returns
    -------
    numpy.ndarray
        Binary image with the same shape and dtype ``uint8``.

    Raises
    ------
    TypeError
        If *image* is not a ``numpy.ndarray`` or *thresh* is not numeric.
    ValueError
        If *image* is not 2-D.

    Examples
    --------
    >>> binary = threshold_global(gray, thresh=127)
    """
    _CALLER = "threshold_global"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    _require_2d(image, _CALLER)

    if not isinstance(thresh, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"[{_CALLER}] 'thresh' must be numeric, got "
            f"{type(thresh).__name__}."
        )

    img = image.astype(np.float64)
    out = np.where(img > float(thresh), float(maxval), 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── 4b. Otsu Thresholding ───────────────────────────────────────────

def threshold_otsu(image, maxval=255):
    """Automatic binary thresholding using Otsu's method.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` image, dtype ``uint8``.
    maxval : int, optional
        Foreground value (default 255).

    Returns
    -------
    binary : numpy.ndarray
        Binary image (uint8).
    optimal_thresh : int
        The threshold value selected by Otsu's algorithm.

    Raises
    ------
    TypeError
        If *image* is not uint8.
    ValueError
        If *image* is not 2-D.

    Notes
    -----
    **Otsu's algorithm** exhaustively searches all 256 possible
    thresholds and selects the one that maximises the **between-class
    variance** σ²_B:

    .. math::

        \\sigma_B^2(t) = \\omega_0(t)\\,\\omega_1(t)
                         \\bigl[\\mu_0(t) - \\mu_1(t)\\bigr]^2

    where ω₀, ω₁ are the class probabilities and μ₀, μ₁ the class
    means for threshold *t*.

    The implementation is fully vectorized over all 256 thresholds
    using cumulative sums — no Python loop over thresholds.

    Examples
    --------
    >>> binary, t = threshold_otsu(gray)
    >>> print(f"Otsu threshold = {t}")
    """
    _CALLER = "threshold_otsu"
    validate_image_array(image, allow_float=False, caller=_CALLER)
    _require_2d(image, _CALLER)

    if image.dtype != np.uint8:
        raise TypeError(
            f"[{_CALLER}] Otsu requires uint8 image, got {image.dtype}."
        )

    # ── histogram (256 bins) ─────────────────────────────────────────
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    total = image.size

    # ── vectorized Otsu over all thresholds 0..255 ───────────────────
    # Cumulative quantities
    weight_bg = np.cumsum(hist)                              # ω₀(t)
    weight_fg = total - weight_bg                            # ω₁(t)

    intensity = np.arange(256, dtype=np.float64)
    cum_mean = np.cumsum(hist * intensity)                   # Σ i·h(i)
    global_mean = cum_mean[-1]

    mean_bg = np.divide(
        cum_mean, weight_bg,
        out=np.zeros(256), where=weight_bg > 0,
    )
    mean_fg = np.divide(
        global_mean - cum_mean, weight_fg,
        out=np.zeros(256), where=weight_fg > 0,
    )

    # Between-class variance
    sigma_b_sq = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    optimal_thresh = int(np.argmax(sigma_b_sq))
    binary = threshold_global(image, optimal_thresh, maxval=maxval)
    return binary, optimal_thresh


# ── 4c. Adaptive Thresholding ───────────────────────────────────────

def threshold_adaptive(image, block_size, C=0, *, method="mean",
                       maxval=255, pad_mode="edge"):
    """Local adaptive binary thresholding.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` image (uint8 or float).
    block_size : int
        Side length of the local neighbourhood (must be positive and
        odd).
    C : int or float, optional
        Constant subtracted from the local mean/weighted-mean before
        comparison.  Default 0.
    method : str, optional
        ``"mean"`` — threshold is the local box-filter mean minus *C*.
        ``"gaussian"`` — threshold is the local Gaussian-weighted mean
        minus *C* (sigma is auto-set to ``0.3 * ((block_size - 1)/2 - 1) + 0.8``
        following OpenCV's convention).
        Default ``"mean"``.
    maxval : int or float, optional
        Value for foreground pixels (default 255).
    pad_mode : str, optional
        Boundary handling mode (default ``"edge"``).

    Returns
    -------
    numpy.ndarray
        Binary image (uint8).

    Raises
    ------
    TypeError / ValueError
        On bad inputs.

    Notes
    -----
    For each pixel the local threshold *T(x, y)* is computed from its
    ``block_size × block_size`` neighbourhood:

    * ``method="mean"``  →  ``T = local_mean − C``
    * ``method="gaussian"``  →  ``T = local_gaussian_mean − C``

    Pixels where ``I(x,y) > T(x,y)`` are set to *maxval*, else 0.

    Examples
    --------
    >>> ada = threshold_adaptive(gray, block_size=11, C=2, method="gaussian")
    """
    _CALLER = "threshold_adaptive"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    _require_2d(image, _CALLER)

    if not isinstance(block_size, (int, np.integer)):
        raise TypeError(
            f"[{_CALLER}] 'block_size' must be int, "
            f"got {type(block_size).__name__}."
        )
    block_size = int(block_size)
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(
            f"[{_CALLER}] 'block_size' must be odd and >= 3, "
            f"got {block_size}."
        )

    if not isinstance(method, str):
        raise TypeError(
            f"[{_CALLER}] 'method' must be str, got {type(method).__name__}."
        )
    method = method.strip().lower()
    if method not in ("mean", "gaussian"):
        raise ValueError(
            f"[{_CALLER}] method must be 'mean' or 'gaussian', "
            f"got '{method}'."
        )

    img = image.astype(np.float64)

    # ── compute local threshold surface ──────────────────────────────
    if method == "mean":
        kernel = np.ones((block_size, block_size), dtype=np.float64)
        kernel /= kernel.sum()
        local_mean = convolve2d(img, kernel, pad_mode=pad_mode)
    else:
        # gaussian — auto sigma following OpenCV convention
        sigma = 0.3 * ((block_size - 1) * 0.5 - 1) + 0.8
        k = gaussian_kernel(block_size, sigma)
        local_mean = convolve2d(img, k, pad_mode=pad_mode)

    threshold_surface = local_mean - float(C)

    out = np.where(img > threshold_surface, float(maxval), 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


# =====================================================================
#  6.  Bit-plane Slicing
# =====================================================================

def bit_plane_slice(image, bit):
    """Extract a single bit-plane from a uint8 grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` image with dtype ``uint8``.
    bit : int
        Bit index to extract, 0 (LSB) to 7 (MSB).

    Returns
    -------
    numpy.ndarray
        Binary ``(H, W)`` image (uint8) containing only 0 and 255.

    Raises
    ------
    TypeError
        If *image* dtype is not uint8 or *bit* is not int.
    ValueError
        If *bit* is not in ``[0, 7]`` or *image* is not 2-D.

    Notes
    -----
    Bit-plane *k* is computed as::

        plane = (image >> k) & 1

    and then scaled to ``{0, 255}`` for display.  Higher bit-planes
    (6, 7) carry the most visual information; lower planes resemble
    noise.

    Examples
    --------
    >>> msb = bit_plane_slice(gray, bit=7)
    >>> np.unique(msb)
    array([  0, 255], dtype=uint8)
    """
    _CALLER = "bit_plane_slice"
    validate_image_array(image, allow_float=False, caller=_CALLER)
    _require_2d(image, _CALLER)

    if image.dtype != np.uint8:
        raise TypeError(
            f"[{_CALLER}] Image must be uint8, got {image.dtype}."
        )
    if not isinstance(bit, (int, np.integer)):
        raise TypeError(
            f"[{_CALLER}] 'bit' must be an int, got {type(bit).__name__}."
        )
    bit = int(bit)
    if bit < 0 or bit > 7:
        raise ValueError(
            f"[{_CALLER}] 'bit' must be in [0, 7], got {bit}."
        )

    plane = ((image >> bit) & 1).astype(np.uint8) * 255
    return plane


# =====================================================================
#  7.  Histogram & Histogram Equalization
# =====================================================================

def histogram(image):
    """Compute the intensity histogram of a grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` image, dtype ``uint8``.

    Returns
    -------
    hist : numpy.ndarray
        1-D array of shape ``(256,)`` with dtype ``int64``.
        ``hist[i]`` is the count of pixels with intensity *i*.

    Raises
    ------
    TypeError
        If *image* is not uint8.
    ValueError
        If *image* is not 2-D.

    Examples
    --------
    >>> h = histogram(gray)
    >>> h.shape
    (256,)
    >>> int(h.sum()) == gray.size
    True
    """
    _CALLER = "histogram"
    validate_image_array(image, allow_float=False, caller=_CALLER)
    _require_2d(image, _CALLER)

    if image.dtype != np.uint8:
        raise TypeError(
            f"[{_CALLER}] Image must be uint8, got {image.dtype}."
        )

    return np.bincount(image.ravel(), minlength=256).astype(np.int64)


def histogram_equalization(image):
    """Apply global histogram equalization to a grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` image, dtype ``uint8``.

    Returns
    -------
    equalized : numpy.ndarray
        Equalized ``(H, W)`` image, dtype ``uint8``.
    hist_before : numpy.ndarray
        Histogram of the original image ``(256,)``.
    hist_after : numpy.ndarray
        Histogram of the equalized image ``(256,)``.

    Raises
    ------
    TypeError / ValueError
        On non-uint8 or non-2-D input.

    Notes
    -----
    **Algorithm:**

    1. Compute the histogram ``h[i]`` and the CDF:
       ``cdf[i] = Σ_{j=0}^{i} h[j]``.
    2. Normalise the CDF to [0, 255]:
       ``lut[i] = round(255 · (cdf[i] − cdf_min) / (N − cdf_min))``
       where *N* is the total pixel count and *cdf_min* is the first
       non-zero CDF value.
    3. Map every pixel through the look-up table.

    The entire operation is vectorized via ``numpy`` fancy indexing.

    Examples
    --------
    >>> eq, h_before, h_after = histogram_equalization(low_contrast)
    """
    _CALLER = "histogram_equalization"
    validate_image_array(image, allow_float=False, caller=_CALLER)
    _require_2d(image, _CALLER)

    if image.dtype != np.uint8:
        raise TypeError(
            f"[{_CALLER}] Image must be uint8, got {image.dtype}."
        )

    hist_before = histogram(image)
    cdf = hist_before.astype(np.float64).cumsum()

    cdf_min = cdf[cdf > 0].min()
    N = float(image.size)

    # Build look-up table
    lut = np.round(255.0 * (cdf - cdf_min) / (N - cdf_min))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    equalized = lut[image]   # vectorized fancy-index mapping
    hist_after = histogram(equalized)

    return equalized, hist_before, hist_after


# =====================================================================
#  8.  Two Additional / Bonus Techniques
# =====================================================================

# ── 8a. Unsharp Masking (Sharpening) ────────────────────────────────

def unsharp_mask(image, ksize=5, sigma=1.0, amount=1.5, *, pad_mode="edge"):
    """Sharpen an image using unsharp masking.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    ksize : int, optional
        Gaussian kernel size (positive, odd).  Default 5.
    sigma : float, optional
        Gaussian sigma.  Default 1.0.
    amount : float, optional
        Sharpening strength (1.0 = moderate, 2.0 = strong).
        Default 1.5.
    pad_mode : str, optional
        Boundary handling mode (default ``"edge"``).

    Returns
    -------
    numpy.ndarray
        Sharpened image, dtype ``uint8``, clipped to [0, 255].

    Notes
    -----
    **Unsharp masking** is a classic darkroom-era technique:

    1. Blur the image to produce a "mask":  ``B = Gaussian(I)``.
    2. Compute the detail (high-frequency) layer:  ``D = I − B``.
    3. Add amplified detail back:  ``S = I + amount · D``.

    This is equivalent to:

    .. math::

        S = (1 + \\alpha)\\,I - \\alpha\\,B

    where α = *amount*.

    Examples
    --------
    >>> sharp = unsharp_mask(img, ksize=5, sigma=1.0, amount=1.5)
    """
    _CALLER = "unsharp_mask"
    validate_image_array(image, allow_float=True, caller=_CALLER)

    if not isinstance(amount, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"[{_CALLER}] 'amount' must be numeric, got "
            f"{type(amount).__name__}."
        )
    amount = float(amount)

    blurred = gaussian_filter(image, ksize=ksize, sigma=sigma,
                              pad_mode=pad_mode)
    img_f = image.astype(np.float64)
    sharpened = img_f + amount * (img_f - blurred)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ── 8b. Laplacian Edge Detector ─────────────────────────────────────

# Standard 3×3 Laplacian kernels
_LAPLACIAN_4 = np.array([[ 0, -1,  0],
                          [-1,  4, -1],
                          [ 0, -1,  0]], dtype=np.float64)

_LAPLACIAN_8 = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float64)


def laplacian(image, *, variant="4-connected", pad_mode="zero"):
    """Compute the Laplacian (second-order derivative) of an image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    variant : str, optional
        ``"4-connected"`` — uses the 4-neighbour Laplacian kernel
        ``[[0,-1,0],[-1,4,-1],[0,-1,0]]``.
        ``"8-connected"`` — uses the 8-neighbour kernel
        ``[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]``.
        Default ``"4-connected"``.
    pad_mode : str, optional
        Boundary handling mode (default ``"zero"``).

    Returns
    -------
    numpy.ndarray
        Laplacian response, dtype ``float64``.  Values may be
        negative; take ``np.abs(result)`` for edge magnitude.

    Raises
    ------
    ValueError
        If *variant* is unrecognised.

    Notes
    -----
    The Laplacian is a rotationally symmetric, second-order
    derivative operator:

    .. math::

        \\nabla^2 I = \\frac{\\partial^2 I}{\\partial x^2}
                    + \\frac{\\partial^2 I}{\\partial y^2}

    It detects edges and regions of rapid intensity change.  Unlike
    Sobel, it responds equally to edges in all orientations but is
    more sensitive to noise.

    Examples
    --------
    >>> edges = laplacian(gray, variant="8-connected")
    >>> display = np.clip(np.abs(edges), 0, 255).astype(np.uint8)
    """
    _CALLER = "laplacian"
    validate_image_array(image, allow_float=True, caller=_CALLER)

    if not isinstance(variant, str):
        raise TypeError(
            f"[{_CALLER}] 'variant' must be str, got "
            f"{type(variant).__name__}."
        )
    variant = variant.strip().lower()

    kernels = {
        "4-connected": _LAPLACIAN_4,
        "4": _LAPLACIAN_4,
        "8-connected": _LAPLACIAN_8,
        "8": _LAPLACIAN_8,
    }
    if variant not in kernels:
        raise ValueError(
            f"[{_CALLER}] Unknown variant '{variant}'. "
            f"Use '4-connected' or '8-connected'."
        )

    return spatial_filter(image, kernels[variant], pad_mode=pad_mode)


# =====================================================================
#  Internal helpers
# =====================================================================

def _require_2d(image, caller):
    """Raise if image is not 2-D (H, W)."""
    if image.ndim != 2:
        raise ValueError(
            f"[{caller}] Expected a 2-D grayscale image (H, W), "
            f"got shape {image.shape}."
        )

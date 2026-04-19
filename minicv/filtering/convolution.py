"""
minicv.filtering.convolution — True 2-D convolution and spatial filtering.

Public API
----------
convolve2d      : True 2-D convolution on a **single-channel** (H, W) image.
spatial_filter  : Convenience wrapper — applies a kernel to grayscale
                  images directly and to RGB images per-channel.

Implementation notes
--------------------
* The kernel is **flipped** (rotated 180°) before the sliding-window
  multiply-accumulate pass, making this a true mathematical convolution
  rather than a correlation.
* Boundary pixels are handled by calling :func:`minicv.utils.padding.pad`,
  so the padding strategy is configurable from a single place.
* The inner loop is fully vectorized via :func:`numpy.lib.stride_tricks.as_strided`
  to extract all overlapping windows in one shot, then a single
  ``einsum`` call performs the weighted sum — **no Python loops over
  pixels**.
"""

import numpy as np

from minicv.utils.validation import validate_image_array, is_grayscale
from minicv.utils.padding import pad as _pad


# =====================================================================
#  Kernel validation (shared by all filtering functions)
# =====================================================================

def _validate_kernel(kernel, *, caller=""):
    """Check that *kernel* is a valid 2-D convolution kernel.

    Parameters
    ----------
    kernel : numpy.ndarray
        Candidate kernel array.
    caller : str, optional
        Calling function name for error messages.

    Returns
    -------
    numpy.ndarray
        The kernel cast to ``float64``.

    Raises
    ------
    TypeError
        If *kernel* is not a ``numpy.ndarray`` or has a non-numeric dtype.
    ValueError
        If *kernel* is empty, not 2-D, or has even dimensions.
    """
    prefix = f"[{caller}] " if caller else ""

    if not isinstance(kernel, np.ndarray):
        raise TypeError(
            f"{prefix}'kernel' must be a numpy.ndarray, "
            f"got {type(kernel).__name__}."
        )

    if kernel.ndim != 2:
        raise ValueError(
            f"{prefix}Kernel must be 2-D, got {kernel.ndim}-D "
            f"(shape {kernel.shape})."
        )

    if kernel.size == 0:
        raise ValueError(f"{prefix}Kernel is empty (shape {kernel.shape}).")

    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(
            f"{prefix}Kernel dimensions must be odd; "
            f"got {kh}×{kw}."
        )

    if not np.issubdtype(kernel.dtype, np.number):
        raise TypeError(
            f"{prefix}Kernel dtype must be numeric, "
            f"got '{kernel.dtype}'."
        )

    return kernel.astype(np.float64)


# =====================================================================
#  True 2-D Convolution  (single-channel only)
# =====================================================================

def convolve2d(image, kernel, *, pad_mode="zero"):
    """Apply true 2-D convolution to a single-channel grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape ``(H, W)`` with dtype ``uint8`` or
        ``float``.
    kernel : numpy.ndarray
        2-D convolution kernel with **odd** height and width
        (e.g. 3×3, 5×5).  Will be cast to ``float64`` internally.
    pad_mode : str, optional
        Boundary handling strategy passed to
        :func:`minicv.utils.padding.pad`.  One of ``"zero"`` (default),
        ``"edge"``, ``"reflect"``, or ``"wrap"``.

    Returns
    -------
    numpy.ndarray
        Convolved image of shape ``(H, W)`` with dtype ``float64``.
        The caller is responsible for any subsequent clipping or dtype
        conversion.

    Raises
    ------
    TypeError
        If *image* or *kernel* are not ``numpy.ndarray``.
    ValueError
        If *image* is not 2-D, or the kernel fails validation (even
        size, empty, non-numeric).

    Notes
    -----
    **Mathematical definition:**

    .. math::

        (f * g)[i, j] = \\sum_{m} \\sum_{n}
            f[i - m,\\; j - n] \\cdot g[m,\\; n]

    which is equivalent to flipping the kernel 180° and then computing
    the cross-correlation.

    **Performance:**  All ``H × W`` overlapping windows are extracted
    simultaneously via ``numpy.lib.stride_tricks.as_strided`` and the
    weighted sum is computed with a single ``numpy.einsum`` call, so
    there are **zero** Python-level pixel loops.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.zeros((5, 5), dtype=np.float64)
    >>> img[2, 2] = 1.0                       # impulse
    >>> k = np.array([[0, -1, 0],
    ...               [-1, 5, -1],
    ...               [0, -1, 0]], dtype=np.float64)
    >>> out = convolve2d(img, k)
    >>> float(out[2, 2])
    5.0
    """
    _CALLER = "convolve2d"

    # ── validate inputs ──────────────────────────────────────────────
    validate_image_array(image, allow_float=True, caller=_CALLER)

    if image.ndim != 2:
        raise ValueError(
            f"[{_CALLER}] Expected a 2-D grayscale image, "
            f"got shape {image.shape}.  For RGB images use "
            f"spatial_filter() instead."
        )

    kernel = _validate_kernel(kernel, caller=_CALLER)

    # ── flip kernel (true convolution = correlation with flipped k) ──
    kernel_flipped = kernel[::-1, ::-1]

    kh, kw = kernel_flipped.shape
    ph, pw = kh // 2, kw // 2          # half-sizes for same-size output

    # ── pad the image ────────────────────────────────────────────────
    padded = _pad(image, ph, pw, mode=pad_mode).astype(np.float64)

    # ── extract all windows via stride_tricks ────────────────────────
    H, W = image.shape
    out_h, out_w = H, W                # "same" convolution

    # shape of the view: (out_h, out_w, kh, kw)
    s = padded.strides                  # (stride_row, stride_col)
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(out_h, out_w, kh, kw),
        strides=(s[0], s[1], s[0], s[1]),
    )

    # ── weighted sum via einsum (zero Python loops) ──────────────────
    result = np.einsum("ijmn,mn->ij", windows, kernel_flipped)

    return result


# =====================================================================
#  Spatial Filter  (grayscale + RGB convenience wrapper)
# =====================================================================

def spatial_filter(image, kernel, *, pad_mode="zero"):
    """Apply a spatial convolution filter to a grayscale **or** RGB image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.  Accepted shapes:

        * ``(H, W)``      — grayscale (passed directly to
          :func:`convolve2d`).
        * ``(H, W, 1)``   — grayscale (squeezed, then convolved).
        * ``(H, W, 3)``   — RGB  (convolved **per-channel**).

    kernel : numpy.ndarray
        2-D convolution kernel with odd dimensions.
    pad_mode : str, optional
        Boundary handling mode (default ``"zero"``).

    Returns
    -------
    numpy.ndarray
        Filtered image with dtype ``float64`` and the **same spatial
        shape** as the input:

        * Grayscale input ``(H, W)`` → output ``(H, W)``.
        * RGB input ``(H, W, 3)`` → output ``(H, W, 3)``.

    Raises
    ------
    TypeError
        If *image* or *kernel* are not ``numpy.ndarray``.
    ValueError
        If *image* has an unsupported channel count (e.g. 4) or the
        kernel is invalid.

    Notes
    -----
    **RGB strategy:**  Each channel (R, G, B) is convolved
    independently with the same kernel and the results are stacked
    back into ``(H, W, 3)``.  This is the standard approach used by
    OpenCV for linear filters on colour images.

    Examples
    --------
    >>> import numpy as np
    >>> rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> box3 = np.ones((3, 3), dtype=np.float64) / 9.0
    >>> blurred = spatial_filter(rgb, box3)
    >>> blurred.shape
    (100, 100, 3)
    """
    _CALLER = "spatial_filter"

    validate_image_array(image, allow_float=True, caller=_CALLER)
    kernel = _validate_kernel(kernel, caller=_CALLER)

    # ── squeeze (H, W, 1) ───────────────────────────────────────────
    img = image
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # ── grayscale path ───────────────────────────────────────────────
    if img.ndim == 2:
        return convolve2d(img, kernel, pad_mode=pad_mode)

    # ── RGB path — per-channel convolution ───────────────────────────
    if img.ndim == 3 and img.shape[2] == 3:
        channels = []
        for c in range(3):
            conv_c = convolve2d(img[:, :, c], kernel, pad_mode=pad_mode)
            channels.append(conv_c)
        return np.stack(channels, axis=-1)

    # ── unsupported channel count ────────────────────────────────────
    raise ValueError(
        f"[{_CALLER}] Unsupported image shape {img.shape}. "
        f"Expected (H, W), (H, W, 1), or (H, W, 3)."
    )

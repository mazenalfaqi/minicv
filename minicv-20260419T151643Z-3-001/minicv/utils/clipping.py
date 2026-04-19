"""
minicv.utils.clipping — Pixel value clipping.

Restricts image pixel values to a user-defined range, handling both
uint8 and float images transparently.
"""

import numpy as np

from minicv.utils.validation import validate_image_array


def clip_image(image, low=None, high=None):
    """Clip pixel values to the interval ``[low, high]``.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2-D or 3-D).  Both ``uint8`` and float dtypes are
        accepted.
    low : int or float or None, optional
        Lower bound.  Pixels below this value are set to *low*.
        If ``None``, the natural dtype minimum is used (``0`` for uint8,
        ``0.0`` for float).
    high : int or float or None, optional
        Upper bound.  Pixels above this value are set to *high*.
        If ``None``, the natural dtype maximum is used (``255`` for
        uint8, ``1.0`` for float).

    Returns
    -------
    numpy.ndarray
        Clipped image with the **same dtype** as the input.

    Raises
    ------
    TypeError
        If *image* is not a ``numpy.ndarray``, or *low* / *high* are
        not numeric.
    ValueError
        If ``low > high`` after resolving defaults.

    Notes
    -----
    * The function delegates to :func:`numpy.clip` for performance.
    * For uint8 images, default bounds are ``[0, 255]``.
    * For float images, default bounds are ``[0.0, 1.0]``.  If you
      are working with z-score-normalized data that may be negative,
      pass explicit bounds such as ``low=-3.0, high=3.0``.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[0, 128, 300]], dtype=np.float64)
    >>> clip_image(img, low=0, high=255)
    array([[  0., 128., 255.]])

    >>> u8 = np.array([[10, 200, 250]], dtype=np.uint8)
    >>> clip_image(u8, low=50, high=200)
    array([[ 50, 200, 200]], dtype=uint8)
    """
    _CALLER = "clip_image"

    validate_image_array(image, allow_float=True, caller=_CALLER)

    # ── resolve defaults ─────────────────────────────────────────────
    if image.dtype == np.uint8:
        default_lo, default_hi = 0, 255
    else:
        default_lo, default_hi = 0.0, 1.0

    low = default_lo if low is None else low
    high = default_hi if high is None else high

    # ── validate bounds ──────────────────────────────────────────────
    if not isinstance(low, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"[{_CALLER}] 'low' must be numeric, got {type(low).__name__}."
        )
    if not isinstance(high, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"[{_CALLER}] 'high' must be numeric, got {type(high).__name__}."
        )

    low, high = float(low), float(high)
    if low > high:
        raise ValueError(
            f"[{_CALLER}] 'low' ({low}) must be <= 'high' ({high})."
        )

    # ── clip (preserves dtype) ───────────────────────────────────────
    return np.clip(image, low, high)

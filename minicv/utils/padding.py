"""
minicv.utils.padding — Image boundary padding.

Four modes are provided: **zero**, **edge** (replicate), **reflect**,
and **wrap** (circular).  All modes delegate to :func:`numpy.pad` for
performance but add image-specific validation and a unified API that
the convolution engine relies on.
"""

import numpy as np

from minicv.utils.validation import validate_image_array


# ── public constant ──────────────────────────────────────────────────────
PADDING_MODES = ("zero", "edge", "reflect", "wrap")


def pad(image, pad_h, pad_w=None, *, mode="zero"):
    """Pad an image along its spatial dimensions.

    Parameters
    ----------
    image : numpy.ndarray
        Input image — ``(H, W)`` or ``(H, W, C)``.
    pad_h : int
        Number of rows to add on **both** the top and bottom.
    pad_w : int or None, optional
        Number of columns to add on **both** the left and right.
        If ``None``, defaults to *pad_h* (symmetric padding).
    mode : str, optional
        Padding strategy.  One of:

        ``"zero"``
            Pad with zeros (0 for uint8, 0.0 for float).
        ``"edge"``
            Replicate the border pixel values outward (nearest-neighbour
            extrapolation).
        ``"reflect"``
            Mirror-reflect at the boundaries **excluding** the border
            pixel itself (``numpy.pad`` mode ``"reflect"``).
        ``"wrap"``
            Circular / periodic wrap-around (``numpy.pad`` mode
            ``"wrap"``).

        Default is ``"zero"``.

    Returns
    -------
    numpy.ndarray
        Padded image with shape ``(H + 2*pad_h, W + 2*pad_w [, C])``.
        dtype is preserved.

    Raises
    ------
    TypeError
        If *image* is not a ``numpy.ndarray``, or *pad_h* / *pad_w* are
        not integers.
    ValueError
        If *pad_h* or *pad_w* are negative, if *mode* is unrecognised,
        or if a reflect pad exceeds the spatial dimension (NumPy
        constraint).

    Notes
    -----
    * The channel axis (if present) is **never** padded.
    * ``"reflect"`` mode requires ``pad_h < H`` and ``pad_w < W``
      because NumPy cannot reflect more rows/columns than exist.
    * This function is the **sole** boundary handler used by the
      convolution engine — all filtering functions call ``pad`` rather
      than implementing their own boundary logic.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.ones((3, 3), dtype=np.uint8) * 128
    >>> padded = pad(img, 1, mode="zero")
    >>> padded.shape
    (5, 5)
    >>> int(padded[0, 0])
    0

    >>> padded_e = pad(img, 1, mode="edge")
    >>> int(padded_e[0, 0])
    128
    """
    _CALLER = "pad"

    # ── validate image ───────────────────────────────────────────────
    validate_image_array(image, allow_float=True, caller=_CALLER)

    # ── validate pad sizes ───────────────────────────────────────────
    if not isinstance(pad_h, (int, np.integer)):
        raise TypeError(
            f"[{_CALLER}] 'pad_h' must be an int, got {type(pad_h).__name__}."
        )
    if pad_w is None:
        pad_w = pad_h
    if not isinstance(pad_w, (int, np.integer)):
        raise TypeError(
            f"[{_CALLER}] 'pad_w' must be an int, got {type(pad_w).__name__}."
        )

    pad_h, pad_w = int(pad_h), int(pad_w)
    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"[{_CALLER}] Pad sizes must be >= 0; got pad_h={pad_h}, "
            f"pad_w={pad_w}."
        )
    if pad_h == 0 and pad_w == 0:
        return image.copy()

    # ── validate mode ────────────────────────────────────────────────
    if not isinstance(mode, str):
        raise TypeError(
            f"[{_CALLER}] 'mode' must be a str, got {type(mode).__name__}."
        )
    mode = mode.strip().lower()
    if mode not in PADDING_MODES:
        raise ValueError(
            f"[{_CALLER}] Unknown mode '{mode}'. Supported: {PADDING_MODES}."
        )

    # reflect constraint check
    H, W = image.shape[0], image.shape[1]
    if mode == "reflect":
        if pad_h >= H:
            raise ValueError(
                f"[{_CALLER}] 'reflect' requires pad_h ({pad_h}) < image "
                f"height ({H})."
            )
        if pad_w >= W:
            raise ValueError(
                f"[{_CALLER}] 'reflect' requires pad_w ({pad_w}) < image "
                f"width ({W})."
            )

    # ── build np.pad width tuple ─────────────────────────────────────
    # 2-D: ((top, bottom), (left, right))
    # 3-D: ((top, bottom), (left, right), (0, 0))  — no channel pad
    spatial_pad = ((pad_h, pad_h), (pad_w, pad_w))
    if image.ndim == 3:
        pad_width = spatial_pad + ((0, 0),)
    else:
        pad_width = spatial_pad

    # ── map mode string to numpy mode ────────────────────────────────
    _MODE_MAP = {
        "zero": "constant",
        "edge": "edge",
        "reflect": "reflect",
        "wrap": "wrap",
    }
    np_mode = _MODE_MAP[mode]

    kwargs = {}
    if mode == "zero":
        kwargs["constant_values"] = 0

    return np.pad(image, pad_width, mode=np_mode, **kwargs)

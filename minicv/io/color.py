"""
minicv.io.color — Color-space conversion utilities.

All conversions are fully vectorized via NumPy — no pixel loops.
"""

import numpy as np

from minicv.utils.validation import validate_image_array, is_grayscale, is_rgb


def convert_color(image, mode):
    """Convert an image between RGB and Grayscale color spaces.

    Parameters
    ----------
    image : numpy.ndarray
        Source image.  Accepted shapes / dtypes:

        * ``(H, W)`` or ``(H, W, 1)`` uint8/float — grayscale.
        * ``(H, W, 3)`` uint8/float — RGB.

    mode : str
        Target color space.  Must be one of:

        * ``"rgb2gray"`` — convert a 3-channel RGB image to 2-D
          grayscale using the ITU-R BT.601 luminance formula:
          ``Y = 0.299·R + 0.587·G + 0.114·B``.
        * ``"gray2rgb"`` — replicate a single-channel grayscale image
          across three channels to produce ``(H, W, 3)``.

    Returns
    -------
    numpy.ndarray
        Converted image with the **same dtype** as the input:

        * ``"rgb2gray"`` → ``(H, W)``
        * ``"gray2rgb"`` → ``(H, W, 3)``

    Raises
    ------
    TypeError
        If *image* is not a ``numpy.ndarray`` or *mode* is not a ``str``.
    ValueError
        If *mode* is unrecognised, or the source image shape does not
        match the requested conversion direction (e.g. trying
        ``"rgb2gray"`` on an already-grayscale image).

    Notes
    -----
    * The BT.601 weights are the same ones used by OpenCV's
      ``cv2.cvtColor(..., cv2.COLOR_RGB2GRAY)``.
    * For uint8 images the output is clipped to [0, 255] after the
      weighted sum.  For float images the output remains in [0.0, 1.0].
    * ``(H, W, 1)`` inputs are squeezed to ``(H, W)`` before any
      conversion so the output shape is always exactly 2-D or 3-D with
      3 channels — never ``(H, W, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> gray = convert_color(rgb, "rgb2gray")
    >>> gray.shape
    (100, 100)
    >>> gray.dtype
    dtype('uint8')

    >>> back = convert_color(gray, "gray2rgb")
    >>> back.shape
    (100, 100, 3)
    """
    _CALLER = "convert_color"

    # ---- validate image --------------------------------------------------
    validate_image_array(image, allow_float=True, caller=_CALLER)

    if not isinstance(mode, str):
        raise TypeError(
            f"[{_CALLER}] 'mode' must be a str, got {type(mode).__name__}."
        )

    mode = mode.strip().lower()
    _VALID_MODES = ("rgb2gray", "gray2rgb")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"[{_CALLER}] Unknown mode '{mode}'. "
            f"Supported modes: {_VALID_MODES}."
        )

    # Squeeze (H, W, 1) → (H, W) for uniform handling
    img = image
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # ==================================================================
    # RGB → Grayscale
    # ==================================================================
    if mode == "rgb2gray":
        if not is_rgb(img):
            raise ValueError(
                f"[{_CALLER}] 'rgb2gray' requires a (H, W, 3) RGB image, "
                f"got shape {img.shape}."
            )

        # ITU-R BT.601 luminance
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)

        gray = img.astype(np.float64) @ weights          # (H, W)

        if image.dtype == np.uint8:
            return np.clip(gray, 0, 255).astype(np.uint8)
        else:
            return np.clip(gray, 0.0, 1.0).astype(image.dtype)

    # ==================================================================
    # Grayscale → RGB
    # ==================================================================
    if mode == "gray2rgb":
        if not is_grayscale(img):
            raise ValueError(
                f"[{_CALLER}] 'gray2rgb' requires a grayscale image "
                f"(H, W) or (H, W, 1), got shape {img.shape}."
            )

        # Stack the same channel three times → (H, W, 3)
        return np.stack([img, img, img], axis=-1)

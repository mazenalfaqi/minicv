"""
minicv.io.reader — Load image files into NumPy arrays.

Uses :func:`matplotlib.image.imread` as the decoding backend so that no
external image-processing library (PIL, OpenCV, etc.) is required.
"""

import numpy as np
import matplotlib.image as _mpl_img

from minicv.utils.validation import validate_path


def read_image(path, *, grayscale=False):
    """Load an image from disk into a NumPy array.

    Parameters
    ----------
    path : str
        Path to the image file.  PNG is fully supported; JPEG support
        depends on the Matplotlib backend (Pillow is **not** imported by
        MiniCV itself, but Matplotlib may use it transparently).
    grayscale : bool, optional
        If ``True``, the loaded image is converted to a 2-D grayscale
        array ``(H, W)`` with dtype ``uint8`` using the ITU-R BT.601
        luminance formula.  Default is ``False``.

    Returns
    -------
    numpy.ndarray
        * **RGB** — shape ``(H, W, 3)``, dtype ``uint8``, values in
          ``[0, 255]``.
        * **Grayscale** — shape ``(H, W)``, dtype ``uint8``, values in
          ``[0, 255]``.

    Raises
    ------
    TypeError
        If *path* is not a string.
    FileNotFoundError
        If the file does not exist on disk.
    ValueError
        If Matplotlib cannot decode the file or the resulting array has
        an unexpected shape.

    Notes
    -----
    * ``matplotlib.image.imread`` returns float32 in [0, 1] for PNG and
      uint8 in [0, 255] for JPEG (when the Pillow backend is active).
      This function **always** normalises the output to uint8 [0, 255]
      for a consistent downstream API.
    * RGBA images are silently converted to RGB by dropping the alpha
      channel.

    Examples
    --------
    >>> img = read_image("photo.png")
    >>> img.shape
    (480, 640, 3)
    >>> img.dtype
    dtype('uint8')

    >>> gray = read_image("photo.png", grayscale=True)
    >>> gray.shape
    (480, 640)
    """
    validate_path(path, must_exist=True, caller="read_image")

    if not isinstance(grayscale, bool):
        raise TypeError(
            f"[read_image] 'grayscale' must be bool, "
            f"got {type(grayscale).__name__}."
        )

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    try:
        raw = _mpl_img.imread(path)
    except Exception as exc:
        raise ValueError(
            f"[read_image] Failed to decode '{path}': {exc}"
        ) from exc

    if not isinstance(raw, np.ndarray) or raw.ndim not in (2, 3):
        raise ValueError(
            f"[read_image] Unexpected image shape after decoding: "
            f"{getattr(raw, 'shape', '?')}."
        )

    # ------------------------------------------------------------------
    # Normalise to uint8  [0, 255]
    # ------------------------------------------------------------------
    if np.issubdtype(raw.dtype, np.floating):
        # PNG path — float32 in [0, 1]
        image = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
    elif raw.dtype == np.uint8:
        image = raw
    else:
        # Unusual dtype — best-effort cast
        image = raw.astype(np.uint8)

    # ------------------------------------------------------------------
    # Handle channel count
    # ------------------------------------------------------------------
    if image.ndim == 3 and image.shape[2] == 4:
        # RGBA → RGB  (drop alpha)
        image = image[:, :, :3]

    if image.ndim == 3 and image.shape[2] == 1:
        # (H, W, 1) → (H, W)
        image = image[:, :, 0]

    # ------------------------------------------------------------------
    # Optional grayscale conversion
    # ------------------------------------------------------------------
    if grayscale and image.ndim == 3:
        # ITU-R BT.601 luminance weights
        image = (
            0.299 * image[:, :, 0].astype(np.float64)
            + 0.587 * image[:, :, 1].astype(np.float64)
            + 0.114 * image[:, :, 2].astype(np.float64)
        )
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image

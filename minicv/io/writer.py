"""
minicv.io.writer — Save in-memory images (NumPy arrays) to disk.

Uses :func:`matplotlib.image.imsave` as the encoding backend.
"""

import os
import numpy as np
import matplotlib.image as _mpl_img

from minicv.utils.validation import (
    validate_image_array,
    validate_path,
    ensure_uint8,
)


# Formats natively supported by matplotlib's Agg backend
_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def export_image(image, path):
    """Save a NumPy image array to an image file on disk.

    Parameters
    ----------
    image : numpy.ndarray
        Image to save.  Accepted shapes:

        * ``(H, W)``      — grayscale, saved as a single-channel PNG/JPG.
        * ``(H, W, 1)``   — grayscale (squeezed automatically).
        * ``(H, W, 3)``   — RGB colour image.

        Accepted dtypes: ``uint8`` [0, 255] or ``float`` [0.0, 1.0].
        Float images are converted to uint8 before saving.
    path : str
        Destination file path.  The extension determines the format and
        must be one of ``.png``, ``.jpg``, or ``.jpeg`` (case-insensitive).

    Returns
    -------
    str
        The absolute path of the written file.

    Raises
    ------
    TypeError
        If *image* is not a ``numpy.ndarray`` or *path* is not a ``str``.
    ValueError
        If *image* has an unsupported shape / dtype, the file extension
        is unrecognised, or the target directory does not exist.

    Notes
    -----
    * ``matplotlib.image.imsave`` expects uint8 data or float data in
      [0, 1].  This function always converts to uint8 first for
      deterministic behaviour.
    * Parent directories are **not** created automatically — the target
      folder must already exist.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> export_image(img, "black.png")
    '/absolute/path/to/black.png'

    >>> gray = np.full((64, 64), 128, dtype=np.uint8)
    >>> export_image(gray, "gray_square.jpg")
    """
    _CALLER = "export_image"

    # ---- Validate inputs ------------------------------------------------
    validate_image_array(image, allow_float=True, caller=_CALLER)
    validate_path(path, must_exist=False, caller=_CALLER)

    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"[{_CALLER}] Unsupported file extension '{ext}'. "
            f"Use one of {sorted(_SUPPORTED_EXTENSIONS)}."
        )

    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise ValueError(
            f"[{_CALLER}] Target directory does not exist: '{parent}'."
        )

    # ---- Prepare the array for matplotlib --------------------------------
    out = ensure_uint8(image, caller=_CALLER)

    # Squeeze (H, W, 1) → (H, W)
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]

    # matplotlib.image.imsave needs a grayscale image to have a cmap
    if out.ndim == 2:
        _mpl_img.imsave(path, out, cmap="gray")
    else:
        _mpl_img.imsave(path, out)

    return os.path.abspath(path)

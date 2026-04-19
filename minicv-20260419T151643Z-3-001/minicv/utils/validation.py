"""
minicv.utils.validation — Centralized input-validation helpers.

Every public function in MiniCV calls into these helpers so that error
messages are consistent and informative across the whole library.
"""

import numpy as np


def validate_image_array(image, *, allow_float=True, caller=""):
    """Validate that *image* is a well-formed image array.

    Parameters
    ----------
    image : numpy.ndarray
        The candidate image array to validate.
    allow_float : bool, optional
        If ``True`` (default), float images in [0, 1] are accepted in
        addition to uint8 images in [0, 255].
    caller : str, optional
        Name of the calling function, used to generate clearer error
        messages (e.g. ``"export_image"``).

    Returns
    -------
    numpy.ndarray
        The same *image* reference, unchanged.

    Raises
    ------
    TypeError
        If *image* is not a ``numpy.ndarray``.
    ValueError
        If *image* does not have 2 or 3 dimensions, if a 3-D image does
        not have 1, 3, or 4 channels, or if the dtype is not uint8 /
        float when expected.

    Notes
    -----
    Grayscale images must be 2-D ``(H, W)`` **or** 3-D with one channel
    ``(H, W, 1)``.  RGB images must be ``(H, W, 3)``.  RGBA images
    ``(H, W, 4)`` are accepted but some downstream functions may not
    support them.
    """
    prefix = f"[{caller}] " if caller else ""

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"{prefix}Expected a numpy.ndarray, got {type(image).__name__}."
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            f"{prefix}Image must be 2-D (H, W) or 3-D (H, W, C), "
            f"got shape {image.shape} with {image.ndim} dimensions."
        )

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"{prefix}3-D image must have 1, 3, or 4 channels, "
            f"got {image.shape[2]} (shape {image.shape})."
        )

    if image.size == 0:
        raise ValueError(f"{prefix}Image array is empty (shape {image.shape}).")

    # dtype check
    if image.dtype == np.uint8:
        pass  # always fine
    elif np.issubdtype(image.dtype, np.floating):
        if not allow_float:
            raise ValueError(
                f"{prefix}Float images are not allowed here; "
                f"convert to uint8 first (dtype is {image.dtype})."
            )
    else:
        raise ValueError(
            f"{prefix}Unsupported dtype '{image.dtype}'. "
            f"Expected uint8 or a float type."
        )

    return image


def validate_path(path, *, must_exist=False, caller=""):
    """Validate a filesystem path string.

    Parameters
    ----------
    path : str
        The file path to validate.
    must_exist : bool, optional
        If ``True``, raise ``FileNotFoundError`` when the path does not
        point to an existing file.
    caller : str, optional
        Name of the calling function for error-message context.

    Returns
    -------
    str
        The validated path.

    Raises
    ------
    TypeError
        If *path* is not a ``str``.
    FileNotFoundError
        If *must_exist* is ``True`` and the file does not exist.
    """
    import os

    prefix = f"[{caller}] " if caller else ""

    if not isinstance(path, str):
        raise TypeError(
            f"{prefix}Expected a file path string, got {type(path).__name__}."
        )

    if must_exist and not os.path.isfile(path):
        raise FileNotFoundError(f"{prefix}File not found: '{path}'.")

    return path


def is_grayscale(image):
    """Return ``True`` if *image* is a grayscale array.

    Parameters
    ----------
    image : numpy.ndarray
        A validated image array (2-D or 3-D).

    Returns
    -------
    bool
        ``True`` when the image is 2-D **or** 3-D with exactly 1 channel.
    """
    if image.ndim == 2:
        return True
    return image.ndim == 3 and image.shape[2] == 1


def is_rgb(image):
    """Return ``True`` if *image* is an RGB array ``(H, W, 3)``.

    Parameters
    ----------
    image : numpy.ndarray
        A validated image array.

    Returns
    -------
    bool
    """
    return image.ndim == 3 and image.shape[2] == 3


def ensure_uint8(image, *, caller=""):
    """Convert a float [0, 1] image to uint8 [0, 255], or pass uint8 through.

    Parameters
    ----------
    image : numpy.ndarray
        Image in uint8 or float [0, 1].
    caller : str, optional
        Calling function name for error messages.

    Returns
    -------
    numpy.ndarray
        Image with dtype ``uint8``.

    Raises
    ------
    ValueError
        If a float image has values outside [0, 1].
    """
    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        mn, mx = float(image.min()), float(image.max())
        if mn < -1e-6 or mx > 1.0 + 1e-6:
            raise ValueError(
                f"[{caller}] Float image values must be in [0, 1]; "
                f"found range [{mn:.4f}, {mx:.4f}]."
            )
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    raise ValueError(
        f"[{caller}] Cannot convert dtype '{image.dtype}' to uint8."
    )

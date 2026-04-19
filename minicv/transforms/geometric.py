"""
minicv.transforms.geometric — Geometric image transformations.

All transforms operate on NumPy arrays.  Coordinate mapping is fully
vectorized via meshgrid + fancy indexing — no Python pixel loops.

Public API
----------
resize    : Resize with nearest-neighbour or bilinear interpolation.
rotate    : Rotate about the image centre by an arbitrary angle.
translate : Shift the image in X and Y.
"""

import numpy as np

from minicv.utils.validation import validate_image_array


# =====================================================================
#  Interpolation helpers  (private, shared by resize / rotate / translate)
# =====================================================================

def _interp_nearest(image, map_y, map_x):
    """Sample *image* at fractional coordinates using nearest-neighbour.

    Parameters
    ----------
    image : numpy.ndarray
        Source image ``(H, W)`` or ``(H, W, C)``.
    map_y, map_x : numpy.ndarray
        Float arrays of source coordinates, shape ``(out_H, out_W)``.

    Returns
    -------
    numpy.ndarray
        Sampled image with shape ``(out_H, out_W [, C])``.
    """
    H, W = image.shape[:2]
    yi = np.clip(np.round(map_y).astype(np.intp), 0, H - 1)
    xi = np.clip(np.round(map_x).astype(np.intp), 0, W - 1)
    return image[yi, xi]


def _interp_bilinear(image, map_y, map_x):
    """Sample *image* at fractional coordinates using bilinear interpolation.

    Parameters
    ----------
    image : numpy.ndarray
        Source image ``(H, W)`` or ``(H, W, C)``, cast to float64
        internally.
    map_y, map_x : numpy.ndarray
        Float arrays of source coordinates, shape ``(out_H, out_W)``.

    Returns
    -------
    numpy.ndarray
        Interpolated image (float64) with shape ``(out_H, out_W [, C])``.

    Notes
    -----
    For each output pixel at fractional position ``(y, x)`` in the
    source, the four surrounding integer-coordinate neighbours are
    combined with weights derived from the fractional parts:

    .. math::

        I'(y, x) = (1-a)(1-b)\\,I[y_0, x_0] + (1-a)b\\,I[y_0, x_1]
                  + a(1-b)\\,I[y_1, x_0] + a\\,b\\,I[y_1, x_1]

    where ``a = y - y_0``, ``b = x - x_0``.
    Coordinates outside the image are clamped to the border.
    """
    H, W = image.shape[:2]
    src = image.astype(np.float64)

    # Floor and ceil coordinates (clamped)
    y0 = np.clip(np.floor(map_y).astype(np.intp), 0, H - 1)
    x0 = np.clip(np.floor(map_x).astype(np.intp), 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)

    # Fractional weights
    a = map_y - np.floor(map_y)    # vertical fraction
    b = map_x - np.floor(map_x)    # horizontal fraction

    # For multi-channel images we need extra dims for broadcasting
    if src.ndim == 3:
        a = a[..., np.newaxis]
        b = b[..., np.newaxis]

    val = (
        (1 - a) * (1 - b) * src[y0, x0]
        + (1 - a) * b       * src[y0, x1]
        + a       * (1 - b) * src[y1, x0]
        + a       * b       * src[y1, x1]
    )
    return val


_INTERP_MAP = {
    "nearest": _interp_nearest,
    "bilinear": _interp_bilinear,
}


def _resolve_interp(method, caller=""):
    """Return the interpolation function for *method* string."""
    if not isinstance(method, str):
        raise TypeError(
            f"[{caller}] 'interpolation' must be str, "
            f"got {type(method).__name__}."
        )
    method = method.strip().lower()
    if method not in _INTERP_MAP:
        raise ValueError(
            f"[{caller}] Unknown interpolation '{method}'. "
            f"Supported: {list(_INTERP_MAP.keys())}."
        )
    return _INTERP_MAP[method]


# =====================================================================
#  1.  Resize
# =====================================================================

def resize(image, new_height, new_width, *, interpolation="bilinear"):
    """Resize an image to ``(new_height, new_width)``.

    Parameters
    ----------
    image : numpy.ndarray
        Input image ``(H, W)`` or ``(H, W, C)``.
    new_height : int
        Target height in pixels (must be ≥ 1).
    new_width : int
        Target width in pixels (must be ≥ 1).
    interpolation : str, optional
        ``"nearest"`` — nearest-neighbour (fast, blocky).
        ``"bilinear"`` — bilinear (smooth, default).

    Returns
    -------
    numpy.ndarray
        Resized image.  dtype matches the input for nearest-neighbour;
        for bilinear the output is ``float64`` (caller can clip + cast
        to uint8 if needed).

    Raises
    ------
    TypeError
        If *image* is not ndarray or dimensions are not int.
    ValueError
        If target dimensions are < 1.

    Notes
    -----
    Coordinate mapping uses the **centre-aligned** convention:

    .. math::

        src_y = \\frac{(dst_y + 0.5) \\cdot H_{src}}{H_{dst}} - 0.5

    which avoids the half-pixel shift artefact that a naïve
    ``dst_y * H_src / H_dst`` mapping produces.  This matches the
    ``align_corners=False`` behaviour in most modern frameworks.

    Examples
    --------
    >>> small = resize(img, 64, 64)
    >>> big   = resize(img, 1024, 1024, interpolation="nearest")
    """
    _CALLER = "resize"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    interp_fn = _resolve_interp(interpolation, _CALLER)

    for name, val in [("new_height", new_height), ("new_width", new_width)]:
        if not isinstance(val, (int, np.integer)):
            raise TypeError(
                f"[{_CALLER}] '{name}' must be int, got "
                f"{type(val).__name__}."
            )
        if int(val) < 1:
            raise ValueError(
                f"[{_CALLER}] '{name}' must be ≥ 1, got {val}."
            )

    new_h, new_w = int(new_height), int(new_width)
    H, W = image.shape[:2]

    # ── build coordinate maps (centre-aligned) ───────────────────────
    dst_y = np.arange(new_h, dtype=np.float64)
    dst_x = np.arange(new_w, dtype=np.float64)

    src_y = (dst_y + 0.5) * H / new_h - 0.5
    src_x = (dst_x + 0.5) * W / new_w - 0.5

    map_y, map_x = np.meshgrid(src_y, src_x, indexing="ij")

    result = interp_fn(image, map_y, map_x)

    # Preserve dtype for nearest
    if interpolation.strip().lower() == "nearest":
        return result.astype(image.dtype)
    return result


# =====================================================================
#  2.  Rotation
# =====================================================================

def rotate(image, angle_deg, *, interpolation="bilinear",
           border_value=0):
    """Rotate an image about its centre by *angle_deg* degrees.

    Parameters
    ----------
    image : numpy.ndarray
        Input image ``(H, W)`` or ``(H, W, C)``.
    angle_deg : float
        Counter-clockwise rotation angle in **degrees**.
    interpolation : str, optional
        ``"nearest"`` or ``"bilinear"`` (default).
    border_value : int or float, optional
        Value used to fill pixels that fall outside the original image
        after rotation.  Default 0 (black).

    Returns
    -------
    numpy.ndarray
        Rotated image with the **same spatial dimensions** as the
        input.  Pixels outside the source are filled with
        *border_value*.  dtype is ``float64`` for bilinear, or same
        as input for nearest.

    Notes
    -----
    The rotation is performed via **inverse mapping**: for each pixel
    ``(dy, dx)`` in the output, the corresponding source coordinate is
    computed as:

    .. math::

        \\begin{pmatrix} sx \\\\ sy \\end{pmatrix}
        = R^{-1}(\\theta)
          \\begin{pmatrix} dx - c_x \\\\ dy - c_y \\end{pmatrix}
        + \\begin{pmatrix} c_x \\\\ c_y \\end{pmatrix}

    where ``R⁻¹(θ) = R(−θ)`` is the inverse rotation matrix and
    ``(c_x, c_y)`` is the image centre.

    Examples
    --------
    >>> rotated = rotate(img, 45)   # 45° CCW
    >>> rotated = rotate(img, -90, interpolation="nearest")
    """
    _CALLER = "rotate"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    interp_fn = _resolve_interp(interpolation, _CALLER)

    if not isinstance(angle_deg, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"[{_CALLER}] 'angle_deg' must be numeric, "
            f"got {type(angle_deg).__name__}."
        )

    H, W = image.shape[:2]
    theta = np.deg2rad(float(angle_deg))

    # Centre of image
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    # Inverse rotation matrix (rotate by -θ to find source coordinates)
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)

    # Build destination coordinate grid
    dst_y, dst_x = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )

    # Translate to origin, rotate, translate back
    dx = dst_x - cx
    dy = dst_y - cy

    src_x = cos_t * dx - sin_t * dy + cx
    src_y = sin_t * dx + cos_t * dy + cy

    # ── determine which pixels fall inside the source ────────────────
    inside = (
        (src_y >= -0.5) & (src_y <= H - 0.5) &
        (src_x >= -0.5) & (src_x <= W - 0.5)
    )

    # Clamp for interpolation (border pixels extrapolate safely)
    src_y_clamped = np.clip(src_y, 0, H - 1)
    src_x_clamped = np.clip(src_x, 0, W - 1)

    result = interp_fn(image, src_y_clamped, src_x_clamped)

    # Fill out-of-bounds with border_value
    if image.ndim == 2:
        result = np.where(inside, result, border_value)
    else:
        result = np.where(inside[..., np.newaxis], result, border_value)

    if interpolation.strip().lower() == "nearest":
        return result.astype(image.dtype)
    return result.astype(np.float64)


# =====================================================================
#  3.  Translation
# =====================================================================

def translate(image, tx, ty, *, interpolation="bilinear",
              border_value=0):
    """Shift an image by ``(tx, ty)`` pixels.

    Parameters
    ----------
    image : numpy.ndarray
        Input image ``(H, W)`` or ``(H, W, C)``.
    tx : int or float
        Horizontal shift (positive = right).
    ty : int or float
        Vertical shift (positive = down).
    interpolation : str, optional
        ``"nearest"`` or ``"bilinear"`` (default).
    border_value : int or float, optional
        Fill value for uncovered regions (default 0).

    Returns
    -------
    numpy.ndarray
        Translated image with the same spatial dimensions.  Pixels
        outside the source are filled with *border_value*.

    Notes
    -----
    Implemented via inverse mapping: for each destination pixel
    ``(dy, dx)``, the source pixel is ``(dy − ty, dx − tx)``.

    Sub-pixel shifts are supported when *interpolation* is
    ``"bilinear"`` — this is useful for sub-pixel registration.

    Examples
    --------
    >>> shifted = translate(img, tx=10, ty=-5)
    """
    _CALLER = "translate"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    interp_fn = _resolve_interp(interpolation, _CALLER)

    for name, val in [("tx", tx), ("ty", ty)]:
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"[{_CALLER}] '{name}' must be numeric, "
                f"got {type(val).__name__}."
            )

    H, W = image.shape[:2]
    tx, ty = float(tx), float(ty)

    dst_y, dst_x = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )

    src_x = dst_x - tx
    src_y = dst_y - ty

    inside = (
        (src_y >= -0.5) & (src_y <= H - 0.5) &
        (src_x >= -0.5) & (src_x <= W - 0.5)
    )

    src_y_c = np.clip(src_y, 0, H - 1)
    src_x_c = np.clip(src_x, 0, W - 1)

    result = interp_fn(image, src_y_c, src_x_c)

    if image.ndim == 2:
        result = np.where(inside, result, border_value)
    else:
        result = np.where(inside[..., np.newaxis], result, border_value)

    if interpolation.strip().lower() == "nearest":
        return result.astype(image.dtype)
    return result.astype(np.float64)

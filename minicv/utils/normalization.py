"""
minicv.utils.normalization — Image intensity normalization.

Four distinct modes are provided, covering the most common use-cases in
machine-vision preprocessing pipelines.  Every operation is fully
vectorized via NumPy.
"""

import numpy as np

from minicv.utils.validation import validate_image_array


# ── public constant for introspection ────────────────────────────────────
NORMALIZATION_MODES = ("minmax", "zscore", "mean", "range")


def normalize(image, mode="minmax", *, target_range=(0.0, 1.0)):
    """Normalize an image's pixel intensities.

    Parameters
    ----------
    image : numpy.ndarray
        Input image — 2-D ``(H, W)`` grayscale or 3-D ``(H, W, C)`` colour.
        Both ``uint8`` and float dtypes are accepted.
    mode : str, optional
        Normalization strategy.  One of:

        ``"minmax"``
            Linear rescaling into *target_range*:
            ``out = (pixel - min) / (max - min) * (hi - lo) + lo``.
            If the image is constant (max == min) the output is filled
            with the midpoint of *target_range*.
        ``"zscore"``
            Standardise to zero mean and unit variance:
            ``out = (pixel - μ) / σ``.  If σ == 0 the output is all
            zeros.  The result is a float64 array that can contain
            negative values.
        ``"mean"``
            Mean-centre only (subtract the global mean):
            ``out = pixel - μ``.  Result is float64 and may be negative.
        ``"range"``
            Scale by the value range without shifting:
            ``out = pixel / (max - min)``.  If the image is constant
            the output is 0.0.

        Default is ``"minmax"``.
    target_range : tuple of (float, float), optional
        Only used when *mode* is ``"minmax"``.  The desired output
        ``(low, high)`` range.  Default ``(0.0, 1.0)``.

    Returns
    -------
    numpy.ndarray
        Normalized image with dtype ``float64``.

    Raises
    ------
    TypeError
        If *image* is not a ``numpy.ndarray`` or *mode* is not a ``str``.
    ValueError
        If *mode* is unrecognised, or *target_range* is invalid.

    Notes
    -----
    * The output is **always** ``float64`` regardless of the input dtype.
      Downstream functions (e.g. ``clip_image``, ``ensure_uint8``) can
      convert back to uint8 if needed.
    * For multi-channel images the statistics (min, max, μ, σ) are
      computed **globally** across all channels, which is the standard
      convention for image normalization.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[0, 128], [64, 255]], dtype=np.uint8)
    >>> out = normalize(img, "minmax")
    >>> float(out.min()), float(out.max())
    (0.0, 1.0)

    >>> out_z = normalize(img, "zscore")
    >>> abs(float(out_z.mean())) < 1e-10
    True
    """
    _CALLER = "normalize"

    # ── validate image ───────────────────────────────────────────────
    validate_image_array(image, allow_float=True, caller=_CALLER)

    if not isinstance(mode, str):
        raise TypeError(
            f"[{_CALLER}] 'mode' must be a str, got {type(mode).__name__}."
        )
    mode = mode.strip().lower()
    if mode not in NORMALIZATION_MODES:
        raise ValueError(
            f"[{_CALLER}] Unknown mode '{mode}'. "
            f"Supported: {NORMALIZATION_MODES}."
        )

    # ── validate target_range ────────────────────────────────────────
    if mode == "minmax":
        if (
            not isinstance(target_range, (tuple, list))
            or len(target_range) != 2
        ):
            raise ValueError(
                f"[{_CALLER}] 'target_range' must be a 2-tuple (lo, hi), "
                f"got {target_range!r}."
            )
        lo, hi = float(target_range[0]), float(target_range[1])
        if lo >= hi:
            raise ValueError(
                f"[{_CALLER}] target_range low ({lo}) must be < high ({hi})."
            )

    # ── work in float64 ──────────────────────────────────────────────
    arr = image.astype(np.float64)

    # ── apply mode ───────────────────────────────────────────────────
    if mode == "minmax":
        mn, mx = arr.min(), arr.max()
        if mx - mn == 0:
            return np.full_like(arr, (lo + hi) / 2.0)
        return (arr - mn) / (mx - mn) * (hi - lo) + lo

    if mode == "zscore":
        mu = arr.mean()
        sigma = arr.std()
        if sigma == 0:
            return np.zeros_like(arr)
        return (arr - mu) / sigma

    if mode == "mean":
        return arr - arr.mean()

    # mode == "range"
    mn, mx = arr.min(), arr.max()
    rng = mx - mn
    if rng == 0:
        return np.zeros_like(arr)
    return arr / rng

"""
minicv.features.descriptors — Image feature extraction.

All descriptors produce fixed-length 1-D feature vectors from images
using only NumPy.  They are designed for use in machine-vision
classification pipelines (Milestone 2).

Public API — Global Descriptors
-------------------------------
color_histogram_descriptor : Multi-channel intensity histogram (flattened).
statistical_moments        : Per-channel mean, std, skew, kurtosis.

Public API — Gradient Descriptors
---------------------------------
hog_descriptor  : Simplified Histogram of Oriented Gradients (HOG).
edge_descriptor : Edge density + gradient orientation histogram.
"""

import numpy as np

from minicv.utils.validation import validate_image_array, is_grayscale
from minicv.filtering.filters import sobel_x, sobel_y


# =====================================================================
#  Global Descriptor 1:  Color / Intensity Histogram
# =====================================================================

def color_histogram_descriptor(image, bins=32):
    """Compute a normalised color-histogram feature vector.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image, dtype uint8.
    bins : int, optional
        Number of histogram bins per channel (default 32).

    Returns
    -------
    numpy.ndarray
        1-D feature vector of length ``bins`` (grayscale) or
        ``3 * bins`` (RGB), dtype ``float64``, L1-normalised
        (sums to 1).

    Raises
    ------
    TypeError
        If *image* is not uint8 or *bins* is not int.
    ValueError
        If *bins* < 1 or > 256.

    Notes
    -----
    Each channel's [0, 255] range is quantised into *bins* equal-width
    buckets and the resulting counts are concatenated (R, G, B order
    for colour images) and normalised by total pixel count so the
    descriptor is **scale-invariant** (independent of image size).

    This is one of the simplest yet most effective global descriptors
    for texture-free classification tasks.

    Examples
    --------
    >>> feat = color_histogram_descriptor(rgb_img, bins=64)
    >>> feat.shape
    (192,)
    >>> abs(feat.sum() - 1.0) < 1e-10
    True
    """
    _CALLER = "color_histogram_descriptor"
    validate_image_array(image, allow_float=False, caller=_CALLER)

    if image.dtype != np.uint8:
        raise TypeError(f"[{_CALLER}] Image must be uint8, got {image.dtype}.")
    if not isinstance(bins, (int, np.integer)):
        raise TypeError(
            f"[{_CALLER}] 'bins' must be int, got {type(bins).__name__}."
        )
    bins = int(bins)
    if bins < 1 or bins > 256:
        raise ValueError(f"[{_CALLER}] 'bins' must be in [1, 256], got {bins}.")

    bin_edges = np.linspace(0, 256, bins + 1)

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        plane = image.ravel()
        hist = np.histogram(plane, bins=bin_edges)[0].astype(np.float64)
    else:
        # Per-channel histograms concatenated: R | G | B
        hists = []
        for c in range(image.shape[2]):
            h = np.histogram(image[:, :, c].ravel(), bins=bin_edges)[0]
            hists.append(h.astype(np.float64))
        hist = np.concatenate(hists)

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# =====================================================================
#  Global Descriptor 2:  Statistical Moments
# =====================================================================

def statistical_moments(image):
    """Compute per-channel statistical moments as a feature vector.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image (uint8 or
        float).

    Returns
    -------
    numpy.ndarray
        1-D float64 feature vector of length ``4`` (grayscale) or
        ``12`` (RGB).  For each channel the four values are:

        0. **mean** — first moment (average intensity).
        1. **std** — second central moment (spread).
        2. **skewness** — third standardised moment (asymmetry).
        3. **kurtosis** — fourth standardised moment (tail weight),
           *excess* kurtosis (normal = 0).

    Raises
    ------
    TypeError
        If *image* is not ndarray.

    Notes
    -----
    Skewness and kurtosis are computed from scratch:

    .. math::

        \\text{skew} = \\frac{E[(X - \\mu)^3]}{\\sigma^3}, \\quad
        \\text{kurt} = \\frac{E[(X - \\mu)^4]}{\\sigma^4} - 3

    When σ = 0 (constant channel), both are set to 0.

    Examples
    --------
    >>> m = statistical_moments(rgb_img)
    >>> m.shape
    (12,)
    """
    _CALLER = "statistical_moments"
    validate_image_array(image, allow_float=True, caller=_CALLER)

    def _moments_1d(arr):
        """Return [mean, std, skewness, excess_kurtosis] for a flat array."""
        a = arr.astype(np.float64)
        mu = a.mean()
        sigma = a.std()
        if sigma < 1e-12:
            return np.array([mu, 0.0, 0.0, 0.0], dtype=np.float64)
        centered = a - mu
        skew = (centered ** 3).mean() / (sigma ** 3)
        kurt = (centered ** 4).mean() / (sigma ** 4) - 3.0
        return np.array([mu, sigma, skew, kurt], dtype=np.float64)

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return _moments_1d(image.ravel())

    # Multi-channel — concatenate per-channel moments
    parts = []
    for c in range(image.shape[2]):
        parts.append(_moments_1d(image[:, :, c].ravel()))
    return np.concatenate(parts)


# =====================================================================
#  Gradient Descriptor 1:  Simplified HOG
# =====================================================================

def hog_descriptor(image, cell_size=8, n_bins=9):
    """Compute a simplified Histogram of Oriented Gradients (HOG).

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` image, dtype uint8 or float.
    cell_size : int, optional
        Side length of each spatial cell in pixels (default 8).
    n_bins : int, optional
        Number of orientation bins spanning [0°, 180°) — the standard
        *unsigned* gradient convention (default 9 → 20° per bin).

    Returns
    -------
    numpy.ndarray
        1-D float64 feature vector of length
        ``(H // cell_size) * (W // cell_size) * n_bins``.
        Each cell's histogram is L2-normalised.

    Raises
    ------
    TypeError / ValueError
        On invalid inputs.

    Notes
    -----
    **Algorithm (simplified — no block-level normalisation):**

    1. Compute gradients Gx, Gy via 3×3 Sobel (using our convolution
       engine).
    2. Compute magnitude ``M = sqrt(Gx² + Gy²)`` and unsigned
       orientation ``θ = arctan2(Gy, Gx) mod 180°`` (maps to
       [0°, 180°)).
    3. Divide the image into non-overlapping ``cell_size × cell_size``
       cells.
    4. For each cell, accumulate a weighted orientation histogram
       (weights = magnitudes) into *n_bins* bins.
    5. L2-normalise each cell histogram.
    6. Concatenate all cells into a single feature vector.

    This omits the 2×2 block normalisation pass used in Dalal & Triggs
    (2005) for simplicity, but still produces a strong descriptor.

    Examples
    --------
    >>> feat = hog_descriptor(gray, cell_size=8, n_bins=9)
    >>> feat.shape  # for a 64×64 image → 8×8 cells × 9 bins
    (576,)
    """
    _CALLER = "hog_descriptor"
    validate_image_array(image, allow_float=True, caller=_CALLER)

    if image.ndim != 2:
        raise ValueError(
            f"[{_CALLER}] Expected 2-D grayscale, got shape {image.shape}. "
            f"Convert to grayscale first."
        )
    for name, val in [("cell_size", cell_size), ("n_bins", n_bins)]:
        if not isinstance(val, (int, np.integer)):
            raise TypeError(
                f"[{_CALLER}] '{name}' must be int, got "
                f"{type(val).__name__}."
            )
    cell_size, n_bins = int(cell_size), int(n_bins)
    if cell_size < 1:
        raise ValueError(f"[{_CALLER}] 'cell_size' must be ≥ 1, got {cell_size}.")
    if n_bins < 1:
        raise ValueError(f"[{_CALLER}] 'n_bins' must be ≥ 1, got {n_bins}.")

    H, W = image.shape

    # ── gradients ────────────────────────────────────────────────────
    gx = sobel_x(image, pad_mode="edge")   # (H, W) float64
    gy = sobel_y(image, pad_mode="edge")

    mag = np.sqrt(gx ** 2 + gy ** 2)

    # Unsigned orientation [0, 180) degrees
    orient = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    # ── crop to exact multiple of cell_size ──────────────────────────
    cells_h = H // cell_size
    cells_w = W // cell_size
    crop_h = cells_h * cell_size
    crop_w = cells_w * cell_size
    mag    = mag[:crop_h, :crop_w]
    orient = orient[:crop_h, :crop_w]

    # ── reshape into cells ───────────────────────────────────────────
    # (cells_h, cell_size, cells_w, cell_size)
    mag_cells = mag.reshape(cells_h, cell_size, cells_w, cell_size)
    ori_cells = orient.reshape(cells_h, cell_size, cells_w, cell_size)

    # Transpose to (cells_h, cells_w, cell_size, cell_size) then flatten
    mag_cells = mag_cells.transpose(0, 2, 1, 3).reshape(cells_h, cells_w, -1)
    ori_cells = ori_cells.transpose(0, 2, 1, 3).reshape(cells_h, cells_w, -1)

    # ── bin the orientations per cell (vectorized) ───────────────────
    bin_width = 180.0 / n_bins
    bin_idx = np.clip((ori_cells / bin_width).astype(np.intp), 0, n_bins - 1)

    # Build histograms for all cells at once
    histograms = np.zeros((cells_h, cells_w, n_bins), dtype=np.float64)
    for b in range(n_bins):
        mask = (bin_idx == b)
        histograms[:, :, b] = np.sum(mag_cells * mask, axis=2)

    # ── L2-normalise each cell ───────────────────────────────────────
    norms = np.sqrt((histograms ** 2).sum(axis=2, keepdims=True)) + 1e-12
    histograms /= norms

    return histograms.ravel()


# =====================================================================
#  Gradient Descriptor 2:  Edge Density + Orientation Histogram
# =====================================================================

def edge_descriptor(image, n_orient_bins=8, edge_threshold=30.0):
    """Compute an edge-based feature vector.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` image, dtype uint8 or float.
    n_orient_bins : int, optional
        Number of orientation bins spanning [0°, 360°) (default 8 →
        45° per bin).
    edge_threshold : float, optional
        Minimum gradient magnitude for a pixel to be considered an
        edge (default 30.0).

    Returns
    -------
    numpy.ndarray
        1-D float64 feature vector of length ``n_orient_bins + 3``:

        * ``[0]``                    — **edge density** (fraction of
          pixels classified as edges).
        * ``[1]``                    — **mean edge magnitude** (over
          edge pixels only; 0 if no edges).
        * ``[2]``                    — **std of edge magnitude**.
        * ``[3 : 3+n_orient_bins]`` — normalised histogram of edge
          orientations across [0°, 360°).

    Raises
    ------
    TypeError / ValueError
        On invalid inputs.

    Notes
    -----
    This descriptor captures both **how much** edge content an image
    has (density and magnitude statistics) and **in which directions**
    those edges run (orientation histogram).  It is complementary to
    HOG: while HOG is spatial (cell grid), this descriptor is purely
    global and very compact.

    Examples
    --------
    >>> feat = edge_descriptor(gray)
    >>> feat.shape
    (11,)
    """
    _CALLER = "edge_descriptor"
    validate_image_array(image, allow_float=True, caller=_CALLER)

    if image.ndim != 2:
        raise ValueError(
            f"[{_CALLER}] Expected 2-D grayscale, got shape {image.shape}."
        )
    if not isinstance(n_orient_bins, (int, np.integer)):
        raise TypeError(
            f"[{_CALLER}] 'n_orient_bins' must be int, "
            f"got {type(n_orient_bins).__name__}."
        )
    n_orient_bins = int(n_orient_bins)
    if n_orient_bins < 1:
        raise ValueError(
            f"[{_CALLER}] 'n_orient_bins' must be ≥ 1, got {n_orient_bins}."
        )

    gx = sobel_x(image, pad_mode="edge")
    gy = sobel_y(image, pad_mode="edge")
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # Edge mask
    edge_mask = mag >= float(edge_threshold)
    total_pixels = float(image.size)
    edge_count = float(edge_mask.sum())

    # Density
    density = edge_count / total_pixels if total_pixels > 0 else 0.0

    # Magnitude statistics (over edge pixels only)
    if edge_count > 0:
        edge_mags = mag[edge_mask]
        mean_mag = float(edge_mags.mean())
        std_mag = float(edge_mags.std())
    else:
        mean_mag = 0.0
        std_mag = 0.0

    # Signed orientation [0, 360) degrees
    orient = np.rad2deg(np.arctan2(gy, gx)) % 360.0

    # Orientation histogram (edge pixels only, magnitude-weighted)
    bin_width = 360.0 / n_orient_bins
    orient_edges = orient[edge_mask]
    mag_edges = mag[edge_mask]

    hist = np.zeros(n_orient_bins, dtype=np.float64)
    if edge_count > 0:
        bin_idx = np.clip(
            (orient_edges / bin_width).astype(np.intp), 0, n_orient_bins - 1
        )
        # Vectorized accumulation via bincount
        hist = np.bincount(bin_idx, weights=mag_edges,
                           minlength=n_orient_bins).astype(np.float64)
        h_sum = hist.sum()
        if h_sum > 0:
            hist /= h_sum

    return np.concatenate([[density, mean_mag, std_mag], hist])

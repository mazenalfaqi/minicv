"""
minicv.drawing.primitives — Canvas drawing operations on NumPy arrays.

All functions **mutate** the image array in-place and also return it
for chaining.  Every function clips to canvas boundaries — drawing
off-screen never raises an error.

Color convention
~~~~~~~~~~~~~~~~
* Grayscale image ``(H, W)`` → *color* is an ``int`` (0–255).
* RGB image ``(H, W, 3)`` → *color* is a ``(R, G, B)`` tuple of ints.

Public API
----------
draw_point     : Set one or more pixels.
draw_line      : Bresenham's line algorithm.
draw_rectangle : Filled and/or outline rectangle.
draw_polygon   : Outline polygon (filled optional).
put_text       : Render text with a built-in 5×7 blocky pixel font.
"""

import numpy as np

from minicv.utils.validation import validate_image_array


# =====================================================================
#  Internal helpers
# =====================================================================

def _resolve_color(image, color, caller=""):
    """Validate and normalise *color* to match the image type."""
    prefix = f"[{caller}] " if caller else ""
    if image.ndim == 2:
        if not isinstance(color, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"{prefix}For grayscale images, 'color' must be a scalar "
                f"int/float, got {type(color).__name__}."
            )
        return int(color)
    # RGB
    if (
        not isinstance(color, (tuple, list, np.ndarray))
        or len(color) != 3
    ):
        raise TypeError(
            f"{prefix}For RGB images, 'color' must be a 3-element "
            f"tuple/list (R, G, B), got {color!r}."
        )
    return tuple(int(c) for c in color)


def _set_pixel(image, y, x, color):
    """Set a single pixel with boundary clipping (no-op if outside)."""
    H, W = image.shape[:2]
    if 0 <= y < H and 0 <= x < W:
        image[y, x] = color


def _set_pixels_bulk(image, ys, xs, color):
    """Set many pixels at once with clipping — fully vectorized."""
    H, W = image.shape[:2]
    mask = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    image[ys[mask], xs[mask]] = color


# =====================================================================
#  1. Draw Point
# =====================================================================

def draw_point(image, y, x, color, *, thickness=1):
    """Draw a point (or small filled square) on the image.

    Parameters
    ----------
    image : numpy.ndarray
        Canvas image (modified in-place).
    y : int
        Row coordinate (0-indexed from top).
    x : int
        Column coordinate (0-indexed from left).
    color : int or tuple
        Scalar for grayscale, ``(R, G, B)`` for RGB.
    thickness : int, optional
        Side-length of the square point marker (default 1 = single
        pixel).  Must be ≥ 1.

    Returns
    -------
    numpy.ndarray
        The same *image* array (modified in-place).

    Raises
    ------
    TypeError
        If coordinates or color are wrong type.
    ValueError
        If thickness < 1.

    Notes
    -----
    Pixels outside the canvas are silently clipped.
    """
    _CALLER = "draw_point"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    color = _resolve_color(image, color, _CALLER)

    if not isinstance(thickness, (int, np.integer)):
        raise TypeError(
            f"[{_CALLER}] 'thickness' must be int, "
            f"got {type(thickness).__name__}."
        )
    thickness = int(thickness)
    if thickness < 1:
        raise ValueError(f"[{_CALLER}] 'thickness' must be ≥ 1, got {thickness}.")

    y, x = int(y), int(x)
    half = thickness // 2

    ys = np.arange(y - half, y - half + thickness, dtype=np.intp)
    xs = np.arange(x - half, x - half + thickness, dtype=np.intp)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    _set_pixels_bulk(image, yy.ravel(), xx.ravel(), color)

    return image


# =====================================================================
#  2. Draw Line  (Bresenham's Algorithm)
# =====================================================================

def draw_line(image, y0, x0, y1, x1, color, *, thickness=1):
    """Draw a straight line using Bresenham's algorithm.

    Parameters
    ----------
    image : numpy.ndarray
        Canvas image (modified in-place).
    y0, x0 : int
        Start point (row, col).
    y1, x1 : int
        End point (row, col).
    color : int or tuple
        Pixel color.
    thickness : int, optional
        Line width in pixels (default 1).  Implemented by drawing a
        filled square of this size at each Bresenham point.

    Returns
    -------
    numpy.ndarray
        The same *image* array (modified in-place).

    Notes
    -----
    **Bresenham's line algorithm** uses only integer arithmetic
    (additions and comparisons) to determine which pixels best
    approximate a straight line between two endpoints.  The
    implementation collects all pixel coordinates first, then writes
    them in a single vectorized call.

    Endpoints outside the canvas are allowed — only on-canvas pixels
    are drawn.
    """
    _CALLER = "draw_line"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    color = _resolve_color(image, color, _CALLER)

    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)

    # ── Bresenham core ───────────────────────────────────────────────
    pts_y, pts_x = _bresenham(y0, x0, y1, x1)

    if thickness <= 1:
        _set_pixels_bulk(image, pts_y, pts_x, color)
    else:
        # Thicken: for each Bresenham point, draw a square
        half = thickness // 2
        offsets = np.arange(-half, -half + thickness, dtype=np.intp)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        dy, dx = dy.ravel(), dx.ravel()

        all_y = (pts_y[:, None] + dy[None, :]).ravel()
        all_x = (pts_x[:, None] + dx[None, :]).ravel()
        _set_pixels_bulk(image, all_y, all_x, color)

    return image


def _bresenham(y0, x0, y1, x1):
    """Return arrays of (y, x) coordinates along the Bresenham line.

    This is the classic integer-arithmetic algorithm generalized to
    all octants.  Returns ``(ys, xs)`` as int numpy arrays.
    """
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    pts_y = []
    pts_x = []
    cx, cy = x0, y0
    while True:
        pts_y.append(cy)
        pts_x.append(cx)
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            cx += sx
        if e2 <= dx:
            err += dx
            cy += sy

    return np.array(pts_y, dtype=np.intp), np.array(pts_x, dtype=np.intp)


# =====================================================================
#  3. Draw Rectangle
# =====================================================================

def draw_rectangle(image, y0, x0, y1, x1, color, *,
                   thickness=1, filled=False):
    """Draw an axis-aligned rectangle.

    Parameters
    ----------
    image : numpy.ndarray
        Canvas image (modified in-place).
    y0, x0 : int
        Top-left corner (row, col).
    y1, x1 : int
        Bottom-right corner (row, col), **inclusive**.
    color : int or tuple
        Pixel color.
    thickness : int, optional
        Outline width (default 1).  Ignored when *filled* is ``True``.
    filled : bool, optional
        If ``True``, fill the entire rectangle (default ``False``).

    Returns
    -------
    numpy.ndarray
        The same *image* array (modified in-place).
    """
    _CALLER = "draw_rectangle"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    color = _resolve_color(image, color, _CALLER)

    # Ensure correct ordering
    y0, y1 = min(int(y0), int(y1)), max(int(y0), int(y1))
    x0, x1 = min(int(x0), int(x1)), max(int(x0), int(x1))

    H, W = image.shape[:2]

    if filled:
        # Clip to canvas then fill
        ry0 = max(y0, 0)
        rx0 = max(x0, 0)
        ry1 = min(y1, H - 1)
        rx1 = min(x1, W - 1)
        if ry0 <= ry1 and rx0 <= rx1:
            image[ry0:ry1 + 1, rx0:rx1 + 1] = color
    else:
        # Outline — draw 4 sides
        t = max(1, int(thickness))
        draw_line(image, y0, x0, y0, x1, color, thickness=t)  # top
        draw_line(image, y1, x0, y1, x1, color, thickness=t)  # bottom
        draw_line(image, y0, x0, y1, x0, color, thickness=t)  # left
        draw_line(image, y0, x1, y1, x1, color, thickness=t)  # right

    return image


# =====================================================================
#  4. Draw Polygon
# =====================================================================

def draw_polygon(image, points, color, *, thickness=1, filled=False):
    """Draw a polygon defined by a list of vertices.

    Parameters
    ----------
    image : numpy.ndarray
        Canvas image (modified in-place).
    points : list of (int, int)
        Vertices as ``[(y0, x0), (y1, x1), ...]``.  At least 3
        vertices are required.
    color : int or tuple
        Pixel color.
    thickness : int, optional
        Outline width (default 1).
    filled : bool, optional
        If ``True``, fill the polygon interior using a scanline
        algorithm (default ``False`` — outline only).

    Returns
    -------
    numpy.ndarray
        The same *image* array (modified in-place).

    Notes
    -----
    **Outline:** connects consecutive vertices and closes the polygon
    (last vertex → first vertex) via :func:`draw_line`.

    **Fill:** uses a standard scanline algorithm: for each row,
    compute all x-intersections with the polygon edges, sort them,
    and fill pixel spans between consecutive pairs.  This is fully
    vectorized per-row.
    """
    _CALLER = "draw_polygon"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    color = _resolve_color(image, color, _CALLER)

    if not isinstance(points, (list, tuple, np.ndarray)):
        raise TypeError(
            f"[{_CALLER}] 'points' must be a list of (y, x) tuples."
        )
    if len(points) < 3:
        raise ValueError(
            f"[{_CALLER}] Need at least 3 vertices, got {len(points)}."
        )

    pts = [(int(p[0]), int(p[1])) for p in points]
    n = len(pts)

    if filled:
        _fill_polygon(image, pts, color)
    # Always draw the outline (over the fill if both)
    for i in range(n):
        j = (i + 1) % n
        draw_line(image, pts[i][0], pts[i][1],
                  pts[j][0], pts[j][1], color, thickness=thickness)

    return image


def _fill_polygon(image, pts, color):
    """Scanline polygon fill.

    For each scanline y, find all x-intersections with polygon edges,
    sort them, and fill between consecutive pairs.
    """
    H, W = image.shape[:2]
    n = len(pts)

    ys = [p[0] for p in pts]
    y_min = max(0, min(ys))
    y_max = min(H - 1, max(ys))

    for y in range(y_min, y_max + 1):
        x_intersections = []
        for i in range(n):
            j = (i + 1) % n
            y0, x0 = pts[i]
            y1, x1 = pts[j]
            if y0 == y1:
                continue   # horizontal edge — skip
            if (y0 <= y < y1) or (y1 <= y < y0):
                # Linear interpolation for x at this y
                t = (y - y0) / (y1 - y0)
                x_int = x0 + t * (x1 - x0)
                x_intersections.append(x_int)
        x_intersections.sort()

        # Fill between pairs
        for k in range(0, len(x_intersections) - 1, 2):
            xl = max(0, int(np.ceil(x_intersections[k])))
            xr = min(W - 1, int(np.floor(x_intersections[k + 1])))
            if xl <= xr:
                image[y, xl:xr + 1] = color


# =====================================================================
#  5. Text Placement  (built-in 5×7 pixel font)
# =====================================================================

# A minimal 5-wide × 7-tall bitmap font for printable ASCII (32–126).
# Each character is encoded as 7 bytes (rows top to bottom), where each
# byte stores 5 bits (columns left to right, MSB = leftmost).
# This font is deliberately compact — it covers all printable ASCII so
# that any text string can be rendered without external dependencies.

_FONT_5x7 = {
    ' ':  [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    '!':  [0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04],
    '"':  [0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00],
    '#':  [0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A],
    '$':  [0x04, 0x0F, 0x14, 0x0E, 0x05, 0x1E, 0x04],
    '%':  [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
    '&':  [0x08, 0x14, 0x14, 0x08, 0x15, 0x12, 0x0D],
    "'":  [0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00],
    '(':  [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
    ')':  [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
    '*':  [0x04, 0x15, 0x0E, 0x1F, 0x0E, 0x15, 0x04],
    '+':  [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
    ',':  [0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08],
    '-':  [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
    '.':  [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04],
    '/':  [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
    '0':  [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
    '1':  [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
    '2':  [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
    '3':  [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
    '4':  [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
    '5':  [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
    '6':  [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
    '7':  [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
    '8':  [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
    '9':  [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
    ':':  [0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00],
    ';':  [0x00, 0x00, 0x04, 0x00, 0x04, 0x04, 0x08],
    '<':  [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
    '=':  [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
    '>':  [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
    '?':  [0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04],
    '@':  [0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0E],
    'A':  [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    'B':  [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
    'C':  [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
    'D':  [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
    'E':  [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
    'F':  [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
    'G':  [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
    'H':  [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    'I':  [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
    'J':  [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
    'K':  [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
    'L':  [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
    'M':  [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
    'N':  [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
    'O':  [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    'P':  [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
    'Q':  [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
    'R':  [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
    'S':  [0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E],
    'T':  [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
    'U':  [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    'V':  [0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04],
    'W':  [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
    'X':  [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
    'Y':  [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
    'Z':  [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
    '[':  [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E],
    '\\': [0x10, 0x08, 0x08, 0x04, 0x02, 0x02, 0x01],
    ']':  [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E],
    '^':  [0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00],
    '_':  [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
    '`':  [0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00],
    'a':  [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F],
    'b':  [0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E],
    'c':  [0x00, 0x00, 0x0E, 0x11, 0x10, 0x11, 0x0E],
    'd':  [0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F],
    'e':  [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
    'f':  [0x06, 0x08, 0x08, 0x1E, 0x08, 0x08, 0x08],
    'g':  [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E],
    'h':  [0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x11],
    'i':  [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
    'j':  [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],
    'k':  [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
    'l':  [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
    'm':  [0x00, 0x00, 0x1A, 0x15, 0x15, 0x15, 0x11],
    'n':  [0x00, 0x00, 0x1E, 0x11, 0x11, 0x11, 0x11],
    'o':  [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],
    'p':  [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10],
    'q':  [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x01],
    'r':  [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
    's':  [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E],
    't':  [0x08, 0x08, 0x1E, 0x08, 0x08, 0x09, 0x06],
    'u':  [0x00, 0x00, 0x11, 0x11, 0x11, 0x11, 0x0F],
    'v':  [0x00, 0x00, 0x11, 0x11, 0x0A, 0x0A, 0x04],
    'w':  [0x00, 0x00, 0x11, 0x15, 0x15, 0x15, 0x0A],
    'x':  [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],
    'y':  [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E],
    'z':  [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],
    '{':  [0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02],
    '|':  [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
    '}':  [0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08],
    '~':  [0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00],
}

_FONT_W = 5
_FONT_H = 7


def put_text(image, text, x, y, color, *, scale=1, spacing=1):
    """Render text onto the image using a built-in 5×7 pixel font.

    Parameters
    ----------
    image : numpy.ndarray
        Canvas image (modified in-place).
    text : str
        The string to render.  All printable ASCII characters (32–126)
        are supported.  Unsupported characters are rendered as a solid
        5×7 block.
    x : int
        Column position of the top-left corner of the first character.
    y : int
        Row position of the top-left corner of the first character.
    color : int or tuple
        Text color.
    scale : int, optional
        Integer scaling factor (default 1).  At ``scale=2`` each font
        pixel becomes a 2×2 block, so characters are 10×14 pixels.
    spacing : int, optional
        Extra pixels between characters (default 1).

    Returns
    -------
    numpy.ndarray
        The same *image* array (modified in-place).

    Raises
    ------
    TypeError
        If *text* is not a string.
    ValueError
        If *scale* < 1.

    Notes
    -----
    The built-in font is a classic 5×7 monospaced bitmap covering all
    printable ASCII characters.  Each character is stored as 7 bytes
    (one per row), with 5 bits per byte encoding the pixel columns.
    This eliminates any dependency on PIL, FreeType, or external font
    files.

    The rendering is done by pre-computing all "on" pixel coordinates
    for the entire string, then writing them in a single vectorized
    call to the canvas — so the cost is proportional to the number of
    "on" pixels, not the canvas size.

    Examples
    --------
    >>> put_text(canvas, "Hello MiniCV!", x=10, y=10, color=255, scale=2)
    """
    _CALLER = "put_text"
    validate_image_array(image, allow_float=True, caller=_CALLER)
    color = _resolve_color(image, color, _CALLER)

    if not isinstance(text, str):
        raise TypeError(
            f"[{_CALLER}] 'text' must be str, got {type(text).__name__}."
        )
    if not isinstance(scale, (int, np.integer)):
        raise TypeError(
            f"[{_CALLER}] 'scale' must be int, got {type(scale).__name__}."
        )
    scale = int(scale)
    if scale < 1:
        raise ValueError(f"[{_CALLER}] 'scale' must be ≥ 1, got {scale}.")

    x, y = int(x), int(y)
    spacing = int(spacing)

    char_step = (_FONT_W + spacing) * scale

    # Pre-compute all "on" pixel coordinates for the whole string
    all_ys = []
    all_xs = []

    for ci, ch in enumerate(text):
        bitmap = _FONT_5x7.get(ch, [0x1F] * _FONT_H)  # fallback = block
        cx = x + ci * char_step

        for row in range(_FONT_H):
            bits = bitmap[row]
            for col in range(_FONT_W):
                if bits & (1 << (_FONT_W - 1 - col)):
                    # This font pixel is "on" — expand by scale
                    base_y = y + row * scale
                    base_x = cx + col * scale
                    for sy in range(scale):
                        for sx in range(scale):
                            all_ys.append(base_y + sy)
                            all_xs.append(base_x + sx)

    if all_ys:
        _set_pixels_bulk(
            image,
            np.array(all_ys, dtype=np.intp),
            np.array(all_xs, dtype=np.intp),
            color,
        )

    return image

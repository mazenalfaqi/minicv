"""
MiniCV — A lightweight, from-scratch image-processing library.

Built on top of NumPy, Pandas, Matplotlib, and the Python standard library.
Emulates a well-defined subset of OpenCV for educational and research purposes.

Modules
-------
io          : Image reading, writing, and color-space conversion.
filtering   : Spatial filters (mean, Gaussian, median) and convolution engine.
transforms  : Geometric transforms (resize, rotate, translate).
features    : Global and gradient-based feature descriptors.
drawing     : Canvas primitives (point, line, rectangle, polygon, text).
utils       : Shared helpers — validation, padding, normalization, clipping.
"""

__version__ = "0.1.0"
__author__ = "MiniCV Team"

# ---------------------------------------------------------------------------
# Convenience re-exports so users can write:
#     import minicv as mcv
#     img = mcv.read_image("photo.png")
# ---------------------------------------------------------------------------
from minicv.io.reader import read_image                    # noqa: F401
from minicv.io.writer import export_image                   # noqa: F401
from minicv.io.color import convert_color                   # noqa: F401

from minicv.utils.normalization import normalize            # noqa: F401
from minicv.utils.clipping import clip_image                # noqa: F401
from minicv.utils.padding import pad                        # noqa: F401

from minicv.filtering.convolution import convolve2d         # noqa: F401
from minicv.filtering.convolution import spatial_filter     # noqa: F401

from minicv.filtering.filters import mean_filter            # noqa: F401
from minicv.filtering.filters import gaussian_kernel        # noqa: F401
from minicv.filtering.filters import gaussian_filter        # noqa: F401
from minicv.filtering.filters import median_filter          # noqa: F401
from minicv.filtering.filters import sobel_x, sobel_y      # noqa: F401
from minicv.filtering.filters import sobel                  # noqa: F401

from minicv.filtering.processing import threshold_global            # noqa: F401
from minicv.filtering.processing import threshold_otsu              # noqa: F401
from minicv.filtering.processing import threshold_adaptive          # noqa: F401
from minicv.filtering.processing import bit_plane_slice             # noqa: F401
from minicv.filtering.processing import histogram                   # noqa: F401
from minicv.filtering.processing import histogram_equalization      # noqa: F401
from minicv.filtering.processing import unsharp_mask                # noqa: F401
from minicv.filtering.processing import laplacian                   # noqa: F401

from minicv.transforms.geometric import resize                     # noqa: F401
from minicv.transforms.geometric import rotate                     # noqa: F401
from minicv.transforms.geometric import translate                  # noqa: F401

from minicv.features.descriptors import color_histogram_descriptor # noqa: F401
from minicv.features.descriptors import statistical_moments        # noqa: F401
from minicv.features.descriptors import hog_descriptor             # noqa: F401
from minicv.features.descriptors import edge_descriptor            # noqa: F401

from minicv.drawing.primitives import draw_point                   # noqa: F401
from minicv.drawing.primitives import draw_line                    # noqa: F401
from minicv.drawing.primitives import draw_rectangle               # noqa: F401
from minicv.drawing.primitives import draw_polygon                 # noqa: F401
from minicv.drawing.primitives import put_text                     # noqa: F401

# MiniCV

A lightweight, from-scratch image-processing library that emulates a 
subset of OpenCV — built using only NumPy, Matplotlib, Pandas, and 
the Python standard library.

## Installation

```bash
pip install git+https://github.com/YOUR_USERNAME/minicv.git
```

## Quick Start

```python
import minicv

img  = minicv.read_image("photo.png")
gray = minicv.convert_color(img, 'rgb2gray')
blur = minicv.gaussian_filter(gray, ksize=5, sigma=1.5)
minicv.export_image(blur, "output.png")
```

## Package Structure

| Module | What it does |
|---|---|
| `minicv.io` | Read, write, color conversion |
| `minicv.utils` | Normalize, clip, pad |
| `minicv.filtering` | Convolution, filters, thresholding, histogram |
| `minicv.transforms` | Resize, rotate, translate |
| `minicv.features` | HOG, histogram, moments descriptors |
| `minicv.drawing` | Point, line, rectangle, polygon, text |

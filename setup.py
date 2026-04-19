from setuptools import setup, find_packages

setup(
    name="minicv",
    version="0.1.0",
    description="A lightweight from-scratch image-processing library.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy", "matplotlib", "pandas"],
)

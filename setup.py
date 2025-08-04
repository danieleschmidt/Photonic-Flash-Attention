#!/usr/bin/env python3
"""Setup script for Photonic Flash Attention library."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="photonic-flash-attention",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.ai",
    description="Hybrid photonic-electronic Flash-Attention implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/photonic-mlir-synth-bridge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0,<2.0.0",
        "psutil>=5.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
            "pre-commit>=2.17.0",
        ],
        "benchmark": [
            "transformers>=4.20.0",
            "datasets>=2.0.0",
            "wandb>=0.13.0",
            "tensorboard>=2.10.0",
        ],
        "hardware": [
            "pynvml>=11.4.0",
            "py3nvml>=0.2.7",
        ],
        "simulation": [
            "cupy-cuda12x>=12.0.0",
            "numba>=0.56.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "photonic-benchmark=photonic_flash_attention.cli:benchmark",
            "photonic-calibrate=photonic_flash_attention.cli:calibrate",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
# ViScore

ViScore is a toolkit for evaluation of dimensionality reduction.

**It is published together with [ViVAE](https://github.com/saeyslab/ViVAE), a toolkit for single-cell data denoising and dimensionality reduction.**

## Installation

ViScore is a Python package.
We recommend creating a new Anaconda environment for ViScore.

On Linux or macOS, use the command line for installation.
On Windows, use Anaconda Prompt.

```
conda create --name ViScore python=3.9 \
    numpy numba scikit-learn scipy pynndescent matplotlib pyemd
pip install --upgrade git+https://github.com/saeyslab/ViScore.git
```

## Usage

ViScore uses unsupervised scores for assessing local and global structure preservation in low-dimensional (LD) embeddings of high-dimensional (HD) data.
If working with labelled data, supervised evaluation metrics can be used to elucidate source of error (shape and positional distortion).

See documentation for `ViScore.score`, `ViScore.xnpe`, `ViScore.neighbourhood_composition` and `ViScore.neighbourhood_composition_plot`.


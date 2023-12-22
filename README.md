# ViScore

ViScore is a toolkit for evaluation of dimensionality reduction.

**It is published together with [ViVAE](https://github.com/saeyslab/ViVAE), a toolkit for single-cell data denoising and dimensionality reduction.**

## Installation

ViScore is a Python package.
We recommend creating a new Anaconda environment for ViScore, or using the one you may have already created for *ViVAE*.

On Linux or macOS, use the command line for installation.
On Windows, use Anaconda Prompt.

*(A test install run on 2020 MacBook Air runs under 1 minutes.)*

### Stand-alone *ViScore* installation

```
conda create --name ViScore --channel conda-forge python=3.9 \
    numpy==1.22.4 numba==0.58.1 scikit-learn==1.3.2 scipy==1.11.4 pynndescent==0.5.11 matplotlib==3.8.2 pyemd==1.0.0
conda activate ViScore
pip install --upgrade git+https://github.com/saeyslab/ViScore.git
```

### Joint environment for *ViVAE* and *ViScore*

Follow [installation instructions for *ViVAE*](https://github.com/saeyslab/ViVAE#installation) first.

Then, assuming your Anaconda environment is named `ViVAE`, run the following.
```
conda activate ViVAE
conda install -c conda-forge pyemd==1.0.0
pip install --upgrade git+https://github.com/saeyslab/ViScore.git
```

## Usage

ViScore uses unsupervised scores for assessing local and global structure preservation in low-dimensional (LD) embeddings of high-dimensional (HD) data.
If working with labelled data, supervised evaluation metrics can be used to elucidate source of error (shape and positional distortion).

See documentation for `ViScore.score`, `ViScore.xnpe`, `ViScore.neighbourhood_composition` and `ViScore.neighbourhood_composition_plot`.

## Pre-print

The pre-print of our publication is available [here](https://www.biorxiv.org/content/10.1101/2023.11.23.568428v2) on bioRxiv.

It describes underlying methodology of ViVAE and ViScore, reviews past work in dimensionality reduction and evaluation of it and links to publicly available datasets on which performance of ViVAE was evaluated.

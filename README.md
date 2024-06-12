# ViScore

*[David Novak](https://github.com/davnovak), Cyril de Bodt, Pierre Lambert, John A. Lee, Sofie Van Gassen, Yvan Saeys*

ViScore is a toolkit for evaluation of dimensionality reduction.

**It is published together with [ViVAE](https://github.com/saeyslab/ViVAE), a toolkit for single-cell data denoising and dimensionality reduction.**

## Installation

ViScore is a Python package.
We recommend creating a new Anaconda environment for ViScore, or using the one you may have already created for *ViVAE*.

On Linux or macOS, use the command line for installation.
On Windows, use Anaconda Prompt.

*(A test install run on 2020 MacBook Air runs for under 1 minute.)*

```
conda create --name ViScore --channel conda-forge python=3.9 \
    numpy==1.22.4 numba==0.58.1 scikit-learn==1.3.2 scipy==1.11.4 pynndescent==0.5.11 matplotlib==3.8.2 pyemd==1.0.0
conda activate ViScore
pip install --upgrade git+https://github.com/saeyslab/ViScore.git
```

## Usage

ViScore uses unsupervised scores for assessing local and global structure preservation in low-dimensional (LD) embeddings of high-dimensional (HD) data.
If working with labelled data, supervised evaluation metrics can be used to elucidate source of error (shape and positional distortion).

See documentation for `ViScore.score`, `ViScore.xnpe`, `ViScore.neighbourhood_composition` and `ViScore.neighbourhood_composition_plot`.

## Objective structure-preservation scoring

ViScore enables unsupervised assessment of structure preservation in LD embeddings of HD data using scores based on $R_{NX}$ curves.
This is an objective approach based on quantifying neighbourhood preservation between HD and LD for all neighbourhood scales.

RNX curves show (scaled) overlap between neighbour ranks for all neighbourhoods of size from 1 to N-1.

![RNX curve illustration](./rnx_curve_plot.png)

- Taking the AUC (Area-Under-Curve) with *logarithmic* scale for *K* (neighbourhood size), we effectively up-weight the significance of local neighbourhoods, *without* setting a hard cut-off for what is still considered local. This is the **local structure-preservation score** $S_{L}$.

- Taking the AUC with linear scale for *K*, we dispense with the locality bias and assume equal importance for all neighbourhood scales. This is the **global structure-preservation score** $S_{G}$.

Since the computation of an $R_{NX}$ curve has quadratic complexity, this approach is largely impractical impossible to apply to larger single-cell datasets.
We circumvent this limitation by approximating the $R_{NX}$ curve using a repeated vantage point tree-based sampling approach.
This is implemented in `ViScore.score`.
The scRNA-seq example below includes an application of this.

## Example with scRNA-seq data

In our [online tutorial](https://colab.research.google.com/drive/1Ys9fpg8t4rhfmGHUVuX2JPdxQHfongpB?usp=sharing) for dimensionality reduction of biological (scRNA-seq) data using [ViVAE](https://github.com/saeyslab/ViVAE) we use ViScore to compute structure-preservation scores (see section 4) given a ViVAE and [UMAP](https://pypi.org/project/umap-learn/) embedding of the same dataset.

This tutorial contains instructions for running the workflow locally or remotely.
The user can adapt the code to use different dimensionality reduction tools, hyperparameters or datasets.

## Pre-print

The pre-print of our publication is available [here](https://www.biorxiv.org/content/10.1101/2023.11.23.568428v2) on bioRxiv.

It describes underlying methodology of ViVAE and ViScore, reviews past work in dimensionality reduction and evaluation of it and links to publicly available datasets on which performance of ViVAE was evaluated.

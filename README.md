<img src="./logo.png" alt="ViScore" width="250"/>

ViScore (*vee-score*) is a toolkit for evaluating and benchmarking dimensionality reduction.

It is published together with [ViVAE](https://github.com/saeyslab/ViVAE), a tool for single-cell data denoising and dimensionality reduction.
Check out the associated [paper](https://www.biorxiv.org/content/10.1101/2023.11.23.568428v3): *Interpretable models for scRNA-seq data embedding with multi-scale structure preservation*, where we describe and validate our methods in-depth.

<img src="./overview.png" alt="overview" width="900"/>

## Why use ViScore

* ViScore evaluates structure preservation (SP) by DR using [RNX curves](https://www.sciencedirect.com/science/article/abs/pii/S0925231215003641), independent of any labelling of points.
    * We extend RNX-based scoring to evaluate **Local** and **Global** SP separately.
    * We use accurate approximations of RNX curves to newly enable this scoring for large datasets without O(n^2) complexity, which had previously made its use prohibitive in many cases.
* ViScore enables supervised scoring of cell population embeddings using **xNPE**, an extension of the [Neighbourhood Proportion Error](https://github.com/akonstodata/NPE).
    * xNPE detects positional and shape distortion of labelled populations of points and compares them between embeddings.
* ViScore helps visualise distortions of populations using **neighbourhood composition plots**.
    * Neighbourhood composition plots compare the neighbourhoods of a population in terms of neighbourhood labels, comparing low-dimensional embeddings and the high-dimensional input data.
* We offer a **scalable benchmarking framework** powered by ViScore to compare DR methods on multiple datasets.
    * The framework is made for deployment on HPC clusters, fully documented and minimalist (independent of Snakemake or NextFlow).

## Installation

To try out ViScore without installing it locally, follow the tutorial on scRNA-seq data dimensionality reduction in the [ViVAE repository](https://github.com/saeyslab/ViVAE), which gives instructions on usage within [Google Colab](https://colab.research.google.com).

<hr>

ViScore is a Python package.
We recommend creating a new Anaconda environment for ViScore, or using the one you may have already created for *ViVAE*.

On Linux or macOS, use the command line for installation.
On Windows, use Anaconda Prompt.

<details>
<summary><b>Stand-alone installation</b></summary>
<br>

```bash
conda create --name ViScore --channel conda-forge python=3.11.7 \
    numpy==1.26.3 numba==0.59.0 matplotlib==3.8.2 scipy==1.12.0 pynndescent==0.5.11 scikit-learn==1.4.0 pyemd==1.0.0
conda activate ViScore
pip install --upgrade git+https://github.com/saeyslab/ViScore.git
```

<hr>
</details>

<details>
<summary><b>Shared environment with ViVAE</b></summary>
<br>

```bash
conda activate ViVAE
pip install pyemd==1.0.0
pip install --upgrade git+https://github.com/saeyslab/ViScore.git
```

</details>

## Usage

Examples of ViScore usage are shown in tutorials in the [ViVAE repository](https://github.com/saeyslab/ViVAE).

<hr>

* `ViScore.score` quantifies Local and Global SP without the use of labels (higher is better).
* `ViScore.xnpe` quantifies local distortion of labelled populations (lower is better).
* `ViScore.neighbourhood_composition_plot` shows sources of error in local embeddings of labelled populations.

Each of these functions is documented: for example, use `help(ViScore.score)` to find out more about Local and Global SP scoring.

<details>
<summary><b>Objective structure-preservation scoring</b></summary>
<br>

ViScore enables unsupervised assessment of structure preservation in LD embeddings of HD data using scores based on RNX curves.
This is an objective approach based on quantifying neighbourhood preservation between HD and LD for all neighbourhood scales.

RNX curves show (scaled) overlap between neighbour ranks for all neighbourhoods of size from 1 to N-1.

<img src="./rnx_curve_plot.png" alt="RNX curve" width="450"/>

* Taking the AUC (Area-Under-Curve) with *logarithmic* scale for *K* (neighbourhood size), we effectively up-weight the significance of local neighbourhoods, *without* setting a hard cut-off for what is still considered local. This is the **Local SP score** (SL).

* Taking the AUC with linear scale for *K*, we dispense with the locality bias and assume equal importance for all neighbourhood scales. This is the **Global SP score** (SG).

Both of these values are bounded by -1 and 1 (higher is better), where 0 corresponds to SP by a random embedding.

Since the computation of an RNX curve has quadratic complexity, this approach is impractical or impossible to apply to larger datasets.
We circumvent this by approximating the RNX curve using a repeated vantage point tree-based sampling approach.
This is implemented in `ViScore.score`.

</details>

## Benchmarking

You can find our documented benchmarking set-up for comparing DR methods on scRNA-seq data in the `benchmarking` folder in this repository.
It takes you through the process of setting up and deploying a benchmarking (or hyperparameter tuning) workflow on a high-performance computing (HPC) cluster.
The framework is extensible in terms of DR methods and datasets.

Additionally, code to generate figures and LaTeX tables for presenting results of your benchmark is included.

<img src="benchmarking/schematic.png" />

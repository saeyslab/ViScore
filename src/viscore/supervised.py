"""
xNPE is based on Neighbourhood Proportion Error: Konstorum, A., Jekel, N., Vidal,
E., Laubenbacher, R. (2018). Comparative Analysis of Linear and Nonlinear
Dimension Reduction Techniques on Mass Cytometry Data. bioRxiv preprint
doi:10.1101/273862

MIT License

Copyright (c) 2021 Anna Konstorum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Copyright 2024 David Novak

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from typing import Optional, Union, List
from pyemd import emd
import matplotlib.pyplot as plt
from .aux_knn import make_knn
import pynndescent

PALETTE = [
    'grey',    '#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6', '#A30059', '#7A4900', '#dedb8c', '#63FFAC', '#B79762', '#004D43', '#8FB0FF', '#997D87',
    '#5A0007', '#809693', '#1B4400', '#4FC601', '#3B5DFF', '#4A3B53', '#FF2F80', '#61615A', '#BA0900', '#6B7900', '#00C2A0', '#FFAA92', '#FF90C9', '#B903AA',
    '#D16100', '#DDEFFF', '#000035', '#7B4F4B', '#A1C299', '#300018', '#0AA6D8', '#013349', '#00846F', '#372101', '#FFB500', '#C2FFED', '#A079BF', '#CC0744',
    '#C0B9B2', '#C2FF99', '#001E09', '#00489C', '#6F0062', '#0CBD66', '#EEC3FF', '#456D75', '#B77B68', '#7A87A1', '#788D66', '#885578', '#FAD09F', '#FF8A9A',
    '#D157A0', '#BEC459', '#456648', '#0086ED', '#886F4C', '#34362D', '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9', '#FF913F', '#938A81', '#575329',
    '#00FECF', '#B05B6F', '#8CD0FF', '#3B9700', '#04F757', '#C8A1A1', '#1E6E00', '#7900D7', '#A77500', '#6367A9', '#A05837', '#6B002C', '#772600', '#D790FF',
    '#9B9700', '#549E79', '#FFF69F', '#201625', '#72418F', '#BC23FF', '#99ADC0', '#3A2465', '#922329', '#5B4534', '#FDE8DC', '#404E55', '#0089A3', '#CB7E98',
    '#A4E804', '#324E72', '#6A3A4C'
]

def max_val_in_dict(x: dict):
    """Get maximum value from `dict` object

    Returns max of values in a dictionary, ignoring `None` entries.

    Args:
        x (dict): Dictionary object.

    Returns:
        float: Maximum value in the dictionary.
    """
    x = np.array(list(x.values()))
    return np.max([xi for xi in x if xi is not None])

def simple_dist_mat(n: int):
    """Trivial distance matrix

    Computes a distance matrix for two discrete uniform distributions with `n` bins.

    Args:
        n (int): Bin count.

    Returns
    -------
        np.ndarray: Distance matrix.
    """
    if n < 1:
        raise ValueError('`n` must be a positive integer')
    x = np.hstack([np.arange(n-1, 0, step=-1), np.arange(0,n)])
    return np.array([x[np.arange(idx, idx+n)] for idx in np.arange(n-1, -1, step=-1)]).astype(np.float64)

def annot_to_int(annot: np.ndarray):
    """Encode string array as integer array

    Maps each unique string in an array (from `np.unique`) to an integer value.

    Args:
        annot (np.ndarray): String array.

    Returns
    -------
        np.ndarray: Integer array.
    """
    d = {}
    keys = np.unique(annot)
    for idx in range(len(keys)):
        d[keys[idx]] = idx
    return np.array([d[pop] for pop in annot])

def baseline_likeness_distributions(
    annot: np.ndarray,
    k:     int = 1000,
    pop:   Optional[str] = None
):
    """Compute baseline likeness distributions

    Used for random-baseline correction in computing the xNPE.

    Computes population-wise distributions of counts of same-population neighbours, under the assumption that coordinates of points are random.
    This is used to correct xNPE values against a random-embedding baseline.

    In practice, the distributions are computed by setting neighbourhood compositions in terms of population to the overall distribution of populations in the dataset.

    Args:
        annot (np.ndarray): String array of labels per point.
        k (int): Neighbourhood size.
        pop (str, optional): Population name if there is only one population of intersts. Defaults to None.

    Returns
    -------
        if `pop` specified:
            dict: Dictionary of baseline distributions (np.ndarray) per unique population in `annot`.
        else:
            np.ndarray: Baseline distribution for specified population.
    """
    annot_unique = np.unique(annot)
    n = len(annot)

    if pop is None:
        distr_per_pop = {}
        for p in annot_unique:
            p_count = np.sum(annot==p)
            p_proportion = p_count/n
            d = np.histogram(np.repeat(p_proportion*k, repeats=p_count), bins=k+1, range=(0, k))[0]
            ds = np.sum(d)
            d = np.array([x/ds for x in d])
            distr_per_pop.update({p: d})
        return distr_per_pop
    else:
        p_count = np.sum(annot==pop)
        p_proportion = p_count/n
        d = np.histogram(np.repeat(p_proportion*k, repeats=p_count), bins=k+1, range=(0, k))[0]
        ds = np.sum(d)
        d = np.array([x/ds for x in d])
        return d

def likeness_distributions(
    coords:       np.ndarray,
    annot:        np.ndarray,
    knn:          Optional[Union[np.ndarray, list]] = None,
    pop:          Optional[str] = None,
    k:            int = 1000,
    random_state: Optional[int] = None
):
    """Compute likeness distributions

    Used for comparing relative positions of populations between original data and embedding in computing the xNPE.

    Computes population-wise distributions of counts of same-population neighbours.

    Args:
        coords (np.ndarray): Row-wise coordinate matrix.
        annot (np.ndarray): String array of row labels for `coords`.
        knn (np.ndarray or list, optional): Corresponding k-nearest-neighbour graph object from `viscore.make_knn`. Defaults to None.
        pop (str, optional): Population name if there is only one population of intersts. Defaults to None.
        k (int, optional): Neighbourhood size. Defaults to 1000.
        random_state (int, optional): Random seed for k-NNG construction if no k-NNG is provided. Defaults to None.

    Returns
    -------
        if `pop` specified:
            dict: Dictionary of distributions (np.ndarray) per unique population in `annot`.
        else:
            np.ndarray: Distribution for specified population.
    """
    if knn is None:
        knn = make_knn(x=coords, fname=None, k=k, verbose=False, random_state=random_state)
        knn = knn[0]
        knn = knn.astype(int)
    else:
        if isinstance(knn, list):
            knn = knn[0]
        elif len(knn.shape)==3:
            knn = knn[0,:,:]
        knn = knn[:,range(k)]

    annot_int = annot_to_int(annot)
    annot_unique = np.unique(annot)

    ## Get counts of same-population neighbours for each point
    labs_per_row = annot_int[knn]
    same_per_row = [np.sum(lab==lab[0])-1 for lab in labs_per_row]
    same_per_row_by_pop = [
        np.array(same_per_row)[np.where(annot_int==pop)[0]] for pop in range(len(annot_unique))
    ]

    ## Get distribution of same-population counts for each population
    #distr_per_pop = [np.array([np.sum(same_per_row_by_pop[idx_pop] == idx_neighbour) for idx_neighbour in np.arange(0, k+1)]) for idx_pop in range(len(annot_unique))]
    distr_per_pop = [np.histogram(x, bins=k+1, range=(0, k))[0] for x in same_per_row_by_pop]
    distr_per_pop = [x/np.max([1, np.sum(x)]) for x in distr_per_pop]

    if pop is not None:
        idx = np.where(annot_unique == pop)[0][0]
        return distr_per_pop[idx]
    else:
        ## Convert to dictionary with entry for each population
        res = {}
        for idx in range(len(distr_per_pop)):
            res[annot_unique[idx]] = distr_per_pop[idx]

        return res

def xnpe(
    hd:                  np.ndarray,
    ld:                  Optional[np.ndarray],
    annot:               Union[np.ndarray, list],
    knn:                 Optional[Union[np.ndarray, list]] = None,
    knn_ld:              Optional[Union[np.ndarray, list]] = None,
    k:                   Optional[int] = None,
    baseline_correction: bool = True,
    exclude_pops:        List = [],
    reduce:              Optional[str] = None,
    random_state:        Optional[int] = None
):
    """Compute xNPE (Extended Neighbourhood-Proportion-Error) of an embedding

    xNPE is based on the Neighbourhood Propportion Error (NPE).
    While NPE calculates a single unbounded error value, xNPE calculates a bounded score for each labelled population separately.

    A score of 0 corresponds to perfect embedding and is the lower bound.
    If `baseline_correction` is used, a score of 1 corresponds to a random-baseline level of error.
    If it is not used, a score of 1 corresponds to worst possible embedding and is the upper bound.

    The distributions of same-population vs different-population neighbour counts for a neighbourhood of `k` (likeness distribution) are computed for each population.
    Population-level error is a function of the differences between likeness distributions in original, high-dimensional data `hd` and an embedding `ld`.
    This difference is computed using Earth mover's distance (instead of total variation distance used in NPE).

    To compare neighbourhood compositions per population in more detail, use `viscore.neighbourhood_composition` and `viscore.neighbourhood_composition_plot`.

    To return a single score instead of population scores (not recommended), set `reduce` to 'sum' or 'average'.

    Args:
        hd (np.ndarray): Row-wise coordinates of original high-dimensional data.
        ld (np.ndarray, optional): Row-wise coordinates of embedding. Defaults to None.
        annot (np.ndarray): String array of labels per row of `hd` and `ld`.
        knn (np.ndarray or list, optional): k-nearest-neighbour graph for `hd` from `viscore.make_knn`. Defaults to None.
        knn_ld (np.ndarray or list, optional): k-nearest-neighbour graph for `ld` from `viscore.make_knn`. Defaults to None.
        k (int, optional): Neighbourhood size, if not the minimum of neighbourhood sizes of `knn_hd` and `knn_ld`. Defaults to None.
        baseline_correction (bool, optional): Whether to re-scale errors s.t. 1 corresponds to a random embedding. Defaults to True.
        exclude_pops (list, optional): List of populations to exclude (i.e., include the points but treat as unlabelled). Defaults to empty.
        reduce (str, optional): Reduction method to return a single value ('sum', 'average' or None). Defaults to None.
        random_state (int, optional): Random seed for k-NNG construction if needed. Defaults to None.

    Returns
    -------
        if `reduce` is None:
            dict: Dictionary of xNPE scores per population.
        else:
            float: Single xNPE score.
    """

    if reduce is not None and reduce not in ['sum', 'average']:
        raise ValueError('Invalid value of `reduce`')

    if exclude_pops is not None and not isinstance(exclude_pops, list) and isinstance(exclude_pops, str):
        exclude_pops = [exclude_pops]

    ## Resolve k-NNG in HD
    if knn is not None:
        if isinstance(knn, list):
            knn = knn[0]
        elif len(knn.shape)==3:
            knn = knn[0,:,:]
        knn = knn.astype(int)

    ## Resolve k-NNG in LD
    if knn_ld is not None:
        if isinstance(knn_ld, list):
            knn_ld = knn_ld[0]
        elif len(knn_ld.shape)==3:
            knn_ld = knn_ld[0,:,:]
        knn_ld = knn_ld.astype(int)

    ## Resolve common nearest-neighbour count
    if k is None and knn is not None:
        k = knn.shape[1]-1

    ## Get distributions in HD and LD
    distr_hd = likeness_distributions(coords=hd, annot=annot, knn=knn, k=k, random_state=random_state)
    distr_ld = likeness_distributions(coords=ld, annot=annot, knn=knn_ld, k=k, random_state=random_state)

    if baseline_correction:
        ## Compute a baseline distribution for random embedding
        distr_ld_baseline = baseline_likeness_distributions(annot=annot, k=k)

    ## Compute the upper bound on error values
    distr_upper_bound = k # (np.linalg.norm(np.ones(k), ord=1))

    ## Eliminate non-interesting populations (unlabelled cells?)
    pops = np.unique(annot)
    if len(exclude_pops)>0:
        for pop in exclude_pops:
            del distr_hd[pop]
            del distr_ld[pop]
            pops = np.delete(pops, np.argwhere(pops == pop))

    ## Compute and re-scale EMDs per population
    d = {}
    for pop in pops:
        pop_score = emd(
            first_histogram =distr_hd[pop],
            second_histogram=distr_ld[pop],
            distance_matrix =simple_dist_mat(n=k+1)
        )
        pop_score /= distr_upper_bound
        if baseline_correction:
            baseline_score = emd(
                first_histogram =distr_hd[pop],
                second_histogram=distr_ld_baseline[pop],
                distance_matrix =simple_dist_mat(n=k+1)
            )
            baseline_score /= distr_upper_bound
            if baseline_score>0.:
                pop_score /= baseline_score
        d.update({pop: pop_score})

    if reduce is None:
        if len(exclude_pops)>0:
            for pop in exclude_pops:
                d[pop] = None
        return d
    elif reduce == 'sum':
        return np.sum(np.array(list(d.values())))
    elif reduce == 'average':
        return np.mean(np.array(list(d.values())))

def plot_xnpe_barplot(
    res:       list,
    res_names: list,
    palette:   list = PALETTE,
    figsize:   tuple = (6, 4),
    dpi:       int = 120,
    fname:     Optional[str] = None,
    show:      bool = True
):
    """Create a barplot of xNPE scores per population

    Plot xNPE (embedding error) values per population for one or more embeddings of the same dataset.

    The provided `res` object must be a list of xNPE scores per population (i.e., generated by `viscore.xnpe` with the `reduce` argument set to None).

    Args:
        res (list): List of xNPE results from one or more methods evaluated on the same dataset using the same settings.
        res_names (list): Names of methods corresponding to each item in `res`.
        palette (list, optional): Non-default colour palette per each labelled population in hex codes. Defaults to `viscore.PALETTE`.
        figsize (tuple, optional): Width and height of plot in inches. Defaults to (6, 4).
        dpi (int, optional): Pixel density per inch. Defaults to 120.
        fname (str, optional): Name of exported plot (PNG or SVG format). Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.

    Returns
    -------
        fig, ax: A `matplotlib.figure.Figure` object and `matplotlib.axes._axes.Axes` object created with `matplotlib.pyplot.subplots`.
    """
    n_res = len(res)
    pops = list(res[0].keys())

    if len(res_names)!=n_res:
        raise ValueError('`res` and `res_names` must be of same length')
    if n_res>1:
        for i in range(1, n_res):
            if list(res[i].keys())!=pops:
                raise ValueError('Dict names must be the same across all `res` items')

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = []
    x_name = []
    for i, pop in enumerate(pops):
        for j in range(n_res):##
            ax.bar(x=i*(n_res+1)+j, height=list(res[j].values())[i], color=palette[j], edgecolor='#363636', linewidth=1., label=res_names[j] if i==0 else None)
        x_pos.append(i*(n_res+1)+((n_res+1)/2-1))
        x_name.append(pop)
    ax.set_xticks(x_pos, x_name, rotation=90)
    ax.set_xlabel('Population')
    ax.set_ylabel('xNPE')
    leg = fig.legend()

    if fname is not None:
        fig.savefig(fname, dpi=dpi, bbox_extra_artists=(leg,), bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

def plot_xnpe_map(
    proj:       np.ndarray,
    annot:      np.ndarray,
    res:        np.ndarray,
    unassigned: Optional[Union[str,list]] = None,
    figsize:    tuple = (16, 12),
    point_size: Optional[int] = None,
    max_val:    Optional[float] = None,
    palette:    list = PALETTE,
    fname:      Optional[str] = None,
    dpi:        int = 120,
    show:       bool = True
):
    """Plot embedding points coloured by xNPE values

    Plots a 2-dimensional embedding with points coloured by xNPE values of corresponding populations.

    Args:
        proj (np.ndarray): Row-wise coordinate matrix of 2-d embedding.
        annot (np.ndarray): String array of labels per row of `proj`.
        unassigned (str or list, optional): Labels(s) in `annot` corresponding to unlabelled points. Defaults to None.
        figsize (tuple, optional): Width and height of plot in inches. Defaults to (16, 12).
        point_size (int, optional): Point size. If not given, determined automatically based on number of points. Defaults to None.
        max_val (float, optional): Upper limit for xNPE value scale (for aligning multiple plots side-by-side). Defaults to None.
        palette (list, optional): Non-default colour palette per each labelled population in hex codes. Defaults to `viscore.PALETTE`.
        fname (str, optional): Name of exported plot (PNG or SVG format). Defaults to None.
        dpi (int, optional): Pixel density per inch. Defaults to 120.
        show (bool, optional): Whether to display the plot. Defaults to True.

    Returns
    -------
        fig, ax: A `matplotlib.figure.Figure` object and `matplotlib.axes._axes.Axes` object created with `matplotlib.pyplot.subplots`.
    """

    ## Resolve point size
    if point_size is not None:
        s = point_size
    else:
        s = 0.15
        if proj.shape[0] < 10000:
            s = 0.5
        if proj.shape[0] < 7000:
            s = 0.8
        if proj.shape[0] < 5000:
            s = 1.0

    ## Arrange subplots
    fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [20, 20, 2]})
    fig.subplots_adjust(top=6.0)

    ## Create array of xNPE values per point (population-wise values)
    z = np.array([res[pop] for pop in annot])

    ## Plot unassigned cells
    if unassigned is not None:
        multiple = isinstance(unassigned, list) and len(unassigned) > 1
        idcs = np.argwhere(np.isin(annot, np.array(unassigned))).ravel()
        ax1.scatter(proj[idcs,0], proj[idcs,1], label = ', '.join(unassigned) if multiple else [unassigned][0], s=s, c='#c9c9c9')
        ax2.scatter(proj[idcs,0], proj[idcs,1], label = ', '.join(unassigned) if multiple else [unassigned][0], s=s, c='#c9c9c9')
        ann = np.delete(annot, idcs)
        p = np.delete(proj, idcs, axis=0)
        err = np.delete(z, idcs)
    else:
        ann = annot
        p = proj
        err = z

    ## Plot cells coloured by labels
    idx_pop = 0
    for pop in np.unique(ann):
        idcs = np.where(ann == pop)
        ax1.scatter(p[idcs,0], p[idcs,1], label = pop, s=s, c=palette[idx_pop])
        idx_pop += 1

    ## Plot cells coloured by population-wise xNPE values
    a2 = ax2.scatter(x=p[:,0], y=p[:,1], c=err, s=s, cmap='plasma')
    fig.tight_layout()

    ## Plot xNPE colour bar

    handles, labels = ax1.get_legend_handles_labels()
    leg = fig.legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    for handle in leg.legendHandles:
        handle.set_sizes([50.0])
    _ = fig.colorbar(a2, cax=cax)
    a2.set_clim(vmin=0, vmax=np.max(err) if max_val is None else max_val)

    ax1.set_title('Population labels')
    ax2.set_title('Population errors')

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    if fname is not None:
        fig.savefig(fname, dpi=dpi, bbox_extra_artists=(leg,), bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig, (ax1, ax2, cax)

def neighbourhood_composition(
    X:            np.ndarray,
    pop:          str,
    annot:        np.ndarray,
    k:            int = 1000,
    stepsize:     int = 10,
    exclude:      Optional[list] = None,
    random_state: Optional[int] = None
):
    """Compute neighbourhood composition of a population

    This function describes the relative position of a population of points with respect to other populations.

    For a query population `pop`, the composition of its neighbours in terms of other (reference) populations is computed.
    Near-neighbour labels are first determined for each point in `pop` as the `k` nearest neighbours, excluding points from `pop` itself.
    Then, the labels are aggregated across all points in `pop`, yielding a population-level distribution for neighbourhoods of size up to `k`.
    Changes in this distribution across neighbourhood rank are recorded using a sliding-window approach, parametrised by `stepsize`.

    For the nearest-neighbour look-up with exclusion of `pop`, a k-NN index is computed using PyNNDescent.

    Args:
        X (np.ndarray): Row-wise coordinate matrix.
        pop (str): Population name of interest.
        annot (np.ndarray): String array of population labels per row of `X`.
        k (int, optional): Maximum neighbourhood size. Defaults to 1000.
        stepsize (int, optional): Size of bin to aggregate over, in terms of neighbourhood rank. Defaults to 10.
        exclude (list, optional): List of populations to exclude from the count. Defaults to None.
        random_state (int, optional): Random seed for nearest-neighbour index computation. Defaults to None.

    Returns:
    -------
        str: Name of query population `pop`.
        np.ndarray: Names of reference populations.
        list: List of reference population counts per segment of size `stepsize`.
        list: List of reference population proportions per segment of size `stepsize`.
        np.ndarray: Starting neighbour rank indices of each segment.
    """

    ## Query points: self
    idcs_query = np.where(annot == pop)[0]

    ## Reference points: non-self
    idcs_ref     = np.arange(X.shape[0])
    idcs_exclude = np.array([])
    if exclude is not None:
        if not isinstance(exclude, list) and isinstance(exclude, str):
            exclude = [exclude]
        for e in exclude:
            idcs_exclude = np.append(idcs_exclude, np.where(annot == e)[0])
    idcs_exclude = np.append(idcs_exclude, idcs_query)
    idcs_ref     = np.delete(idcs_ref, idcs_exclude.astype(int))

    ## Identify nearest neighbours excluding self & debris
    X_query    = X[idcs_query]
    X_ref      = X[idcs_ref]
    annot_ref  = annot[idcs_ref]
    pops_ref   = np.unique(annot_ref)
    index      = pynndescent.NNDescent(X_ref, random_state=random_state)
    knn        = index.query(X_query, k=k)[0]

    ## Bin by neighbourhood ranks
    idcs_from = np.arange(start=0, stop=k, step=stepsize)
    idcs_to   = np.append(idcs_from[1:]-1, k)
    counts = []
    proportions = []
    for i,j in zip(idcs_from, idcs_to):
        nn            = annot_ref[knn[:,i:j].astype(int)].flatten()
        n             = len(nn)
        d_counts      = []
        d_proportions = []
        for pop_ref in pops_ref:
            count = len(np.where(nn == pop_ref)[0])
            d_counts.append(count)
            d_proportions.append(count/n)
        counts.append(np.array(d_counts))
        proportions.append(np.array(d_proportions))

    return pop, pops_ref, counts, proportions, idcs_from

def neighbourhood_composition_plot(
        nc:             Optional[Union[Union[np.ndarray, List], tuple]] = None,
        X:              Optional[np.ndarray] = None,
        pop:            Optional[str] = None,
        annot:          Optional[np.ndarray] = None,
        k:              Optional[int] = 1000,
        stepsize:       Optional[int] = 10,
        exclude:        Optional[list] = None,
        palette:        list = PALETTE,
        bbox_to_anchor: tuple = (1.8, 1.25),
        show:           bool = True
):
    """Plot neighbourhood composition of a population

    Plots the composition of a local neighbourhood of a population-of-interest in labelled data.
    Best done separately for high-dimensional data and a corresponding low-dimensional embedding thereof to identify distortion in the embedding.
    Inputs should be pre-computed using `viscore.neighbourhood_composition`: then arguments `X`, `pop`, `annot`, `k`, `stepsize`, `exclude` are not needed.

    Args:
        nc (list or tuple, optional): Neighbourhood composition object generated by `viscore.neighbourhood_composition`. Defaults to None.
        X (np.ndarray, optional): Row-wise coordinate matrix. Defaults to None.
        pop (str, optional): Population-of-interest name. Defaults to None.
        annot (np.ndarray, optional): String array of labels per row of `X`. Defaults to None.
        k (int, optional): Maximum neighbourhood size. Defaults to 1000.
        stepsize (int, optional): Size of bin to aggregate over. Defaults to 10.
        exclude (list, optional): List of populations to exclude from the count. Defaults to None.
        palette (list, optional): Non-default colour palette per each labelled population in hex codes. Defaults to `viscore.PALETTE`.
        bbox_to_anchor (tuple, optional): Argument for matplotlib legend position. Defaults to (1.8, 1.25).
        show (bool, optional): Whether to display the plot. Defaults to True.

    Returns
    -------
        fig, ax: A `matplotlib.figure.Figure` object and `matplotlib.axes._axes.Axes` object created with `matplotlib.pyplot.subplots`.
    """

    if nc is None:
        ## Count populations in near neighbourhood
        pop_query, pops_ref, _, proportions, x = neighbourhood_composition(
            X=X, pop=pop, annot=annot, k=k, stepsize=stepsize, exclude=exclude
        )
    else:
        pop_query, pops_ref, _, proportions, x = nc

    ## Create stacked area plot
    fig, ax = plt.subplots()

    y = np.vstack(proportions)
    ax.stackplot(x, y.T, labels=pops_ref, colors=palette)
    ax.legend(bbox_to_anchor=bbox_to_anchor)

    setattr(ax, 'pop_query', pop_query)

    if show:
        plt.show()
    else:
        plt.close()

    return fig,ax

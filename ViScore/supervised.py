"""
Based on Neighbourhood Proportion Error: Konstorum, A., Jekel, N., Vidal, E., Laubenbacher, R. (2018). Comparative Analysis of Linear and Nonlinear
Dimension Reduction Techniques on Mass Cytometry Data. bioRxiv preprint doi:10.1101/273862

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
Extended Neighbourhood Proportion Error:

Copyright 2023 David Novak

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
from typing import Optional, Union
from pyemd import emd
import matplotlib as mpl
import matplotlib.pyplot as plt
from .aux_knn import make_knn
import pynndescent
import copy

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
    """
    Get maximum value from the values in a `dict` object, ignoring entries of `None`
    
    - x: a `dict`
    """
    x = np.array(list(x.values()))
    return np.max([xi for xi in x if xi is not None])

def simple_dist_mat(n: int):
    """
    Distance matrix for two discrete distributions of same length with levels spaced out uniformly

    - n: distribution length (int)
    """
    if n < 1:
        raise ValueError('`n` must be a positive integer')
    x = np.hstack([np.arange(n-1, 0, step=-1), np.arange(0,n)])
    return np.array([x[np.arange(idx, idx+n)] for idx in np.arange(n-1, -1, step=-1)]).astype(np.float64)

def annot_to_int(annot: np.ndarray):
    """
    Convert array of strings to array of mapped integer values

    - annot: array of strings (np.ndarray)
    """
    d = {}
    keys = np.unique(annot)
    for idx in range(len(keys)):
        d[keys[idx]] = idx
    return np.array([d[pop] for pop in annot])

def likeness_distributions(
    coords: np.ndarray,
    annot:  np.ndarray,
    knn:    Optional[Union[np.ndarray, list]],
    pop:    Optional[str] = None,
    k:      int = 50
):
    """
    Compute population-wise distributions of counts of nearest neighbours
    to each vantage point that have the same label as the vantage point

    - coords: matrix of row-wise coordinates (np.ndarray)
    - annot:  corresponding array of string labels per row of `coords` (np.ndarray)
    - knn:    optional k-NN indices of `coords`, created with `ViScore.make_knn` (list/np.ndarray)
    - pop:    optional name of population if there is only one population of interest (str)
    - k:      neighbourhood size (int)
    """
    if knn is None:
        knn = make_knn(x=coords, fname=None, k=k, verbose=False)
        knn = knn[0]
        knn = knn.astype(int)
    else:
        if isinstance(knn, list):
            knn = knn[0]
        elif len(knn.shape) == 3:
            knn = knn[0,:,:]
        knn = knn[:,range(k)]

    annot_int = annot_to_int(annot)
    annot_unique = np.unique(annot)

    ## Get counts of same-population neighbours for each point
    labs_per_row = annot_int[knn]
    same_per_row = [np.sum(lab==lab[0])-1 for lab in labs_per_row]
    same_per_row_per_pop = [np.array(same_per_row)[np.where(annot_int==pop)[0]] for pop in range(len(annot_unique))]
    
    ## Get distribution of same-population counts for each population
    distr_per_pop = [np.array([np.sum(same_per_row_per_pop[idx_pop] == idx_neighbour) for idx_neighbour in np.arange(0, k+1)]) for idx_pop in range(len(annot_unique))]
    distr_per_pop = [x / np.max([1, np.sum(x)]) for x in distr_per_pop]
    
    if pop is not None:
        idx = np.where(annot_unique == pop)[0][0]
        return distr_per_pop[idx]
    else:
        ## Convert to dictionary with entry for each population
        res = {}
        for idx in range(len(distr_per_pop)):
            res[annot_unique[idx]] = distr_per_pop[idx]

        return res

def neighbourhood_composition(
        X:        np.ndarray,
        pop:      str,
        annot:    np.ndarray,
        k:        int = 1000,
        stepsize: int = 10,
        exclude:  Optional[list] = None
):
    """
    Compute neighbourhood composition of a population in terms of labelled neighbours

    This function is called by `ViScore.neighbourhood_composition_plot`.

    Gets counts of points from labelled populations that are in a near
    neighbourhood of a chosen population, binning by a chosen stepsize.

    Returns (1) query population name, (2) reference population names,
    (3) array of counts per population per segment, (4) array of proportions
    per population per segment and (5) bin indices.

    - X:        coordinate matrix (np.ndarray)
    - pop:      single population of interest (str)
    - annot:    labels per point (np.ndarray)
    - k:        neighbourhood size (int)
    - stepsize: size of bin to aggregate over (int)
    - exclude:  optional populations to exclude from the count (list)
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
    
    ## Identify nearest neighbours exluding self & debris
    X_query    = X[idcs_query]
    X_ref      = X[idcs_ref]
    annot_ref  = annot[idcs_ref]
    pops_ref   = np.unique(annot_ref)
    index      = pynndescent.NNDescent(X_ref)
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
        nc:             Optional[np.ndarray] = None,
        X:              Optional[np.ndarray] = None,
        pop:            Optional[str] = None,
        annot:          Optional[np.ndarray] = None,
        k:              Optional[int] = 1000,
        stepsize:       Optional[int] = 10,
        exclude:        Optional[list] = None,
        ax =            None,
        palette:        list = PALETTE,
        bbox_to_anchor: tuple = (1.8,1.25)
):
    """
    Create neighbourhood composition plot

    Produce plot to inspect composition of the neighbourhood
    of a population of interest. This can be applied to any
    labelled data (high-dimensional or a low-dimensional embedding
    thereof). The population of interest itself is excluded
    from the nearest-neighbour search.

    - nc:             optional neighbourhood composition object generated by `ViScore.neighbourhood_composition`
    - X:              (if `nc` not provided) coordinate matrix (np.ndarray)
    - pop:            (if `nc` not provided) single population of interest (str)
    - annot:          (if `nc` not provided) labels per point (np.ndarray)
    - k:              (if `nc` not provided) neighbourhood size (int)
    - stepsize:       (if `nc` not provided) size of bin to aggregate over (int)
    - exclude:        (if `nc` not provided) optional populations to exclude from the count (list)
    - ax:             optional matplotlib subplot to use for the plotting (mpl.axes._subplots.AxesSubplot)
    - palette:        optional alternative colour palette (list)
    - figsize:        size of the resulting figure (tuple)
    - dpi:            pixel density per inch (int)
    - bbox_to_anchor: argument for matplotlib legend position (tuple)
    """

    if nc is None:
        ## Count populations in near neighbourhood
        pop_query, pops_ref, _, proportions, x = neighbourhood_composition(
            X=X, pop=pop, annot=annot, k=k, stepsize=stepsize, exclude=exclude
        )
    else:
        pop_query, pops_ref, _, proportions, x = nc

    ## Create stacked area plot
    if ax is None:
        ax = plt
    y = np.vstack(proportions)
    ax.stackplot(x, y.T, labels=pops_ref, colors=palette)
    ax.legend(bbox_to_anchor=bbox_to_anchor)

    setattr(ax, 'pop_query', pop_query)

    return ax

def xnpe(
    hd:           np.ndarray,
    ld:           np.ndarray,
    annot:        Union[np.ndarray, list],
    knn:          Union[np.ndarray, list],
    knn_ld:       Optional[np.ndarray] = None,
    k:            Optional[int] = None,
    exclude_pops: list = [],
    reduce:       Optional[str] = None
):
    """
    Compute the Extended Neighbourhood Proportion Error (xNPE)

    This is based on the Neighbourhood Propportion Error (NPE) but uses
    Earth mover's distance instead of total variation distance for computing
    differences between distributions of like-vs-unlike samples in local
    neighbourhoods. Additionally, the summed or averaged score (scalar) can
    be returned, or the unreduced vector of scores per sample population
    can be returned instead.

    - hd:           original (high-dimensional) coordinate matrix
    - ld:           projection (lower-dimensional) coordinate matrix
    - annot:        1-d array of string population labels per row of `hd`, `ld`
    - knn:          matrix of row-wise k-nearest-neighbour indices for `hd` (0-indexed and including self), or output of `make_knn` (list of index and distance matrix)
    - knn_ld:       optional k-NN indices for ld
    - k:            neighbourhood size (up to number of columns of knn minus one)
    - exclude_pops: list of populations names from annot to exclude from scoring
    - reduce:       reduction method for the returned value ('sum', 'average' or None)
    """
    
    if reduce is not None and reduce not in ['sum', 'average']:
        raise ValueError('Invalid value of `reduce`')

    if exclude_pops is not None and not isinstance(exclude_pops, list) and isinstance(exclude_pops, str):
        exclude_pops = [exclude_pops]

    if isinstance(knn, list):
        knn = knn[0]
    elif len(knn.shape) == 3:
        knn = knn[0,:,:]
    knn = knn.astype(int)

    if knn_ld is not None:
        if isinstance(knn_ld, list):
            knn_ld = knn_ld[0]
        elif len(knn_ld.shape) == 3:
            knn_ld = knn_ld[0,:,:]
        knn_ld = knn_ld.astype(int)

    if k is None:
        k = knn.shape[1]-1

    distr_hd = likeness_distributions(hd, annot, knn, k)
    distr_ld = likeness_distributions(ld, annot, knn_ld, k)

    pops = np.unique(annot)

    if len(exclude_pops) > 0:
        for pop in exclude_pops:
            del distr_hd[pop]
            del distr_ld[pop]
            pops = np.delete(pops, np.argwhere(pops == pop))

    d = {}
    for pop in pops:
        d[pop] = emd(first_histogram=distr_hd[pop], second_histogram=distr_ld[pop], distance_matrix=simple_dist_mat(n=k+1))

    if reduce is None:
        if len(exclude_pops) > 0:
            for pop in exclude_pops:
                d[pop] = None
        return d
    elif reduce == 'sum':
        return np.sum(np.array(list(d.values())))
    elif reduce == 'average':
        return np.mean(np.array(list(d.values())))

def plot_xnpe_map(
    proj:       np.ndarray,
    annot:      np.ndarray,
    res:        np.ndarray,
    unassigned: Optional[Union[str,list]] = None,
    figsize:    tuple = (16, 12),
    point_size: Optional[int] = None,
    max_val:    Optional[float] = None,
    palette:    list = [
            '#000000', '#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6', '#A30059',
            '#7A4900', '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF', '#997D87',
            '#5A0007', '#809693', '#1B4400', '#4FC601', '#3B5DFF', '#4A3B53', '#FF2F80',
            '#61615A', '#BA0900', '#6B7900', '#00C2A0', '#FFAA92', '#FF90C9', '#B903AA', '#D16100',
            '#DDEFFF', '#000035', '#7B4F4B', '#A1C299', '#300018', '#0AA6D8', '#013349', '#00846F',
            '#372101', '#FFB500', '#C2FFED', '#A079BF', '#CC0744', '#C0B9B2', '#C2FF99', '#001E09',
            '#00489C', '#6F0062', '#0CBD66', '#EEC3FF', '#456D75', '#B77B68', '#7A87A1', '#788D66',
            '#885578', '#FAD09F', '#FF8A9A', '#D157A0', '#BEC459', '#456648', '#0086ED', '#886F4C',
            '#34362D', '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9', '#FF913F', '#938A81',
            '#575329', '#00FECF', '#B05B6F', '#8CD0FF', '#3B9700', '#04F757', '#C8A1A1', '#1E6E00',
            '#7900D7', '#A77500', '#6367A9', '#A05837', '#6B002C', '#772600', '#D790FF', '#9B9700',
            '#549E79', '#FFF69F', '#201625', '#72418F', '#BC23FF', '#99ADC0', '#3A2465', '#922329',
            '#5B4534', '#FDE8DC', '#404E55', '#0089A3', '#CB7E98', '#A4E804', '#324E72', '#6A3A4C'
    ],
    fname:      Optional[str] = None,
    dpi:        int = 120
):
    """
    Plot xNPE (extended Neighbourhood Proportion Error) per population
    as a low-dimensional embedding overlay, along with population labels

    - proj:       2-dimensional embedding row-wise coordinate matrix (np.ndarray)
    - annot:      corresponding array of string labels per row of `proj` (np.ndarray)
    - unassigned: optional (list of) label(s) in `annot` that correspond to unassigned points (str/list)
    - res:        result of running `ViScore.xnpe` on `proj` with `reduce=None` (dict)
    - figsize:    tuple specifying width and height of plot in inches (tuple)
    - point_size: optional point size parameter (if not given, determined automatically based on the number of points) (int)
    - max_val:    optional upper limit for xNPE value scale (for aligning multiple plots) (float)
    - palette:    optional non-default palette of hex codes for colours per each labelled population (list)
    - fname:      optional name of file to save the figure (str)
    - dpi:        pixel density per inch if figure saved (int)
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
    fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [20, 20, 1]})
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
    l = fig.legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    for handle in l.legendHandles:
        handle.set_sizes([50.0])
    cb = fig.colorbar(a2, cax=cax)
    a2.set_clim(vmin=0, vmax=np.max(err) if max_val is None else max_val)

    ax1.set_title('Population labels')
    ax2.set_title('Population errors')

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    if fname is not None:
        fig.savefig(fname, dpi=dpi, bbox_extra_artists=(l,), bbox_inches='tight')

    plt.show()

"""
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

import os
import pynndescent # approximate k-NNG construction
import numpy as np
import copy
from typing import Optional,Union

def ensure_valid_knn(
    n:   int,
    knn: list
):
    """
    Check if `knn` is a valid k-NNG object

    Raise an error if it is not.

    - n:   number of points in the reference coordinate matrix (int)
    - knn: tentative k-NNG object (list)
    """

    if len(knn) != 2 or not isinstance(knn[0], np.ndarray) or not isinstance(knn[1], np.ndarray):
        raise ValueError('`knn` object must be a list of 2 np.ndarray objects (index matrix and distance matrix)')
    if isinstance(knn[0][0,0], (float, np.float32, np.float64)) and knn[0][0,0].is_integer():
        knn = [knn[0].astype(int), knn[1]]
    if knn[0].shape[0] != n or knn[1].shape[0] != n:
        raise ValueError('Neighbour and distance matrices of `knn` must have as many rows as the number of points in the reference coordinate matrix')
    if not isinstance(knn[0][0,0], (int, np.int32, np.int64)):
        print(type(knn[0][0,0]))
        raise ValueError('Neighbour matrix (`knn[0]`) must be an `np.ndarray` of `int`, `np.int32` or `np.int64` or coercible to integer')
    if not isinstance(knn[1][0,0], (float, np.float32, np.float64)):
        raise ValueError('Distance matrix (`knn[1]`) must be an `np.ndarray` of `float`, `np.float32` or `np.float64`')
    if not np.all([knn[0][idx,0]==idx for idx in range(n)]):
        knn = correct_knn_for_duplicates(knn)
    if not np.all(knn[1][:,0] == .0):
        raise ValueError('Neighbourhoods in `knn` must include self and neighbour indices must be 0-indexed')
    
    return knn

def correct_knn_for_duplicates(
    knn: list
):
    """
    Correct k-nearest-neighbour graph for duplicates

    k-NNGs built for data with duplicate points can result in wrong ordering of neighbour indices within rows (due
    to multiple points with zero distance). This function corrects for that, making sure that for each vantage point,
    the first index in the index matrix is itself.

    - knn: k-NNG object (list)
    """

    row_idcs = np.where([knn[0][idx,0]!=idx for idx in range(knn[0].shape[0])])[0]
    for i in row_idcs:
        idx_self = np.where(knn[0][i]==i)[0]
        knn[0][i][0], knn[0][i][idx_self] = knn[0][i][idx_self], knn[0][i][0]
    return knn


def make_knn(
    x:       Optional[np.ndarray] = None,
    fname:   Optional[str] = None,
    k:       int = 100,
    verbose: bool = True
):
    """
    Construct a k-nearest-neighbour graph

    This function uses the approximate algorithm implemented in `pynndescent` to construct a k-NNG.

    - x:       coordinate matrix (np.ndarray)
    - fname:   name of .npy file to save new k-NNG or to load an existing one (str/None)
    - k:       nearest neighbour count (int)
    - verbose: print message (whether k-NNG is constructed or loaded)? (bool)
    """

    if k < 1 or k > (x.shape[0]-1):
        raise ValueError('`k` must be between 1 and (n-1)')

    if fname is not None and os.path.isfile(fname):
        if verbose:
            print('Loading k-NNG')
        res = np.load(fname, allow_pickle=True)
        knn_idcs = np.array(res[0], dtype=np.int64)
        knn_dist = np.array(res[1])
        knn = [knn_idcs, knn_dist]
        return knn
    else:
        if verbose:
            print('Constructing k-NNG')
        knn_index = pynndescent.NNDescent(x, n_neighbors=k+1)
        knn_tuple = knn_index.query(x, k=k+1)
        knn = [knn_tuple[0].astype(np.int64), knn_tuple[1]]
        if not np.all([knn[0][idx,0]==idx for idx in range(x.shape[0])]):
            knn = correct_knn_for_duplicates(knn)
        if fname is not None:
            np.save(fname, knn)
        return knn

def smooth(
    x:      np.ndarray,
    knn:    list,
    k:      int = 50,
    coef:   Optional[float] = 0.1,
    n_iter: int = 1
):
    """
    Apply smoothing to data

    Applies the smoothing algorithm for denoising to a coordinate matrix. Given a k-NNG, each point is moved
    toward the average coordinates of its nearest neighbours.

    To apply a quicker, simple smoothing where each point is replaced with the average of it's neigbours coordinates
    (equivalent to `coef=1.`), `coef` can be set to `None`.

    - x:      coordinate matrix (np.ndarray)
    - knn:    list of k-NNG row-wise neighbour indices and neighbour distances (0-indexed and containing self) (list)
    - k:      nearest neighbour count to use (int)
    - coef:   lambda coefficient (size of shift toward average coordinates of neighbours, between 0. and 1.) or None (float)
    - n_iter: number of iterations to apply (int)
    """

    if k < 1 or k > (x.shape[0]-1):
        raise ValueError('`k` must be between 1 and (n-1)')
    if coef < 0. or coef > 1.:
        raise ValueError('`coef` must be more than 0. and at most 1.')
    if n_iter < 1 or n_iter > 10000:
        raise ValueError('`n_iter` must be at least 1 and at most 10000')
    knn = ensure_valid_knn(n=x.shape[0], knn=knn)

    y = copy.deepcopy(x)
    knn_idcs = knn[0].astype(np.int64)
    res = y
    for it in range(n_iter):
        for i in range(x.shape[0]):
            if coef is None:
                res[i] = y[knn_idcs[i][range(k)]].mean(0)
            else:
                target = y[knn_idcs[i][range(k)]].mean(0)
                vector = target - res[i]
                res[i] = res[i] + vector*coef
        y = res
    return res
    
def compute_lcmc(
    hd:     np.ndarray,
    ld:     np.ndarray,
    knn_hd: Union[np.ndarray, list],
    knn_ld: Optional[Union[np.ndarray, list]] = None,
    k:      int = 50
):
    """
    Compute the Local Continuity Meta-Criterion (LCMC)

    Computes the LCMC (a measure of local structure preservation in K-ary neighbourhood of each point).

    - hd:     high-dimensional (non-reduced) coordinate matrix (np.ndarray)
    - ld:     low-dimensional (reduced) coordinate matrix (np.ndarray)
    - knn_hd: k-nearest-neighbour graph of `hd`, as created by `make_knn` (list)
    - knn_ld: (optional) k-nearest-neighbour graph of `ld`, as created by `make_knn` (list)
    - k:      nearest neighbour count for LCMC computation
    """

    if k < 1 or k > (ld.shape[0]-1):
        raise ValueError('`k` must be between 1 and (n-1)')
    
    if isinstance(knn_hd, list):
        knn_hd = ensure_valid_knn(n=hd.shape[0], knn=knn_hd)
        knn_hd = knn_hd[0]
    if knn_ld is not None and isinstance(knn_ld, list):
        knn_ld = ensure_valid_knn(n=ld.shape[0], knn=knn_ld)
        knn_ld = knn_ld[0]

    knn = copy.deepcopy(knn_hd)
    knn = np.delete(knn, 0, axis=1)
    if knn_ld is None:
        knn_ld = make_knn(x=ld, k=k, verbose=False)[0]
        knn_ld = np.delete(knn_ld, 0, axis=1)
    k_hd = knn.shape[1]
    k_ld = knn_ld.shape[1]
    if k is None:
        k = np.min([k_hd,  k_ld])
    knn = knn[:,range(k)]
    knn_ld = knn_ld[:,range(k)]
    n = hd.shape[0]
    overlap = np.sum([np.intersect1d(x, y).size for x, y in zip(knn, knn_ld)])
    return k/(1-n)+1/(n*k)*overlap

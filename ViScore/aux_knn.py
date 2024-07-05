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


def make_knn(x: Optional[np.ndarray] = None, fname: Optional[str] = None, k: int = 100, as_tuple: bool = False, random_state: Optional[int] = None, verbose: bool = True) -> List:
    """Construct or load a k-NNG

    Construct a k-nearest-neighbour graph object with NN indices and coordinates.

    Args:
        x (Optional[np.ndarray], optional): Point coordinates to build k-NNG on top of. Defaults to None.
        fname (Optional[str], optional): Path to k-NNG file for saving/loading. Defaults to None.
        k (int, optional): Nearest neighbour count. Defaults to 100.
        as_tuple (bool, optional): Whether to return a tuple of NumPy arrays, instead of list. Defaults to False.
        random_state (int, optional): Random state for PyNNDescent to make the result reproducible. Default to None.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        if `as_tuple` is True:
            tuple: tuple of neighbour indices and distances.
        else:
            List: List of neighbour indices and distances.
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
        if as_tuple:
            knn = tuple(knn)
        return knn
    else:
        if verbose:
            print('Constructing k-NNG')
        knn_index = pynndescent.NNDescent(x, n_neighbors=k+1, random_state=random_state)
        knn_tuple = knn_index.query(x, k=k+1)
        knn = [knn_tuple[0].astype(np.int64), knn_tuple[1]]
        if not np.all([knn[0][idx,0]==idx for idx in range(x.shape[0])]):
            knn = correct_knn_for_duplicates(knn)
        if as_tuple:
            knn = tuple(knn)
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

def smooth(x: np.ndarray, knn: list, k: int = 50, coef: Optional[float] = 0.1, n_iter: int = 1) -> np.ndarray:
    """Denoise data by smoothing

    Args:
        x (np.ndarray): Point coordinates.
        knn (list): List of k-nearest-neighbour indices and distances.
        k (int, optional): Nearest neighbour count for smoothing. Defaults to 50.
        coef (Optional[float], optional): Smoothing lambda coefficient. Defaults to 0.1.
        n_iter (int, optional): Number of smoothing iterations. Defaults to 1.

    Returns:
        np.ndarray: Smoothed point coordinates.
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
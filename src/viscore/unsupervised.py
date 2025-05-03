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

import numpy as np
from .aux_fastqa import fast_eval_dr_quality, eucl_dist, cos_dist

def score(
    hd:     np.ndarray,
    ld:     np.ndarray,
    distf_hd: str = 'euclidean',
    distf_ld: str = 'euclidean'
) -> dict:
    """Compute embedding structure preservation

    Evaluates Structure Preservation (SP) by a lower-dimensional embedding of high-dimensional tabular data.
    SP is computed as preservation of neighbourhood ranks, assessed using a novel RNX curve approximation algorithm.

    The RNX values quantify preservation of neighbourhood ranks for different neighbourhood sizes, corrected against a random baseline.
    Local SP is the area-under-RNX-curve with smooth upweighting of local structure by log-scaling neighbourhood rank values.
    Global SP is the area-under-RNX-curve with linear scaling of neighbourhood rank values, resulting in a scale-agnostic score.

    Args:
        hd (np.ndarray): Row-wise coordinates of original high-dimensional data.
        ld (np.ndarray): Row-wise coordinates of reduced low-dimensional data.
        distf_hd (str): Distance function for high-dimensional data: 'euclidean' or 'cosine'. Defaults to 'euclidean'.
        distf_hd (str): Distance function for low-dimensional data: 'euclidean' or 'cosine'. Defaults to 'euclidean'.

    Returns:
        dict: Local SP as `Sl`, Global SP as `Sg` and RNX curve values as `RNX` (all `np.ndarray`).
    """
    if distf_hd=='euclidean':
        distf_hd = eucl_dist
    elif distf_hd=='cosine':
        distf_hd = cos_dist
    else:
        raise ValueError("distf_hd must be 'euclidean' or 'cosine'")
    
    if distf_ld=='euclidean':
        distf_ld = eucl_dist
    elif distf_ld=='cosine':
        distf_ld = cos_dist
    else:
        raise ValueError("distf_ld must be 'euclidean' or 'cosine'")

    s = fast_eval_dr_quality(
        X_hd    = hd,
        X_ld    = ld,
        dist_hd = distf_hd,
        dist_ld = distf_ld,
        n       = -1,
        seed    = 1,
        pow2K   = False,
        vp_samp = True
    )
    Sl = s[3]
    Sg = s[2]
    RNX = s[1]
    QNX = s[0]
    return {'Sl': Sl, 'Sg': Sg, 'RNX': RNX, 'QNX': QNX}


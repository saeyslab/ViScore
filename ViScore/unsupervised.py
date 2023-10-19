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
from .aux_knn import compute_lcmc
from .aux_fastqa import fast_eval_dr_quality, eucl_dist
from typing import Optional,Union

def score(
    hd:     np.ndarray,
    ld:     np.ndarray,
    knn_hd: Union[np.ndarray, list],
    knn_ld: Optional[Union[np.ndarray, list]] = None,
    k:      int = 50
) -> dict:
    """
    Compute structure preservation scores

    Computes the Local Continuity Meta-Criterion (LCMC) and the RNX-curve-based local (Sl) and global (Sg) structure-
    -preservation scores for a lower-dimensional embedding of a high-dimensional dataset.

    Sl is the area under the RNX curve, with neighbourhood size re-scaled logarithmically.

    Sg is the area under the RNX curve, with neighbourhood size scaled linearly.

    - hd:     high-dimensional (non-reduced) coordinate matrix (np.ndarray)
    - ld:     low-dimensional (reduced) coordinate matrix (np.ndarray)
    - knn_hd: k-nearest-neighbour graph of `hd`, as created by `make_knn` (list)
    - knn_ld: (optional) k-nearest-neighbour graph of `ld`, as created by `make_knn` (list)
    - k:      nearest neighbour count for LCMC computation
    """
    auc = fast_eval_dr_quality(
        X_hd    = hd,
        X_ld    = ld,
        dist_hd = eucl_dist,
        dist_ld = eucl_dist,
        n       = -1,
        seed    = 1,
        pow2K   = False,
        vp_samp = True
    )
    lcmc = compute_lcmc(hd=hd, ld=ld, knn_hd=knn_hd, knn_ld=knn_ld, k=k)
    Sl = auc[3]
    Sg = auc[2]
    return {'LCMC': lcmc, 'Sl': Sl, 'Sg': Sg}


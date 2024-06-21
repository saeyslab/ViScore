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
from .aux_fastqa import fast_eval_dr_quality, eucl_dist
from typing import Optional,Union

def score(
    hd:     np.ndarray,
    ld:     np.ndarray,
) -> dict:
    """
    Compute structure preservation scores

    Computes the RNX-curve-based local (Sl) and global (Sg) structure-preservation scores
    for a lower-dimensional embedding of a high-dimensional dataset. The RNX curve itself
    is also returned.

    Sl is the area under the RNX curve, with neighbourhood size re-scaled logarithmically.

    Sg is the area under the RNX curve, with neighbourhood size scaled linearly.


    - hd:     high-dimensional (non-reduced) coordinate matrix (np.ndarray)
    - ld:     low-dimensional (reduced) coordinate matrix (np.ndarray)

    Returns:
        List with 'Sl', 'Sg' and 'RNX' (all NumPy arrays).
    """
    s = fast_eval_dr_quality(
        X_hd    = hd,
        X_ld    = ld,
        dist_hd = eucl_dist,
        dist_ld = eucl_dist,
        n       = -1,
        seed    = 1,
        pow2K   = False,
        vp_samp = True
    )
    Sl = s[3]
    Sg = s[2]
    RNX = s[1]
    return {'Sl': Sl, 'Sg': Sg, 'RNX': RNX}


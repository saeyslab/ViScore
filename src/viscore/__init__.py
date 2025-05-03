"""ViScore: Dimensionality reduction evaluation toolkit"""

import logging
from importlib.metadata import version

logging.basicConfig(level=logging.INFO)

from .supervised import PALETTE, xnpe, plot_xnpe_map, plot_xnpe_barplot, max_val_in_dict, neighbourhood_composition, neighbourhood_composition_plot
from .unsupervised import score
from .aux_knn import make_knn, smooth

__all__ = [
    "PALETTE",
    "xnpe",
    "plot_xnpe_map",
    "plot_xnpe_barplot",
    "max_val_in_dict",
    "neighbourhood_composition",
    "neighbourhood_composition_plot",
    "score",
    "make_knn",
    "smooth"
]
__version__ = version("viscore")

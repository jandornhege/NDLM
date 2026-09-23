import torch
import torch.nn as nn

from .common import _SourceKGNNAdapter


class SourceThreeGNN(_SourceKGNNAdapter):
    """The repository's 3-Malkin k-GNN adapted to NDLM outputs."""

    tuple_order = 3

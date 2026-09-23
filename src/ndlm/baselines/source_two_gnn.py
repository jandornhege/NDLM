import torch
import torch.nn as nn

from .common import _SourceKGNNAdapter


class SourceTwoGNN(_SourceKGNNAdapter):
    """The repository's 2-Malkin k-GNN adapted to NDLM outputs."""

    tuple_order = 2

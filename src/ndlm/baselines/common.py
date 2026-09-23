import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


_BASELINE_ROOT = Path(__file__).resolve().parents[2] / "baseline_models"
_K_GNN_ROOT = _BASELINE_ROOT / "k-gnn"
_LEGACY_K_GNN_ROOT = Path(__file__).resolve().parents[2] / "k-gnn"


def _load_k_gnn_components():
    k_gnn_root = _K_GNN_ROOT if _K_GNN_ROOT.exists() else _LEGACY_K_GNN_ROOT
    if str(k_gnn_root) not in sys.path:
        sys.path.insert(0, str(k_gnn_root))
    try:
        from k_gnn import Complete, ConnectedThreeMalkin, GraphConv, TwoMalkin, ThreeMalkin, avg_pool
        from torch_geometric.data import Data
    except ImportError as error:
        raise ImportError(
            "The 2GNN/3GNN baselines require the dependencies in "
            "src/baseline_models/k-gnn "
            "(PyTorch Geometric, torch-scatter, and the built graph_cpu extension)."
        ) from error
    return Data, GraphConv, TwoMalkin, ThreeMalkin, ConnectedThreeMalkin, avg_pool, Complete


def _load_edge_transformer_module():
    module_path = _BASELINE_ROOT / "towards-principled-gts" / "edge_transformer.py"
    spec = importlib.util.spec_from_file_location("ndlm_edge_transformer", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the edge transformer module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relation_edge_index(roles):
    """Build the topology graph used by the Morris k-GNN.

    The original Morris GraphConv implementation only consumes a bare edge_index;
    it does not accept per-edge relation labels. We therefore construct the graph
    from the underlying connectivity and encode role information into node feature
    summaries instead of pretending each duplicate edge carries a distinct label.
    """
    if roles.dim() != 3:
        raise ValueError(f"Expected roles of shape [relations, objects, objects], got {tuple(roles.shape)}")

    adjacency = roles.abs().sum(dim=0) > 0
    edge_index = adjacency.nonzero(as_tuple=False).transpose(0, 1).contiguous()
    return edge_index


def _relation_aware_node_features(concepts, roles):
    """Attach relation summaries to each node to make the k-GNN input relation-aware.

    The NDLM convention is [concepts, objects]; the Morris k-GNN consumes
    [objects, features]. We normalize both layouts here before appending the
    incoming/outgoing role summary vectors.
    """
    if roles.dim() != 3:
        raise ValueError(f"Expected roles of shape [relations, objects, objects], got {tuple(roles.shape)}")

    if concepts.dim() != 2:
        raise ValueError(f"Expected concepts of shape [objects, features] or [features, objects], got {tuple(concepts.shape)}")

    if concepts.shape[0] != roles.shape[1] and concepts.shape[1] == roles.shape[1]:
        concepts = concepts.transpose(0, 1)

    if concepts.shape[0] != roles.shape[1]:
        raise ValueError(
            f"Concepts/object mismatch: concepts has {concepts.shape[0]} rows but roles has {roles.shape[1]} objects."
        )

    if roles.shape[0] == 0:
        return concepts

    incoming = roles.abs().sum(dim=2).transpose(0, 1)  # [objects, relations]
    outgoing = roles.abs().sum(dim=1).transpose(0, 1)  # [objects, relations]
    return torch.cat([concepts, incoming, outgoing], dim=-1)


def _tuple_members(assignment, tuple_order, tuple_count):
    members = [[] for _ in range(tuple_count)]
    for node_index, tuple_index in assignment.t().tolist():
        members[tuple_index].append(node_index)
    return [tuple(values[:tuple_order]) for values in members]


class _PairDecoder(nn.Module):
    def __init__(self, hidden_size, in_roles, out_roles):
        super().__init__()
        self.role_head = nn.Sequential(
            nn.Linear(hidden_size + in_roles, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_roles),
        )

    def forward(self, pair_features, roles):
        if roles.shape[1] > 0:
            pair_features = torch.cat([pair_features, roles.permute(0, 2, 3, 1)], dim=-1)
        return self.role_head(pair_features).permute(0, 3, 1, 2)


class _SourceKGNNAdapter(nn.Module):
    """Adapt the source k-GNN tuple graphs to NDLM's dense tensor contract."""

    tuple_order = None

    def __init__(self, in_concepts, in_roles, out_concepts, out_roles, hidden_size, num_layers, max_objects=None):
        super().__init__()
        Data, GraphConv, TwoMalkin, ThreeMalkin, _, avg_pool, _ = _load_k_gnn_components()
        self._Data = Data
        self._GraphConv = GraphConv
        self._avg_pool = avg_pool
        self._tuple_transform = TwoMalkin if self.tuple_order == 2 else ThreeMalkin
        self.in_roles = in_roles
        self.max_objects = max_objects
        self.node_input_dim = in_concepts + 2 * in_roles
        self.node_convs = nn.ModuleList(
            [GraphConv(self.node_input_dim if layer_index == 0 else hidden_size, hidden_size)
             for layer_index in range(num_layers)]
        )
        self.tuple_convs = nn.ModuleList(
            [GraphConv(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.node_head = nn.Linear(hidden_size, out_concepts)
        self.role_head = nn.Sequential(
            nn.Linear(hidden_size + in_roles, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_roles),
        )

    def _make_data(self, concepts, roles):
        edge_index = _relation_edge_index(roles)
        node_features = _relation_aware_node_features(concepts, roles)
        return self._Data(x=node_features, edge_index=edge_index)

    def _decode_pairs(self, tuple_features, assignment, node_features, roles):
        object_count = node_features.shape[0]
        pair_features = torch.cat([
            node_features[:, None, :].expand(-1, object_count, -1),
            node_features[None, :, :].expand(object_count, -1, -1),
        ], dim=-1)
        tuple_members = _tuple_members(
            assignment,
            self.tuple_order,
            tuple_features.shape[0],
        )
        pair_sum = torch.zeros_like(pair_features)
        pair_count = torch.zeros(
            object_count, object_count, 1,
            dtype=pair_features.dtype,
            device=pair_features.device,
        )

        for tuple_feature, members in zip(tuple_features, tuple_members):
            if len(members) < 2:
                continue
            if self.tuple_order == 2:
                ordered_pairs = [(members[0], members[1]), (members[1], members[0])]
            else:
                if len(members) != 3:
                    continue
                i, j, k = members
                ordered_pairs = [
                    (i, j), (j, i),
                    (i, k), (k, i),
                    (j, k), (k, j),
                ]
            for source, target in ordered_pairs:
                pair_sum[source, target] += tuple_feature
                pair_count[source, target] += 1

        pair_features = torch.where(
            pair_count > 0,
            pair_sum / pair_count.clamp_min(1),
            pair_features,
        )
        if self.in_roles > 0:
            pair_features = torch.cat([pair_features, roles.permute(1, 2, 0)], dim=-1)
        return self.role_head(pair_features).permute(2, 0, 1)

def forward(self, concepts, roles):
    outputs = []

    for concepts_one, roles_one in zip(concepts, roles):
        data = self._make_data(concepts_one, roles_one)
        data = self._tuple_transform()(data)

        node_features = data.x
        for convolution in self.node_convs:
            node_features = torch.relu(
                convolution(node_features, data.edge_index)
            )

        assignment = (
            data.assignment_index_2
            if self.tuple_order == 2
            else data.assignment_index_3
        )

        # Initial k-tuple representations from node representations
        tuple_features = self._avg_pool(node_features, assignment)

        # k-GNN message passing
        tuple_edge_index = (
            data.edge_index_2
            if self.tuple_order == 2
            else data.edge_index_3
        )

        for convolution in self.tuple_convs:
            tuple_features = torch.relu(
                convolution(tuple_features, tuple_edge_index)
            )

        # ---------------------------------------------------------
        # Pool final k-tuple representations back to nodes
        # ---------------------------------------------------------
        tuple_members = _tuple_members(
            assignment,
            self.tuple_order,
            tuple_features.shape[0],
        )

        object_count = node_features.shape[0]

        node_sum = torch.zeros_like(node_features)
        node_count = torch.zeros(
            object_count,
            1,
            dtype=node_features.dtype,
            device=node_features.device,
        )

        for tuple_feature, members in zip(tuple_features, tuple_members):
            for node in members:
                node_sum[node] += tuple_feature
                node_count[node] += 1

        kgnn_node_features = node_sum / node_count.clamp_min(1)

        # Concept prediction now uses the final k-GNN representation
        concept_logits = self.node_head(kgnn_node_features).transpose(0, 1)

        # Role prediction remains based on final k-tuple representations
        role_logits = self._decode_pairs(
            tuple_features,
            assignment,
            node_features,
            roles_one,
        )

        outputs.append((concept_logits, role_logits))

    return (
        torch.stack([output[0] for output in outputs]),
        torch.stack([output[1] for output in outputs]),
    )
import itertools
from types import SimpleNamespace

import torch
import torch.nn as nn

from .common import _load_k_gnn_components


class _FallbackGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.root_weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        bound = self.weight.shape[0] ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.root_weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features, edge_index):
        output = features @ self.root_weight
        if edge_index.numel() > 0:
            messages = features[edge_index[1]] @ self.weight
            degree = features.new_zeros(features.shape[0])
            degree.index_add_(0, edge_index[0], features.new_ones(edge_index.shape[1]))
            aggregated = output.new_zeros(output.shape)
            aggregated.index_add_(0, edge_index[0], messages)
            output = output + aggregated / degree.clamp_min(1).unsqueeze(-1)
        return output + self.bias


class _FallbackTupleTransform:
    def __init__(self, tuple_order):
        self.tuple_order = tuple_order

    def __call__(self, data):
        tuples = list(itertools.combinations(range(data.num_nodes), self.tuple_order))
        rows = torch.tensor(
            [node for tuple_nodes in tuples for node in tuple_nodes],
            dtype=torch.long,
            device=data.x.device,
        )
        cols = torch.arange(len(tuples), device=data.x.device).repeat_interleave(self.tuple_order)
        assignment = torch.stack([rows, cols])
        tuple_count = len(tuples)
        tuple_row = torch.arange(tuple_count, device=data.x.device).repeat_interleave(tuple_count - 1)
        tuple_col = torch.cat([
            torch.cat([torch.arange(i, device=data.x.device), torch.arange(i + 1, tuple_count, device=data.x.device)])
            for i in range(tuple_count)
        ])
        tuple_edge_index = torch.stack([tuple_row, tuple_col])
        setattr(data, f"assignment_index_{self.tuple_order}", assignment)
        setattr(data, f"edge_index_{self.tuple_order}", tuple_edge_index)
        return data


class _FallbackData(SimpleNamespace):
    @property
    def num_nodes(self):
        return self.x.shape[0]


class _FallbackComplete:
    def __call__(self, data):
        return data


def _fallback_avg_pool(features, assignment):
    values, tuple_indices = assignment
    pooled = features.new_zeros(tuple_indices.max().item() + 1, features.shape[-1])
    counts = features.new_zeros(pooled.shape[0], 1)
    pooled.index_add_(0, tuple_indices, features[values])
    counts.index_add_(0, tuple_indices, features.new_ones(values.shape[0], 1))
    return pooled / counts.clamp_min(1)


class CompleteKGNN(nn.Module):
    tuple_order = None

    def __init__(self, in_concepts, in_roles, out_concepts, out_roles, hidden_size=64, num_layers=3, max_objects=None):
        super().__init__()
        try:
            Data, GraphConv, TwoMalkin, ThreeMalkin, _, avg_pool, Complete = _load_k_gnn_components()
            self._using_k_gnn = True
        except ImportError:
            Data, GraphConv, avg_pool, Complete = _FallbackData, _FallbackGraphConv, _fallback_avg_pool, _FallbackComplete
            TwoMalkin = ThreeMalkin = None
            self._using_k_gnn = False
        self._Data = Data
        self._Complete = Complete
        self._tuple_transform = (TwoMalkin if self.tuple_order == 2 else ThreeMalkin) if self._using_k_gnn else _FallbackTupleTransform(self.tuple_order)
        self._avg_pool = avg_pool
        self.in_roles = in_roles
        self.max_objects = max_objects

        self.node_encoder = nn.Linear(in_concepts, hidden_size)
        self.node_convs = nn.ModuleList(
            GraphConv(hidden_size, hidden_size) for _ in range(num_layers)
        )
        tuple_role_size = 2 * in_roles if self.tuple_order == 2 else in_roles
        self.tuple_encoder = nn.Linear(hidden_size + tuple_role_size, hidden_size)
        self.tuple_convs = nn.ModuleList(
            GraphConv(hidden_size, hidden_size) for _ in range(num_layers)
        )
        self.concept_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_concepts),
        )
        self.role_head = nn.Sequential(
            nn.Linear(3 * hidden_size + in_roles, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_roles),
        )

    def _make_data(self, concepts, roles):
        object_count = concepts.shape[-1]
        device = concepts.device
        row = torch.arange(object_count, device=device).repeat_interleave(object_count - 1)
        col = torch.cat([
            torch.cat([torch.arange(i, device=device), torch.arange(i + 1, object_count, device=device)])
            for i in range(object_count)
        ])
        edge_attr = roles[:, row, col].transpose(0, 1).contiguous()
        data = self._Data(x=concepts.transpose(0, 1), edge_index=torch.stack([row, col]), edge_attr=edge_attr)
        return self._Complete()(data)

    @staticmethod
    def _tuple_members(assignment, tuple_count):
        return [
            assignment[0, assignment[1] == tuple_index]
            for tuple_index in range(tuple_count)
        ]

    def _tuple_role_features(self, edge_attr, edge_index, members, object_count):
        edge_matrix = edge_attr.new_zeros(object_count, object_count, edge_attr.shape[-1])
        edge_matrix[edge_index[0], edge_index[1]] = edge_attr
        if self.tuple_order == 2:
            first, second = members
            return torch.cat([edge_matrix[first, second], edge_matrix[second, first]], dim=-1)

        pair_features = []
        for first_index in range(3):
            for second_index in range(3):
                if first_index != second_index:
                    pair_features.append(edge_matrix[members[first_index], members[second_index]])
        return torch.stack(pair_features).mean(dim=0)

    def _pool_to_nodes(self, tuple_features, members, object_count):
        node_sum = tuple_features.new_zeros(object_count, tuple_features.shape[-1])
        node_count = tuple_features.new_zeros(object_count, 1)
        for tuple_feature, tuple_nodes in zip(tuple_features, members):
            node_sum.index_add_(0, tuple_nodes, tuple_feature.expand(tuple_nodes.numel(), -1))
            node_count.index_add_(0, tuple_nodes, tuple_features.new_ones(tuple_nodes.numel(), 1))
        return node_sum / node_count.clamp_min(1)

    def _pool_to_pairs(self, tuple_features, members, object_count):
        pair_sum = tuple_features.new_zeros(object_count, object_count, tuple_features.shape[-1])
        pair_count = tuple_features.new_zeros(object_count, object_count, 1)
        for tuple_feature, tuple_nodes in zip(tuple_features, members):
            for first in tuple_nodes:
                for second in tuple_nodes:
                    if first != second:
                        pair_sum[first, second] += tuple_feature
                        pair_count[first, second] += 1
        return pair_sum / pair_count.clamp_min(1)

    def _forward_one(self, concepts, roles):
        data = self._make_data(concepts, roles)
        node_features = torch.relu(self.node_encoder(data.x))
        for convolution in self.node_convs:
            node_features = torch.relu(convolution(node_features, data.edge_index))

        data.x = node_features

        use_fast_path = self._using_k_gnn and not data.x.is_cuda
        tuple_transform = self._tuple_transform if use_fast_path else _FallbackTupleTransform(self.tuple_order)
        avg_pool = self._avg_pool if use_fast_path else _fallback_avg_pool

        data = tuple_transform()(data) if use_fast_path else tuple_transform(data)
        assignment = data.assignment_index_2 if self.tuple_order == 2 else data.assignment_index_3
        tuple_members = self._tuple_members(assignment, assignment[1].max().item() + 1)
        tuple_features = avg_pool(node_features, assignment)
        tuple_roles = torch.stack([
            self._tuple_role_features(data.edge_attr, data.edge_index, members, node_features.shape[0])
            for members in tuple_members
        ])
        tuple_features = torch.relu(self.tuple_encoder(torch.cat([tuple_features, tuple_roles], dim=-1)))
        for convolution in self.tuple_convs:
            tuple_features = torch.relu(convolution(tuple_features, data.edge_index_2 if self.tuple_order == 2 else data.edge_index_3))

        object_count = node_features.shape[0]
        node_features = self._pool_to_nodes(tuple_features, tuple_members, object_count)
        concept_logits = self.concept_head(node_features).transpose(0, 1)
        pair_features = self._pool_to_pairs(tuple_features, tuple_members, object_count)
        source = node_features[:, None, :].expand(-1, object_count, -1)
        target = node_features[None, :, :].expand(object_count, -1, -1)
        role_input = torch.cat([source, target, pair_features, roles.permute(1, 2, 0)], dim=-1)
        role_logits = self.role_head(role_input).permute(2, 0, 1)
        return concept_logits, role_logits

    def forward(self, concepts, roles):
        if concepts.dim() != 3 or roles.dim() != 4:
            raise ValueError("Expected concepts [batch, concepts, objects] and roles [batch, roles, objects, objects]")
        if self.max_objects is not None and concepts.shape[-1] > self.max_objects:
            raise ValueError(f"3-GNN supports at most {self.max_objects} objects")
        outputs = [self._forward_one(concepts_one, roles_one) for concepts_one, roles_one in zip(concepts, roles)]
        return torch.stack([output[0] for output in outputs]), torch.stack([output[1] for output in outputs])
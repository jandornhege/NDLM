import torch
import torch.nn as nn

from .common import _PairDecoder, _load_edge_transformer_module


class EdgeTransformer(nn.Module):
    """Adapter for the project's triangular-attention edge transformer.

    This wraps the implementation in src/baseline_models/towards-principled-gts/
    edge_transformer.py and exposes the NDLM baseline interface used by the rest of
    the repository.
    """

    def __init__(self, in_concepts, in_roles, out_concepts, out_roles, hidden_size=64, num_layers=2, heads=4):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("baseline-hidden-size must be divisible by baseline-heads")
        self.in_roles = in_roles
        self.node_encoder = nn.Linear(in_concepts, hidden_size)
        self.edge_encoder = nn.Linear(hidden_size * 2 + in_roles, hidden_size)
        edge_transformer_module = _load_edge_transformer_module()
        self.attention = nn.ModuleList(
            edge_transformer_module.EdgeAttention(hidden_size, heads, dropout=0.0)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
        self.node_head = nn.Linear(hidden_size, out_concepts)
        self.pair_decoder = _PairDecoder(hidden_size, in_roles, out_roles)

    def forward(self, concepts, roles):
        batch_size, _, object_count = concepts.shape
        node_features = torch.relu(self.node_encoder(concepts.transpose(1, 2)))
        node_i = node_features.unsqueeze(2).expand(-1, -1, object_count, -1)
        node_j = node_features.unsqueeze(1).expand(-1, object_count, -1, -1)
        edge_inputs = [node_i, node_j]
        if self.in_roles > 0:
            edge_inputs.append(roles.permute(0, 2, 3, 1))
        pair_features = torch.relu(self.edge_encoder(torch.cat(edge_inputs, dim=-1)))

        for attention, norm in zip(self.attention, self.norms):
            attended = attention(pair_features, pair_features, pair_features)
            pair_features = norm(pair_features + attended)

        node_features = pair_features.mean(dim=2)
        concept_logits = self.node_head(node_features).transpose(1, 2)
        role_logits = self.pair_decoder(pair_features, roles)
        return concept_logits, role_logits


PairTransformer = EdgeTransformer

import torch
import torch.nn as nn


class MessagePassingPairClassifier(nn.Module):
    """Relation-aware message-passing baseline with unary and pairwise decoders."""

    def __init__(self, in_concepts, in_roles, out_concepts, out_roles, hidden_size=64, num_layers=3):
        super().__init__()
        self.in_roles = in_roles
        self.node_encoder = nn.Linear(in_concepts, hidden_size)
        self.role_gates = nn.ModuleList(
            nn.Linear(in_roles, hidden_size) for _ in range(num_layers)
        )
        self.node_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            for _ in range(num_layers)
        )
        self.concept_head = nn.Linear(hidden_size, out_concepts)
        pair_input_size = hidden_size * 2 + in_roles
        self.role_head = nn.Sequential(
            nn.Linear(pair_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_roles),
        )

    def forward(self, concepts, roles):
        node_features = torch.relu(self.node_encoder(concepts.transpose(1, 2)))
        if self.in_roles > 0:
            edge_features = roles.permute(0, 2, 3, 1)
            for role_gate, node_update in zip(self.role_gates, self.node_updates):
                gates = torch.sigmoid(role_gate(edge_features))
                outgoing = torch.einsum("bijh,bjh->bih", gates, node_features)
                incoming = torch.einsum("bjih,bjh->bih", gates, node_features)
                node_features = node_update(
                    torch.cat([node_features, outgoing, incoming], dim=-1)
                )

        concept_logits = self.concept_head(node_features).transpose(1, 2)
        object_count = concepts.shape[-1]
        source_features = node_features.unsqueeze(2).expand(-1, -1, object_count, -1)
        target_features = node_features.unsqueeze(1).expand(-1, object_count, -1, -1)
        pair_features = [source_features, target_features]
        if self.in_roles > 0:
            pair_features.append(roles.permute(0, 2, 3, 1))
        role_logits = self.role_head(torch.cat(pair_features, dim=-1))
        return concept_logits, role_logits.permute(0, 3, 1, 2)

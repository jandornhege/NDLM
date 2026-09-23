import torch
import torch.nn as nn


class PairwiseMLP(nn.Module):
    """Pairwise classifier without message passing, used as a local-feature baseline."""

    def __init__(self, in_concepts, in_roles, out_concepts, out_roles, hidden_size=64):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(in_concepts, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.concept_head = nn.Linear(hidden_size, out_concepts)
        pair_input_size = hidden_size * 2 + in_roles
        self.role_head = nn.Sequential(
            nn.Linear(pair_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_roles),
        )

    def forward(self, concepts, roles):
        node_features = self.node_encoder(concepts.transpose(1, 2))
        concept_logits = self.concept_head(node_features).transpose(1, 2)

        object_count = concepts.shape[-1]
        source_features = node_features.unsqueeze(2).expand(-1, -1, object_count, -1)
        target_features = node_features.unsqueeze(1).expand(-1, object_count, -1, -1)
        pair_features = [source_features, target_features]
        if roles.shape[1] > 0:
            pair_features.append(roles.permute(0, 2, 3, 1))
        role_logits = self.role_head(torch.cat(pair_features, dim=-1))
        return concept_logits, role_logits.permute(0, 3, 1, 2)

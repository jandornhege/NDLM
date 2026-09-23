import torch
import torch.nn as nn

from .common import _PairDecoder


class _PPGNBlock(nn.Module):
    """PyTorch equivalent of the PPGN regular block used by the source model."""

    def __init__(self, input_channels, output_channels, mlp_depth):
        super().__init__()
        self.left_mlp = self._make_mlp(input_channels, output_channels, mlp_depth)
        self.right_mlp = self._make_mlp(input_channels, output_channels, mlp_depth)
        self.skip = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=1)

    @staticmethod
    def _make_mlp(input_channels, output_channels, depth):
        layers = []
        for layer_index in range(depth):
            layers.append(
                nn.Conv2d(
                    input_channels if layer_index == 0 else output_channels,
                    output_channels,
                    kernel_size=1,
                )
            )
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, inputs):
        left = self.left_mlp(inputs)
        right = self.right_mlp(inputs)
        product = torch.einsum("bhik,bhkj->bhij", left, right)
        return torch.relu(self.skip(torch.cat([inputs, product], dim=1)))


class PPGN(nn.Module):
    """PPGN-style equivariant backbone adapted to action-model predictions.

    The original PPGN implementation is a TensorFlow graph-classification model
    with input [B, S, N, N] and graph-level outputs. This adapter keeps its
    pointwise-MLP/matrix-product blocks, encodes unary concepts on the diagonal,
    and returns the repository contract: concept logits [B, C, N] and role logits
    [B, R, N, N].
    """

    def __init__(
        self,
        in_concepts,
        in_roles,
        out_concepts,
        out_roles,
        hidden_size=64,
        num_layers=3,
        mlp_depth=2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("PPGN requires at least one layer")
        if mlp_depth < 1:
            raise ValueError("PPGN MLP depth must be at least one")

        input_channels = in_concepts + in_roles
        self.in_roles = in_roles
        self.blocks = nn.ModuleList(
            _PPGNBlock(
                input_channels if layer_index == 0 else hidden_size,
                hidden_size,
                mlp_depth,
            )
            for layer_index in range(num_layers)
        )
        # The supplied model uses its new suffix by default: every block emits
        # a prediction and the predictions are summed before returning.
        self.concept_heads = nn.ModuleList(
            nn.Linear(hidden_size, out_concepts) for _ in range(num_layers)
        )
        self.role_heads = nn.ModuleList(
            _PairDecoder(hidden_size, in_roles, out_roles) for _ in range(num_layers)
        )

    def _pair_input(self, concepts, roles):
        batch_size, _, object_count = concepts.shape
        diagonal = torch.eye(
            object_count,
            dtype=concepts.dtype,
            device=concepts.device,
        ).view(1, 1, object_count, object_count)
        concept_matrices = concepts.unsqueeze(-1) * diagonal
        return torch.cat([concept_matrices, roles], dim=1)

    def forward(self, concepts, roles):
        pair_features = self._pair_input(concepts, roles)
        hidden_outputs = []
        for block in self.blocks:
            pair_features = block(pair_features)
            hidden_outputs.append(pair_features)

        concept_logits = 0
        role_logits = 0
        for hidden, concept_head, role_head in zip(
            hidden_outputs, self.concept_heads, self.role_heads
        ):
            diagonal_features = hidden.diagonal(dim1=2, dim2=3).transpose(1, 2)
            concept_logits = concept_logits + concept_head(diagonal_features).transpose(1, 2)
            role_logits = role_logits + role_head(hidden.permute(0, 2, 3, 1), roles)
        return concept_logits, role_logits
"""Run NDLM on the benchmark graph-classification protocol.

The graph repository defines a fixed parameter-selection split (fold 0) and
the ten published cross-validation folds (folds 1-10). This runner reuses its
data loader and Trainer, replacing only the graph model.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


NDLM_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = NDLM_ROOT.parent
GRAPH_ROOT = WORKSPACE_ROOT / "ProvablyPowerfulGraphNetworks_torch"

sys.path.insert(0, str(NDLM_ROOT / "src"))
sys.path.insert(0, str(GRAPH_ROOT))

from data_loader.data_generator import DataGenerator
from ndlm.configs import config_object
from ndlm.modules import MultiLayerNDLM
from trainers.trainer import Trainer
from utils import doc_utils
from utils.config import process_config
from utils.dirs import create_dirs


class NDLMGraphClassifier(nn.Module):
    """Convert dense benchmark graphs to NDLM tensors and classify them."""

    def __init__(self, node_labels, num_classes, ndlm_config):
        super().__init__()
        self.node_labels = node_labels
        in_concepts = node_labels if node_labels > 0 else 1
        self.ndlm = MultiLayerNDLM(
            in_concepts=in_concepts,
            in_roles=1,
            out_concepts=ndlm_config.NUM_HIDDEN_CONCEPTS,
            out_roles=ndlm_config.NUM_HIDDEN_ROLES,
            config=ndlm_config,
        )
        self.classifier = nn.Linear(2 * ndlm_config.NUM_HIDDEN_CONCEPTS, num_classes)

    def forward(self, graphs):
        adjacency = graphs[:, :1]
        if self.node_labels > 0:
            concepts = graphs[:, 1 : self.node_labels + 1].diagonal(dim1=-2, dim2=-1)
        else:
            concepts = torch.ones(
                (graphs.shape[0], 1, graphs.shape[-1]),
                dtype=graphs.dtype,
                device=graphs.device,
            )

        output_concepts, _ = self.ndlm(concepts, adjacency)
        graph_features = torch.cat(
            [output_concepts.mean(dim=-1), output_concepts.amax(dim=-1)],
            dim=1,
        )
        return self.classifier(graph_features)


class NDLMModelWrapper:
    """ModelWrapper-compatible facade used by the graph Trainer."""

    def __init__(self, config, ndlm_config, device):
        self.config = config
        self.model = NDLMGraphClassifier(
            config.node_labels,
            config.num_classes,
            ndlm_config,
        ).to(device)

    def run_model_get_loss_and_results(self, graphs, labels):
        scores = self.model(graphs)
        loss = nn.functional.cross_entropy(scores, labels, reduction="sum")
        correct = (scores.argmax(dim=1) == labels).sum().item()
        return loss, correct

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NDLM on the graph benchmark parameter split or 10-fold CV."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--graph-config",
        type=Path,
        default=GRAPH_ROOT / "configs" / "10fold_config.json",
    )
    parser.add_argument(
        "--protocol",
        choices=("parameter", "cv"),
        default="cv",
        help="Use fold 0 for tuning, or folds 1-10 for final evaluation.",
    )
    parser.add_argument("--fold", type=int, default=None, help="Run one CV fold (1-10).")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--hidden-concepts", type=int, default=10)
    parser.add_argument("--hidden-roles", type=int, default=10)
    parser.add_argument("--mode", choices=("strict", "relaxed"), default="strict")
    parser.add_argument("--activation", choices=("identity", "sigmoid"), default="identity")
    parser.add_argument("--transitive-closure", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--input-residual", action="store_true")
    parser.add_argument("--initial-ffn", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--decay-rate", type=float, default=None)
    parser.add_argument("--optimizer", choices=("adam", "momentum"), default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gpu", default=None, help="Override CUDA_VISIBLE_DEVICES from graph config.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def make_ndlm_config(args):
    config = config_object()
    config.NUM_LAYERS = args.num_layers
    config.NUM_HIDDEN_CONCEPTS = args.hidden_concepts
    config.NUM_HIDDEN_ROLES = args.hidden_roles
    config.MODE = args.mode
    config.ACTIVATION_FUNCTION = nn.Identity() if args.activation == "identity" else nn.Sigmoid()
    config.TRANSITIVE_CLOSURE = args.transitive_closure
    config.RESIDUAL = args.residual
    config.INPUT_RESIDUAL = args.input_residual
    config.INITIAL_FFN = args.initial_ffn
    return config


def main():
    args = parse_args()
    if args.protocol == "parameter" and args.fold is not None:
        raise ValueError("--fold cannot be combined with --protocol parameter")
    if args.fold is not None and not 1 <= args.fold <= 10:
        raise ValueError("--fold must be between 1 and 10")

    graph_config = process_config(str(args.graph_config), args.dataset_name)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        graph_config.gpu = args.gpu
    else:
        graph_config.gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.num_epochs is not None:
        graph_config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        graph_config.hyperparams.batch_size = args.batch_size
    if args.learning_rate is not None:
        graph_config.hyperparams.learning_rate = args.learning_rate
    if args.decay_rate is not None:
        graph_config.hyperparams.decay_rate = args.decay_rate
    if args.optimizer is not None:
        graph_config.hyperparams.optimizer = args.optimizer
    graph_config.hyperparams.momentum = args.momentum
    graph_config.hyperparams.weight_decay = args.weight_decay

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested, but CUDA is unavailable "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}, "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    graph_config.device = str(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    suffix = (
        f"ndlm_{args.mode}_L{args.num_layers}_HC{args.hidden_concepts}_"
        f"HR{args.hidden_roles}_{args.activation}_"
        f"LR{graph_config.hyperparams.learning_rate:g}_"
        f"DR{graph_config.hyperparams.decay_rate:g}_"
        f"WD{graph_config.hyperparams.weight_decay:g}"
    )
    graph_config.parent_dir = f"{graph_config.parent_dir}_{suffix}"
    graph_config.summary_dir = str(WORKSPACE_ROOT / "NDLM/outputs/GRAPH_EXPERIMENTS" / graph_config.parent_dir / "summary")
    graph_config.checkpoint_dir = str(WORKSPACE_ROOT / "NDLM/outputs/GRAPH_EXPERIMENTS" / graph_config.parent_dir / "checkpoint")
    graph_config.ndlm = {
        "num_layers": args.num_layers,
        "hidden_concepts": args.hidden_concepts,
        "hidden_roles": args.hidden_roles,
        "mode": args.mode,
        "activation": args.activation,
        "transitive_closure": args.transitive_closure,
        "residual": args.residual,
        "input_residual": args.input_residual,
        "initial_ffn": args.initial_ffn,
    }
    create_dirs([graph_config.summary_dir, graph_config.checkpoint_dir])
    doc_utils.doc_used_config(graph_config)

    folds = [0] if args.protocol == "parameter" else ([args.fold] if args.fold else range(1, 11))
    ndlm_config = make_ndlm_config(args)
    for fold in folds:
        graph_config.num_fold = fold
        print(f"Dataset={args.dataset_name} protocol={args.protocol} fold={fold}")
        data = DataGenerator(graph_config)
        model_wrapper = NDLMModelWrapper(graph_config, ndlm_config, device)
        Trainer(model_wrapper, data, graph_config).train()

    if args.protocol == "cv" and args.fold is None:
        doc_utils.summary_10fold_results(graph_config.summary_dir)


if __name__ == "__main__":
    main()
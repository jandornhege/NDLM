import resource
import sys
import time
from pathlib import Path
import argparse
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold

from ndlm.configs import config_object
from ndlm.modules import MultiLayerNDLM
import ndlm.main as NDLM_main
import owl_parser.benchmark_owl_ndlm as benchmark_owl_ndlm
import os
import numpy as np

from my_logging import log, init_logger

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = {
    "NTN": "src/tasks/owl_parser/ntn_entities.pt",
    "Biopax": "src/data/Ontolearn/KGs/Biopax/biopax.entities.pt",
    "Family": "src/data/Ontolearn/KGs/Family/family-benchmark_rich_background.entities.pt",
    "Lymphography": "src/data/Ontolearn/KGs/Lymphography/lymphography.entities.pt"
}
owl_files = {
    "NTN": "src/tasks/owl_parser/ntn_entities.pt",
    "Biopax": "src/data/Ontolearn/KGs/Biopax/biopax.owl",
    "Family": "src/data/Ontolearn/KGs/Family/family-benchmark_rich_background.owl",
    "Lymphography": "src/data/Ontolearn/KGs/Lymphography/lymphography.owl"
}

lp_files = {
    "Biopax": "src/data/Ontolearn/LPs/Biopax/lps.json",
    "Family": "src/data/Ontolearn/LPs/Family/lps_difficult.json",
    "Lymphography": "src/data/Ontolearn/LPs/Lymphography/lps.json"
}


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--neighborhood", type=int, default=4)
parser.add_argument("--num-layers", type=int, default=3)
parser.add_argument("--hidden-concepts", type=int, default=5)
parser.add_argument("--hidden-roles", type=int, default=5)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=1e-4,
    help="Adam weight decay; default preserves the original 1e-4 setting.",
)
parser.add_argument(
    "--activation",
    choices=("identity", "sigmoid"),
    default="sigmoid",
)
parser.add_argument("--configuration-name", default="L3H5_sig")
parser.add_argument(
    "--output-root",
    default="BY_EXAMPLE",
    help="Directory below outputs/CONCEPT_LEARNER for this experiment round.",
)
parser.add_argument(
    "--problem-index",
    type=int,
    default=None,
    help="Zero-based index of one learning problem to run; omit to run all.",
)
parser.add_argument(
    "--mark-target-object",
    action="store_true",
    help=(
        "Add a concept marking the target object and supervise the target "
        "for every object in the local neighborhood."
    ),
)
parser.add_argument(
    "--fold-time-limit-seconds",
    type=float,
    default=None,
    help=(
        "Optional strict wall-clock limit per fold; training stops early "
        "(after one final test round) once this many seconds have elapsed "
        "since the fold started."
    ),
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help=(
        "Log detailed per-dataset metrics every epoch (large log files). "
        "By default only one summary line per epoch/test round is logged."
    ),
)
parser.add_argument(
    "--lr-scheduler",
    choices=("none", "plateau", "cosine"),
    default="none",
    help=(
        "Learning-rate schedule. 'plateau' halves the LR when the epoch "
        "training loss stops improving for 20 epochs; 'cosine' decays the "
        "LR smoothly to 0 over --num-epochs (fixed at 500). Default 'none' "
        "keeps the original fixed learning rate."
    ),
)
args = parser.parse_args()
name = args.name  # "Family"  # "NTN"  # "Biopax"  # "Family"  # "Lymphography"
tensor_path = data[name]
owl_files = owl_files[name] 

lp_path = Path(lp_files[name])



experiment_path = (
    Path("outputs/CONCEPT_LEARNER")
    / args.output_root
    / args.configuration_name
    / name
    / f"neighborhood_{args.neighborhood}"
)

problems = benchmark_owl_ndlm.load_lp_examples(lp_path)
problem_items = list(problems.items())

if args.problem_index is not None:
    if not 0 <= args.problem_index < len(problem_items):
        raise ValueError(
            f"problem-index must be between 0 and {len(problem_items) - 1}, "
            f"got {args.problem_index}"
        )
    problem_items = [problem_items[args.problem_index]]

sample_cache = {}

def example_to_ndlm_sample(example, device):
    """
    Create an NDLM sample for one specific example in the local graph.

    `example_index` is the index of the target individual in the
    local object dimension.
    `example_label` is 1 for a positive example and 0 for a negative.
    """

    cache_key = example["name"]
    if cache_key not in sample_cache:
        problem = benchmark_owl_ndlm.make_learning_problem(
            owl_files=owl_files,
            center=example["center"],
            label=example["label"],
            name=example["name"],
            radius=args.neighborhood,
            mark_target_object=args.mark_target_object,
        )
        num_objects = problem["concepts"].shape[1]
        sample_cache[cache_key] = (
            problem["concepts"].unsqueeze(0),
            problem["roles"].unsqueeze(0),
            problem["target"].unsqueeze(0).unsqueeze(0),
            torch.zeros(
                (1, 0, num_objects, num_objects),
                dtype=torch.float32,
            ),
            problem["mask"].unsqueeze(0).unsqueeze(0),
            torch.zeros(
                (1, 0, num_objects, num_objects),
                dtype=torch.bool,
            ),
        )

    return tuple(tensor.to(device) for tensor in sample_cache[cache_key])


for lp_name, examples in problem_items:
    problem_path = experiment_path / lp_name / f"run_{time.time_ns()}"
    summary_log = problem_path / "summary.log"
    init_logger(summary_log)

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    log(f"\n{'#' * 70}")
    log(f"Learning problem: {lp_name}")
    log(f"Number of examples: {len(examples)}")
    log(f"{'#' * 70}")

    labels = [
        example["label"]
        for example in examples
    ]

    skf = StratifiedKFold(
        n_splits=min(10, len(examples), min(np.bincount(labels))),
        shuffle=True,
        random_state=42,
    )

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(examples, labels)
    ):
        torch.manual_seed(42 + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + fold)

        log(f"{'=' * 60}")
        log(f"{lp_name} — Fold {fold + 1}/10")


        train_examples = [
            examples[i]
            for i in train_idx
        ]

        test_examples = [
            examples[i]
            for i in test_idx
        ]

        # ----------------------------------------------------
        # Convert examples to NDLM samples
        # ----------------------------------------------------
       
        train = [
            example_to_ndlm_sample(
                example,
                device,
            )
            for example in train_examples
        ]

        test = [
            example_to_ndlm_sample(
                example,
                device,
            )
            for example in test_examples
        ]

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        config = config_object()

        config.NUM_LAYERS = args.num_layers
        config.NUM_HIDDEN_CONCEPTS = args.hidden_concepts
        config.NUM_HIDDEN_ROLES = args.hidden_roles
        config.MODE = "strict"
        config.TRANSITIVE_CLOSURE = False
        config.INITIAL_FFN = False
        config.ACTIVATION_FUNCTION = (
            nn.Sigmoid() if args.activation == "sigmoid" else nn.Identity()
        )
        

        training_args = argparse.Namespace(
            learning_rate=1e-3,
                weight_decay=args.weight_decay,
            batch_size=5,
            loss_type="BCE",
            num_epochs=500,
            test_interval=100,
            weighted_loss=False,
            experiment_path=problem_path / f"fold_{fold + 1:02d}",
            fold_time_limit_seconds=args.fold_time_limit_seconds,
            verbose=args.verbose,
            lr_scheduler=args.lr_scheduler,
        )

        os.makedirs(
            training_args.experiment_path,
            exist_ok=True,
        )
        if fold == 0:
            log(f"{len(train)} many train datasets found with following shapes:")
            for i, ds in enumerate(train):
                log(f"Dataset {i}: concepts: {ds[0].shape}, roles: {ds[1].shape}, c_target: {ds[2].shape}, r_target: {ds[3].shape} ")

            log(f"{len(test)} many test datasets found with following shapes:")
            for i, ds in enumerate(test):
                log(f"Dataset {i}: concepts: {ds[0].shape}, roles: {ds[1].shape}, c_target: {ds[2].shape}, r_target: {ds[3].shape} ")
                    
       
        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        first_problem = train[0]

        model = MultiLayerNDLM(
            first_problem[0].shape[1],  # concepts, including optional marker
            first_problem[1].shape[1],  # roles
            1,
            0,
            config,
        ).to(device).eval()

        # ----------------------------------------------------
        # Train / evaluate
        # ----------------------------------------------------
        init_logger(
            problem_path
            / "folds"
            / f"fold_{fold + 1:02d}.log"
        )


        tp, tn, fp, fn, _, _, _, _ = NDLM_main.main(
            train,
            test,
            config,
            training_args,
            model,
            checkpoint_path=(
                training_args.experiment_path / "checkpoints"
                if training_args.experiment_path
                else None
            ),
            log=log,
        )
        init_logger(summary_log)

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        log(f"{lp_name} — Fold {fold + 1}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")



    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    accuracy = (
        (total_tp + total_tn)
        / (total_tp + total_fp + total_tn + total_fn)
        if total_tp + total_fp + total_tn + total_fn > 0
        else 0.0
    )

    log(
        f"\n{'=' * 60}\n"
        f"{lp_name} — 10-Fold CV Results\n"
        f"{'=' * 60}\n"
        f"TP={total_tp}, FP={total_fp}, "
        f"TN={total_tn}, FN={total_fn}\n"
        f"Precision={precision:.4f}\n"
        f"Recall={recall:.4f}\n"
        f"F1={f1:.4f}\n"
        f"Accuracy={accuracy:.4f}\n"
    )
    
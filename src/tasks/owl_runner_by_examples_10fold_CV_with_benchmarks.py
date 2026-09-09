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
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.owl_individual import OWLNamedIndividual

from my_logging import log, init_logger

from ontolearn.learners import (
    ALCSAT,
    CELOE,
    CLIP,
    Drill,
    EvoLearner,
    NCES,
    NCES2,
    NERO,
    OCEL,
    ROCES,
    SPELL,
    TDL,
)

ONTOLEARN_LEARNERS = {
    "ALCSAT": ALCSAT,
    "CELOE": CELOE,
    # "CLIP": CLIP,
    "Drill": Drill,
    "EvoLearner": EvoLearner,
    # "NCES": NCES,
    # "NCES2": NCES2,
    # "NERO": NERO,
    "OCEL": OCEL,
    # "ROCES": ROCES,
    # "SPELL": SPELL,
    "TDL": TDL,
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Best configuration per domain, selected from
# other/reports/res_concept_learning_all.txt by picking the highest
# combined F1 among the configs that completed the largest number of
# runs/folds for that domain (ties broken by taking the first-ranked entry).
configs_per_domain = {
    "Biopax": {
        "num_layers": 3,
        "hidden_concepts": 10,
        "hidden_roles": 10,
        "activation": "identity",
        "neighborhood": 1,
        "mark_target_object": False,
    },
    "Carcinogenesis": {
        "num_layers": 5,
        "hidden_concepts": 5,
        "hidden_roles": 5,
        "activation": "sigmoid",
        "neighborhood": 1,
        "mark_target_object": False,
    },
    "Carcinogenesis_drill": {
        "num_layers": 3,
        "hidden_concepts": 10,
        "hidden_roles": 10,
        "activation": "identity",
        "neighborhood": 1,
        "mark_target_object": False,
    },
    "Family": {
        "num_layers": 3,
        "hidden_concepts": 10,
        "hidden_roles": 10,
        "activation": "identity",
        "neighborhood": 6,
        "mark_target_object": False,
    },
    "Lymphography": {
        "num_layers": 3,
        "hidden_concepts": 5,
        "hidden_roles": 5,
        "activation": "sigmoid",
        "neighborhood": 1,
        "mark_target_object": False,
    },
    "Mutagenesis": {
        "num_layers": 3,
        "hidden_concepts": 5,
        "hidden_roles": 5,
        "activation": "identity",
        "neighborhood": 1,
        "mark_target_object": False,
    },
    "Mutagenesis_drill": {
        "num_layers": 3,
        "hidden_concepts": 10,
        "hidden_roles": 10,
        "activation": "sigmoid",
        "neighborhood": 1,
        "mark_target_object": False,
    },
    "Nctrer": {
        "num_layers": 3,
        "hidden_concepts": 10,
        "hidden_roles": 10,
        "activation": "identity",
        "neighborhood": 1,
        "mark_target_object": False,
    },
    # Suramin: no fold or F1 results found in the report; no config selected.
}


owl_files = {
    "Biopax": "src/data/Ontolearn/KGs/Biopax/biopax.owl",
    "Carcinogenesis": "src/data/Ontolearn/KGs/Carcinogenesis/carcinogenesis.owl",
    "Carcinogenesis_drill": "src/data/Ontolearn/KGs/Carcinogenesis/carcinogenesis.owl",
    "Family": "src/data/Ontolearn/KGs/Family/family-benchmark_rich_background.owl",
    "Lymphography": "src/data/Ontolearn/KGs/Lymphography/lymphography.owl",
    "Mutagenesis": "src/data/Ontolearn/KGs/Mutagenesis/mutagenesis.owl",
    "Mutagenesis_drill": "src/data/Ontolearn/KGs/Mutagenesis/mutagenesis.owl",
    "Nctrer": "src/data/Ontolearn/KGs/Nctrer/nctrer.owl", 
    "Suramin": "src/data/Ontolearn/KGs/Suramin/dataset.ttl"
}

lp_files = {
    "Biopax": "src/data/Ontolearn/LPs/Biopax/lps.json",
    "Carcinogenesis": "src/data/Ontolearn/LPs/Carcinogenesis/lps.json",
    "Carcinogenesis_drill": "src/data/Ontolearn/LPs/Carcinogenesis/lps_drill.json",
    "Family": "src/data/Ontolearn/LPs/Family/lps_difficult.json",
    "Lymphography": "src/data/Ontolearn/LPs/Lymphography/lps.json",
    "Mutagenesis": "src/data/Ontolearn/LPs/Mutagenesis/lps.json",
    "Mutagenesis_drill": "src/data/Ontolearn/LPs/Mutagenesis/lps_generated_drill.json",
    "Nctrer": "src/data/Ontolearn/LPs/Nctrer/lps.json", 
    "Suramin": "src/data/Ontolearn/LPs/Suramin/lps.json"
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
    "--mark-target-object-keep-mask",
    action="store_true",
    help=(
        "Only relevant together with --mark-target-object. Still adds the "
        "marker concept, but keeps the original mask/target behaviour of "
        "supervising only the center object instead of the whole "
        "neighborhood."
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
parser.add_argument(
    "--include-benchmarks",
    action="store_true",
    help="Include benchmark datasets in the evaluation.",
)
parser.add_argument(
    "--no-initial-ffn",
    dest="initial_ffn",
    action="store_false",
    help="Disable the initial feed-forward layer (config.INITIAL_FFN); enabled by default.",
)
parser.add_argument(
    "--use-best-config",
    action="store_true",
    help=(
        "Override --neighborhood, --num-layers, --hidden-concepts, "
        "--hidden-roles, --activation and --mark-target-object with the "
        "best known configuration for --name, from configs_per_domain."
    ),
)

args = parser.parse_args()
name = args.name 
owl_files = owl_files[name] 

if args.use_best_config:
    if name not in configs_per_domain:
        raise ValueError(
            f"No best configuration available for domain {name!r}."
        )
    best_config = configs_per_domain[name]
    args.neighborhood = best_config["neighborhood"]
    args.num_layers = best_config["num_layers"]
    args.hidden_concepts = best_config["hidden_concepts"]
    args.hidden_roles = best_config["hidden_roles"]
    args.activation = best_config["activation"]
    # Preserve an explicit --mark-target-object CLI flag instead of letting
    # the stored best config (currently always unmarked) silently clear it.
    args.mark_target_object = (
        args.mark_target_object or best_config["mark_target_object"]
    )

lp_path = Path(lp_files[name])

if args.include_benchmarks:
    kb = KnowledgeBase(path=owl_files)


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
            mark_target_object_keep_mask=args.mark_target_object_keep_mask,
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

def create_ontolearn_learner(method_name, learner_cls, kb):
    """Create an Ontolearn learner with method-specific defaults."""

    if method_name in {"CELOE", "OCEL"}:
        return learner_cls(
            knowledge_base=kb,
            quality_func=F1(),
            max_runtime=60,
        )

    if method_name == "ALCSAT":
        return learner_cls(
            knowledge_base=kb,
            max_runtime=60,
            max_concept_size=30,
        )

    if method_name == "SPELL":
        return learner_cls(
            knowledge_base=kb,
            max_runtime=60,
            max_query_size=10,
            search_mode="full_approx",
        )

    if method_name == "TDL":
        return learner_cls(
            knowledge_base=kb,
            kwargs_classifier={"random_state": 1},
            max_runtime=60,
            verbose=0,
        )

    if method_name == "Drill":
        return learner_cls(
            knowledge_base=kb,
            quality_func=F1(),
            max_runtime=60,
            verbose=0,
        )

    if method_name == "EvoLearner":
        return learner_cls(
            knowledge_base=kb,
            quality_func=F1(),
            max_runtime=60,
        )

    if method_name == "NERO":
        return learner_cls(
            knowledge_base=kb,
            num_embedding_dim=128,
            neural_architecture="DeepSet",
            learning_rate=0.001,
            num_epochs=50,
            batch_size=32,
            quality_func=F1(),
            max_runtime=60,
            verbose=0,
        )

    if method_name == "NCES":
        return learner_cls(
            knowledge_base=kb,
            quality_func=F1(),
            load_pretrained=True,
            num_predictions=200,
            verbose=0,
            enforce_validity=False,
        )

    if method_name == "NCES2":
        return learner_cls(
            knowledge_base=kb,
            quality_func=F1(),
            load_pretrained=True,
            num_predictions=200,
            verbose=0,
            enforce_validity=False,
        )

    if method_name == "ROCES":
        return learner_cls(
            knowledge_base=kb,
            k=50,
            quality_func=F1(),
            load_pretrained=True,
            num_predictions=200,
            verbose=0,
            enforce_validity=False,
        )

    if method_name == "CLIP":
        return learner_cls(
            knowledge_base=kb,
            quality_func=F1(),
            max_num_of_concepts_tested=int(1e9),
            max_runtime=60,
            load_pretrained=True,
        )

    raise ValueError(f"Unknown Ontolearn learner: {method_name}")


def run_ontolearn_fold(
    method_name,
    learner_cls,
    train_lp,
    test_lp,
    kb,
):
    start = time.time()

    learner = create_ontolearn_learner(
        method_name,
        learner_cls,
        kb,
    )

    learner.fit(train_lp)
    
    if method_name == "ALCSAT":
        hypothesis = learner._best_hypothesis
    else:
        hypotheses = learner.best_hypotheses()

        # Ontolearn's best_hypotheses() may return a collection.
        if isinstance(hypotheses, (list, tuple, set)):
            hypothesis = next(iter(hypotheses))
        else:
            hypothesis = hypotheses

    predicted = set(kb.individuals(hypothesis))


    test_pos = set(test_lp.pos)
    test_neg = set(test_lp.neg)

    tp = len(predicted & test_pos)
    fp = len(predicted & test_neg)
    fn = len(test_pos - predicted)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    runtime = time.time() - start

    return f1, runtime


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
        config.INITIAL_FFN = args.initial_ffn
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


        training_args.return_details = True
        result = NDLM_main.main(
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
        test_totals = result["test"]["totals"]
        tp = test_totals["tp_c"]
        tn = test_totals["tn_c"]
        fp = test_totals["fp_c"]
        fn = test_totals["fn_c"]
        log(f"Final train results: {result['train']}")
        init_logger(summary_log)

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        log(f"{lp_name} — Fold {fold + 1}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        # ----------------------------------------------------
        # Benchmark setup
        # ----------------------------------------------------
        if args.include_benchmarks:

            train_pos = {
                OWLNamedIndividual(example["center"].strip("<>"))
                for example in train_examples
                if example["label"] == 1
            }

            train_neg = {
                OWLNamedIndividual(example["center"].strip("<>"))
                for example in train_examples
                if example["label"] == 0
            }

            test_pos = {
                OWLNamedIndividual(example["center"].strip("<>"))
                for example in test_examples
                if example["label"] == 1
            }

            test_neg = {
                OWLNamedIndividual(example["center"].strip("<>"))
                for example in test_examples
                if example["label"] == 0
            }

            train_lp = PosNegLPStandard(
                pos=train_pos,
                neg=train_neg,
            )

            test_lp = PosNegLPStandard(
                pos=test_pos,
                neg=test_neg,
            )

            for method_name, learner_cls in ONTOLEARN_LEARNERS.items():
                try:
                    method_f1, method_runtime = run_ontolearn_fold(
                        method_name,
                        learner_cls,
                        train_lp,
                        test_lp,
                        kb,
                    )

                    log(
                        f"{lp_name} — Fold {fold + 1}: "
                        f"{method_name} F1={method_f1:.4f}, "
                        f"{method_name} Runtime={method_runtime:.4f}"
                    )

                except Exception as exc:
                    log(
                        f"{lp_name} — Fold {fold + 1}: "
                        f"{method_name} FAILED: "
                        f"{type(exc).__name__}: {exc}"
                    )
                



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
    
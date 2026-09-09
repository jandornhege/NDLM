import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
import difflogic.nn.neural_logic.layer as layer

from ndlm.configs import config_object
from ndlm.modules import MultiLayerNDLM
import ndlm.main as NDLM_main
import owl_parser.benchmark_owl_ndlm as benchmark_owl_ndlm

from my_logging import init_logger, log


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


owl_files = {
	"Biopax": "src/data/Ontolearn/KGs/Biopax/biopax.owl",
	"Carcinogenesis": "src/data/Ontolearn/KGs/Carcinogenesis/carcinogenesis.owl",
	"Carcinogenesis_drill": "src/data/Ontolearn/KGs/Carcinogenesis/carcinogenesis.owl",
	"Family": "src/data/Ontolearn/KGs/Family/family-benchmark_rich_background.owl",
	"Lymphography": "src/data/Ontolearn/KGs/Lymphography/lymphography.owl",
	"Mutagenesis": "src/data/Ontolearn/KGs/Mutagenesis/mutagenesis.owl",
	"Mutagenesis_drill": "src/data/Ontolearn/KGs/Mutagenesis/mutagenesis.owl",
	"Nctrer": "src/data/Ontolearn/KGs/Nctrer/nctrer.owl",
	"Suramin": "src/data/Ontolearn/KGs/Suramin/dataset.ttl",
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
	"Suramin": "src/data/Ontolearn/LPs/Suramin/lps.json",
}


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Run one fixed NDLM configuration under leakage-aware Ontolearn "
			"selection protocols. Hyperparameter search is expected to happen "
			"outside this script, for example with separate Slurm array jobs."
		)
	)
	parser.add_argument("--name", choices=sorted(lp_files), required=True)
	parser.add_argument("--neighborhood", type=int, default=4)
	parser.add_argument("--num-layers", type=int, default=3)
	parser.add_argument("--hidden-concepts", type=int, default=5)
	parser.add_argument("--hidden-roles", type=int, default=5)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--activation", choices=("identity", "sigmoid"), default="sigmoid")
	parser.add_argument("--model", choices=("NDLM", "NLM"), default="NDLM")
	parser.add_argument("--nlm-breadth", type=int, default=3)
	parser.add_argument("--nlm-exclude-self", action="store_true")
	parser.add_argument("--nlm-residual", action="store_true")
	parser.add_argument("--configuration-name", default="L3H5_sig")
	parser.add_argument("--output-root", default="BY_EXAMPLE_SPLIT")
	parser.add_argument("--problem-index", type=int, default=None)
	parser.add_argument("--mark-target-object", action="store_true")
	parser.add_argument("--mark-target-object-keep-mask", action="store_true")
	parser.add_argument("--fold-time-limit-seconds", type=float, default=None)
	parser.add_argument("--verbose", action="store_true")
	parser.add_argument(
		"--no-initial-ffn",
		dest="initial_ffn",
		action="store_false",
		help="Disable the initial feed-forward layer; enabled by default.",
	)
	parser.add_argument(
		"--lr-scheduler",
		choices=("none", "plateau", "cosine"),
		default="none",
	)
	parser.add_argument(
		"--stage",
		choices=("legacy", "hparam", "final"),
		default="hparam",
		help=(
			"legacy: old per-LP CV over all selected LPs. hparam: dev-score "
			"stage for this fixed config. final: held-out-score stage for a "
			"chosen fixed config."
		),
	)
	parser.add_argument(
		"--lp-test-fraction",
		type=float,
		default=0.2,
		help="Multi-LP fraction held out for final evaluation.",
	)
	parser.add_argument("--outer-folds", type=int, default=10)
	parser.add_argument("--inner-folds", type=int, default=5)
	parser.add_argument(
		"--outer-fold",
		type=int,
		default=None,
		help=(
			"For single-LP hparam/final stages, run one zero-based outer fold. "
			"Omit to run all outer folds for this fixed config."
		),
	)
	parser.add_argument("--split-seed", type=int, default=42)
	args = parser.parse_args()

	if not 0.0 < args.lp_test_fraction < 1.0:
		parser.error("--lp-test-fraction must be between 0 and 1")
	if args.outer_folds < 2:
		parser.error("--outer-folds must be at least 2")
	if args.inner_folds < 2:
		parser.error("--inner-folds must be at least 2")
	return args


def make_stratified_splits(examples, labels, requested_folds, random_state):
	class_counts = np.bincount(labels, minlength=2)
	n_splits = min(requested_folds, len(examples), int(class_counts.min()))
	if n_splits < 2:
		raise ValueError(
			"Need at least two positive and two negative examples for "
			"stratified CV. Counts: "
			f"positive={int(class_counts[1])}, negative={int(class_counts[0])}"
		)
	splitter = StratifiedKFold(
		n_splits=n_splits,
		shuffle=True,
		random_state=random_state,
	)
	return list(splitter.split(examples, labels))


def metrics_from_counts(tp, fp, tn, fn):
	precision = tp / (tp + fp) if tp + fp > 0 else 0.0
	recall = tp / (tp + fn) if tp + fn > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
	accuracy = (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn > 0 else 0.0
	return precision, recall, f1, accuracy


def build_config(args):
	config = config_object()
	config.NUM_LAYERS = args.num_layers
	config.NUM_HIDDEN_CONCEPTS = args.hidden_concepts
	config.NUM_HIDDEN_ROLES = args.hidden_roles
	config.MODE = "strict"
	config.TRANSITIVE_CLOSURE = False
	config.INITIAL_FFN = args.initial_ffn
	config.ACTIVATION_FUNCTION = nn.Sigmoid() if args.activation == "sigmoid" else nn.Identity()
	return config


def build_model(args, config, in_concepts, in_roles):
	if args.model == "NLM":
		args.activation_function = args.activation
		return layer.NLM_to_NDLM_Adapter(
			in_concepts,
			in_roles,
			1,
			0,
			args,
		)
	return MultiLayerNDLM(in_concepts, in_roles, 1, 0, config)


def make_sample_cache(args, domain_owl_file):
	sample_cache = {}

	def example_to_ndlm_sample(example):
		cache_key = example["name"]
		if cache_key not in sample_cache:
			problem = benchmark_owl_ndlm.make_learning_problem(
				owl_files=domain_owl_file,
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
				torch.zeros((1, 0, num_objects, num_objects), dtype=torch.float32),
				problem["mask"].unsqueeze(0).unsqueeze(0),
				torch.zeros((1, 0, num_objects, num_objects), dtype=torch.bool),
			)
		return tuple(tensor.to(device) for tensor in sample_cache[cache_key])

	return example_to_ndlm_sample


def run_fixed_config_fold(
	args,
	example_to_ndlm_sample,
	lp_name,
	fold_label,
	train_examples,
	test_examples,
	problem_path,
	summary_log,
	fold_dir_name,
	seed,
):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	train = [example_to_ndlm_sample(example) for example in train_examples]
	test = [example_to_ndlm_sample(example) for example in test_examples]
	config = build_config(args)

	training_args = argparse.Namespace(
		learning_rate=1e-3,
		weight_decay=args.weight_decay,
		batch_size=5,
		loss_type="BCE",
		num_epochs=500,
		test_interval=100,
		weighted_loss=False,
		experiment_path=problem_path / fold_dir_name,
		fold_time_limit_seconds=args.fold_time_limit_seconds,
		verbose=args.verbose,
		lr_scheduler=args.lr_scheduler,
	)
	os.makedirs(training_args.experiment_path, exist_ok=True)

	first_sample = train[0]
	model = build_model(
		args,
		config,
		first_sample[0].shape[1],
		first_sample[1].shape[1],
	).to(device).eval()

	init_logger(problem_path / "folds" / f"{fold_dir_name}.log")
	log(f"{lp_name} - {fold_label}")
	log(f"Train examples: {len(train_examples)}")
	log(f"Test examples: {len(test_examples)}")
	log(
		"Config: "
		f"layers={args.num_layers}, "
		f"hidden={args.hidden_concepts}/{args.hidden_roles}, "
		f"activation={args.activation}, "
		f"model={args.model}, "
		f"nlm_breadth={args.nlm_breadth}, "
		f"neighborhood={args.neighborhood}"
	)

	training_args.return_details = True
	result = NDLM_main.main(
		train,
		test,
		config,
		training_args,
		model,
		checkpoint_path=training_args.experiment_path / "checkpoints",
		log=log,
	)
	test_totals = result["test"]["totals"]
	tp = test_totals["tp_c"]
	tn = test_totals["tn_c"]
	fp = test_totals["fp_c"]
	fn = test_totals["fn_c"]
	log(f"Final train results: {result['train']}")
	init_logger(summary_log)
	log(f"{fold_label}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
	return tp, fp, tn, fn


def log_result_block(title, tp, fp, tn, fn):
	precision, recall, f1, accuracy = metrics_from_counts(tp, fp, tn, fn)
	log(
		f"\n{'=' * 60}\n"
		f"{title}\n"
		f"{'=' * 60}\n"
		f"TP={tp}, FP={fp}, TN={tn}, FN={fn}\n"
		f"Precision={precision:.4f}\n"
		f"Recall={recall:.4f}\n"
		f"F1={f1:.4f}\n"
		f"Accuracy={accuracy:.4f}\n"
	)


def run_problem_cv(args, example_to_ndlm_sample, experiment_path, lp_name, examples, fold_prefix):
	problem_path = experiment_path / lp_name / f"run_{time.time_ns()}"
	summary_log = problem_path / "summary.log"
	init_logger(summary_log)

	labels = [example["label"] for example in examples]
	splits = make_stratified_splits(examples, labels, args.outer_folds, args.split_seed)
	if args.stage == "final" and args.outer_fold is not None:
		if not 0 <= args.outer_fold < len(splits):
			raise ValueError(f"--outer-fold must be between 0 and {len(splits) - 1}")
		splits = [splits[args.outer_fold]]

	total_tp = total_fp = total_tn = total_fn = 0
	log(f"Learning problem: {lp_name}")
	log(f"Stage: {args.stage}")
	log(f"Number of examples: {len(examples)}")
	log(f"Number of folds run: {len(splits)}")

	for position, (train_idx, test_idx) in enumerate(splits):
		fold = args.outer_fold if args.stage == "final" and args.outer_fold is not None else position
		train_examples = [examples[i] for i in train_idx]
		test_examples = [examples[i] for i in test_idx]
		tp, fp, tn, fn = run_fixed_config_fold(
			args,
			example_to_ndlm_sample,
			lp_name,
			f"{fold_prefix} {fold + 1}",
			train_examples,
			test_examples,
			problem_path,
			summary_log,
			f"fold_{fold + 1:02d}",
			args.split_seed + fold,
		)
		total_tp += tp
		total_fp += fp
		total_tn += tn
		total_fn += fn

	log_result_block(
		f"{lp_name} - {args.stage} {len(splits)}-fold results",
		total_tp,
		total_fp,
		total_tn,
		total_fn,
	)


def run_single_lp_hparam(args, example_to_ndlm_sample, experiment_path, lp_name, examples):
	problem_path = experiment_path / lp_name / f"run_{time.time_ns()}"
	summary_log = problem_path / "summary.log"
	init_logger(summary_log)

	labels = [example["label"] for example in examples]
	outer_splits = make_stratified_splits(examples, labels, args.outer_folds, args.split_seed)
	if args.outer_fold is not None:
		if not 0 <= args.outer_fold < len(outer_splits):
			raise ValueError(f"--outer-fold must be between 0 and {len(outer_splits) - 1}")
		outer_splits = [(args.outer_fold, outer_splits[args.outer_fold])]
	else:
		outer_splits = list(enumerate(outer_splits))

	total_tp = total_fp = total_tn = total_fn = 0
	log(f"Single-LP hparam stage: {lp_name}")
	log(f"Outer folds run: {len(outer_splits)}")

	for outer_fold, (outer_train_idx, _) in outer_splits:
		outer_train_examples = [examples[i] for i in outer_train_idx]
		outer_train_labels = [example["label"] for example in outer_train_examples]
		inner_splits = make_stratified_splits(
			outer_train_examples,
			outer_train_labels,
			args.inner_folds,
			args.split_seed + outer_fold + 1,
		)

		outer_tp = outer_fp = outer_tn = outer_fn = 0
		for inner_fold, (train_idx, dev_idx) in enumerate(inner_splits):
			train_examples = [outer_train_examples[i] for i in train_idx]
			dev_examples = [outer_train_examples[i] for i in dev_idx]
			tp, fp, tn, fn = run_fixed_config_fold(
				args,
				example_to_ndlm_sample,
				lp_name,
				f"Outer {outer_fold + 1} Inner Fold {inner_fold + 1}",
				train_examples,
				dev_examples,
				problem_path,
				summary_log,
				f"outer_{outer_fold + 1:02d}_inner_{inner_fold + 1:02d}",
				args.split_seed + 1000 * outer_fold + inner_fold,
			)
			outer_tp += tp
			outer_fp += fp
			outer_tn += tn
			outer_fn += fn

		log_result_block(
			f"{lp_name} - outer fold {outer_fold + 1} inner-CV hparam score",
			outer_tp,
			outer_fp,
			outer_tn,
			outer_fn,
		)
		total_tp += outer_tp
		total_fp += outer_fp
		total_tn += outer_tn
		total_fn += outer_fn

	log_result_block(f"{lp_name} - hparam aggregate", total_tp, total_fp, total_tn, total_fn)


def main():
	args = parse_args()
	domain_owl_file = owl_files[args.name]
	lp_path = Path(lp_files[args.name])
	problems = benchmark_owl_ndlm.load_lp_examples(lp_path)
	problem_items = list(problems.items())

	if args.problem_index is not None:
		if not 0 <= args.problem_index < len(problem_items):
			raise ValueError(
				f"--problem-index must be between 0 and {len(problem_items) - 1}, "
				f"got {args.problem_index}"
			)
		problem_items = [problem_items[args.problem_index]]

	is_single_lp = len(problem_items) == 1
	if args.stage != "legacy" and not is_single_lp and args.problem_index is None:
		dev_items, test_items = train_test_split(
			problem_items,
			test_size=args.lp_test_fraction,
			shuffle=True,
			random_state=args.split_seed,
		)
		problem_items = dev_items if args.stage == "hparam" else test_items

	stage_config_name = args.configuration_name if args.stage == "legacy" else f"{args.configuration_name}_{args.stage}"
	experiment_path = (
		Path("outputs/CONCEPT_LEARNER")
		/ args.output_root
		/ stage_config_name
		/ args.name
		/ f"neighborhood_{args.neighborhood}"
	)
	example_to_ndlm_sample = make_sample_cache(args, domain_owl_file)

	if is_single_lp and args.stage == "hparam":
		lp_name, examples = problem_items[0]
		run_single_lp_hparam(args, example_to_ndlm_sample, experiment_path, lp_name, examples)
		return

	fold_prefix = "Outer Fold" if is_single_lp and args.stage == "final" else "Fold"
	for lp_name, examples in problem_items:
		run_problem_cv(args, example_to_ndlm_sample, experiment_path, lp_name, examples, fold_prefix)


if __name__ == "__main__":
	main()

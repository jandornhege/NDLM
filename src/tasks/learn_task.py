import ast
import csv
from pathlib import Path
import resource

import hard_instance_generator
import data_encoding
import ActionModel_data_encoding
import Pruning

import argparse
from difflogic import nn
import ndlm.main as NDLM_main
import ndlm.configs as config
import ndlm.modules as modules
import sys
import os
import torch
import time
import difflogic.nn.neural_logic.layer as layer
from my_logging import log, init_logger

def export_args_to_file(args, file_path):
    with open(file_path, 'w') as f:
        for arg in vars(args):
            if arg == "experiment_path":
                f.write(f"{arg}: {repr(str(getattr(args, arg)))}\n")
            else:
                f.write(f"{arg}: {repr(getattr(args, arg))}\n")
        
def import_args_object_from_file(file_path):
    args = argparse.Namespace()
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ', 1)
            setattr(args, key, ast.literal_eval(value))
    return args


def merge_train_test_actions(train, test):
    """Keep only actions available in both train and test after pruning."""
    train_actions = set(train.keys())
    test_actions = set(test.keys())
    common_actions = sorted(train_actions.intersection(test_actions))
    
    dropped_train_only = sorted(train_actions - test_actions)
    dropped_test_only = sorted(test_actions - train_actions)
    if dropped_train_only:
        log(f"Skipping train-only actions after pruning: {dropped_train_only}")
    if dropped_test_only:
        log(f"Skipping test-only actions after pruning: {dropped_test_only}")
    for action in train_actions.union(test_actions):
        if action not in train:
            train[action] = []
        if action not in test:
            test[action] = []
    
    return {action: (train[action], test[action]) for action in train_actions}

def datapoints_to_dataset(train, test):
    train_dataset = []
    for dataset in train:
        c = torch.stack([item[0] for item in dataset])  # (num_samples, num_concepts, num_objects)
        r = torch.stack([item[1] for item in dataset])      # (num_samples, num_roles, num_objects, num_objects)
        c_targets = torch.stack([item[2] for item in dataset])    # (num_samples, num_out_concepts, num_objects)
        r_targets = torch.stack([item[3] for item in dataset])    # (num_samples, num_out_roles, num_objects, num_objects)
        train_dataset.append((c, r, c_targets, r_targets))
        log(f"Train shapes: {c.shape}, {r.shape}, {c_targets.shape}, {r_targets.shape}")
    
    test_dataset = []
    for dataset in test:
        c = torch.stack([item[0] for item in dataset])  # (num_samples, num_concepts, num_objects)
        r = torch.stack([item[1] for item in dataset])      # (num_samples, num_roles, num_objects, num_objects)
        c_targets = torch.stack([item[2] for item in dataset])    # (num_samples, num_out_concepts, num_objects)
        r_targets = torch.stack([item[3] for item in dataset])    # (num_samples, num_out_roles, num_objects, num_objects)
        test_dataset.append((c, r, c_targets, r_targets))
        log(f"Test shapes: {c.shape}, {r.shape}, {c_targets.shape}, {r_targets.shape}")
        
    return train_dataset, test_dataset

def get_1D_ARC_dataset(args):
    task_name = args.arc_task_name
    all = args.all
    task = f"/work/rleap1/jan.dornhege/NDLM/src/data/1D-ARC-main/dataset/{task_name}/"
    if all:
        train, test = data_encoding.create_dataset_from_dir(task)
    else:
        train, test = data_encoding.create_dataset_from_file(task, task_name+"_0.json")
    
    train= [train]
    test = [test]
    
    train_dataset, test_dataset = datapoints_to_dataset(train, test)
    return {"act": (train_dataset, test_dataset)}

def get_Action_Model_learning_datasets(args):
    train, test = ActionModel_data_encoding.get_dataset(args)
    data_dir = merge_train_test_actions(train, test)
    for action, (train_data, test_data) in data_dir.items():
        train_dataset, test_dataset = datapoints_to_dataset(train_data, test_data)
        data_dir[action] = (train_dataset, test_dataset)
    return data_dir

def get_OPTIMAL_GP_dataset(args):
    train, test = ActionModel_data_encoding.get_OPTIMAL_GP_dataset(args)
    data_dir = merge_train_test_actions(train, test)
    for action, (train_data, test_data) in data_dir.items():
        train_dataset, test_dataset = datapoints_to_dataset(train_data, test_data)
        data_dir[action] = (train_dataset, test_dataset)
    return data_dir

def get_GP_dataset(domain):
    train, test = ActionModel_data_encoding.get_GP_dataset(domain)
    data_dir = merge_train_test_actions(train, test)
    for action, (train_data, test_data) in data_dir.items():
        train_dataset, test_dataset = datapoints_to_dataset(train_data, test_data)
        data_dir[action] = (train_dataset, test_dataset)
    return data_dir


def run_execution_test(args, models_per_action):
    """Evaluate trained GP action models on metadata-selected test problems."""
    domain_information = ActionModel_data_encoding.load_domain_information(args)
    test_paths = (
        domain_information["test_problem_paths"]
        + domain_information.get("eval_problem_paths", [])
    )
    if not test_paths:
        raise ValueError("No execution-test problem paths were selected by domain_information.json")

    results = ActionModel_data_encoding.random_trace_generator.test_model_on_test_problems(
        domain_information["domain_path"],
        test_paths,
        models_per_action,
        parameter_indices=domain_information["argument_indices"],
        max_steps=args.num_test_states,
        soft_policy=False,
        model_type=args.prediction_type,
        one_step_cycle_check=False,
    )
    successes = sum(result["goal_reached"] for result in results.values())
    optimal = sum(
        result["goal_reached"] and result["steps_taken"] == result["optimal_steps"]
        for result in results.values()
    )
    log(f"Execution test: {successes}/{len(results)} goals reached")
    log(f"Execution test optimal: {optimal}/{len(results)} optimal solutions")
    for path, result in results.items():
        log(
            f"Execution result: file={path}, steps_taken={result['steps_taken']}, "
            f"success={result['goal_reached']}, optimal_steps={result['optimal_steps']}"
        )
    return results

def get_hard_graphs_dataset(args):
    data_dir = hard_instance_generator.get_hard_graphs_dataset(args)
    for action, (train_data, test_data) in data_dir.items():
        train_dataset, test_dataset = datapoints_to_dataset([train_data], [test_data])
        data_dir[action] = (train_dataset, test_dataset)
    return data_dir


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def benchmark_forward_pass(model, concepts, roles, repeats=20, warmup=5):
    if concepts.dim() == 2:
        concepts = concepts.unsqueeze(0)
    if roles.dim() == 3:
        roles = roles.unsqueeze(0)

    device = concepts.device
    model.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            model(concepts, roles)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        for _ in range(repeats):
            model(concepts, roles)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    peak_cuda_mb = None
    if device.type == "cuda":
        peak_cuda_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return elapsed / repeats, peak_cuda_mb, peak_rss_mb


def build_model_for_hard_graphs(model_name, args, config_obj, in_concepts, in_roles, out_concepts, out_roles):
    if model_name == "NDLM":
        return modules.MultiLayerNDLM(in_concepts, in_roles, out_concepts, out_roles, config_obj)
    return layer.NLM_to_NDLM_Adapter(in_concepts, in_roles, out_concepts, out_roles, args)


def run_hard_graphs_benchmark(args, config_obj):
    lengths = [length for length in range(2, args.benchmark_max_length + 1, args.benchmark_interval_length)]
    model_names = [value.strip() for value in args.benchmark_models.split(",") if value.strip()]
    num_layers_values = [int(value) for value in args.benchmark_num_layers.split(",") if value.strip()]
    hidden_size_values = [int(value) for value in args.benchmark_hidden_sizes.split(",") if value.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for model_name in model_names:
        if model_name not in {"NDLM", "NLM"}:
            raise ValueError(f"Unknown benchmark model: {model_name}")

        for num_layers in num_layers_values:
            for hidden_size in hidden_size_values:
                benchmark_args = argparse.Namespace(**vars(args))
                benchmark_args.num_layers = num_layers
                benchmark_args.hidden_concepts = hidden_size
                benchmark_args.hidden_roles = hidden_size
                benchmark_args.model = model_name
                benchmark_args.ndlm_depth = num_layers
                benchmark_args.ndlm_hidden_concepts = hidden_size
                benchmark_args.ndlm_hidden_roles = hidden_size
                benchmark_args.ndlm_transitive_closure = args.transitive_closure
                benchmark_args.ndlm_mode = args.mode
                benchmark_args.ndlm_activation_function = args.activation_function
                benchmark_args.ndlm_residual = args.residual
                benchmark_args.ndlm_input_residual = args.input_residual

                benchmark_config = config.config_object()
                benchmark_config.NUM_HIDDEN_CONCEPTS = hidden_size
                benchmark_config.NUM_HIDDEN_ROLES = hidden_size
                benchmark_config.NUM_LAYERS = num_layers
                benchmark_config.ACTIVATION_FUNCTION = torch.nn.Identity() if args.activation_function == "identity" else torch.nn.Sigmoid()
                benchmark_config.RESIDUAL = args.residual
                benchmark_config.MODE = args.mode
                benchmark_config.TRANSITIVE_CLOSURE = args.transitive_closure
                benchmark_config.LEARNING_RATE = args.learning_rate
                benchmark_config.WEIGHT_DECAY = args.weight_decay
                benchmark_config.INPUT_RESIDUAL = args.input_residual
                benchmark_config.SKIP_CONNECTIONS = False
                benchmark_config.RAGG_NORM = False
                benchmark_config.RRA_NORM = False
                benchmark_config.CRA_NORM = False
                benchmark_config.LAYER_TYPE = "single_step_minmax" if args.mode == "strict" else "single_step"

                for length in lengths:
                    in_concepts = 16
                    in_roles = 7
                    out_concepts = 0 
                    out_roles = 1

                    concepts = torch.zeros((in_concepts, length), dtype=torch.float32, device=device)
                    roles = torch.zeros((in_roles, length, length), dtype=torch.float32, device=device)
                    c_targets = torch.zeros((out_concepts, length), dtype=torch.float32, device=device)
                    r_targets = torch.zeros((out_roles, length, length), dtype=torch.float32, device=device)

                    # concepts, roles, c_targets, r_targets = hard_instance_generator.generate_line_problem(length=length)[0]
                    # Raw hard-graph tensors are unbatched: concepts [C, N], roles [R, N, N].
                    # in_concepts = concepts.shape[0]
                    # in_roles = roles.shape[0]
                    # out_concepts = c_targets.shape[0]
                    # out_roles = r_targets.shape[0]

                    benchmark_args.io_dimensions = {
                        "benchmark": {
                            "in_concepts": in_concepts,
                            "in_roles": in_roles,
                            "out_concepts": out_concepts,
                            "out_roles": out_roles,
                        }
                    }
                    benchmark_config.IN_CONCEPTS = in_concepts
                    benchmark_config.IN_ROLES = in_roles
                    benchmark_config.OUT_CONCEPTS = out_concepts
                    benchmark_config.OUT_ROLES = out_roles

                    model = build_model_for_hard_graphs(model_name, benchmark_args, benchmark_config, in_concepts, in_roles, out_concepts, out_roles).to(device)
                    # Keep the exact hard-graphs tensor convention used in training: [B,C,N], [B,R,N,N].
                    concepts = concepts.unsqueeze(0).to(device)
                    roles = roles.unsqueeze(0).to(device)

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    avg_time_s, peak_cuda_mb, peak_rss_mb = benchmark_forward_pass(
                        model,
                        concepts,
                        roles,
                        repeats=args.benchmark_repeats,
                        warmup=args.benchmark_warmup,
                    )

                    row = {
                        "model_name": model_name,
                        "num_layers": num_layers,
                        "hidden_size": hidden_size,
                        "line_length": length,
                        "num_objects": concepts.shape[-1],
                        "param_count": count_parameters(model),
                        "avg_inference_time_s": avg_time_s,
                        "peak_cuda_allocated_mb": peak_cuda_mb,
                        "peak_rss_mb": peak_rss_mb,
                    }
                    rows.append(row)
                    log(
                        "Benchmark | "
                        f"model={row['model_name']} | layers={row['num_layers']} | hidden={row['hidden_size']} | "
                        f"objects={row['num_objects']} | params={row['param_count']} | "
                        f"time={row['avg_inference_time_s']:.6f}s | "
                        f"cuda_peak_mb={row['peak_cuda_allocated_mb']} | rss_peak_mb={row['peak_rss_mb']:.2f}"
                    )

    if args.benchmark_output:
        output_path = Path(args.benchmark_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        log(f"Saved benchmark results to: {output_path}")
    else:
        print("model_name,num_layers,hidden_size,line_length,num_objects,param_count,avg_inference_time_s,peak_cuda_allocated_mb,peak_rss_mb")
        for row in rows:
            peak_cuda_value = "" if row["peak_cuda_allocated_mb"] is None else f"{row['peak_cuda_allocated_mb']:.4f}"
            print(
                f"{row['model_name']},{row['num_layers']},{row['hidden_size']},{row['line_length']},{row['num_objects']},{row['param_count']},"
                f"{row['avg_inference_time_s']:.8f},{peak_cuda_value},{row['peak_rss_mb']:.4f}"
            )

if __name__ == "__main__":

    # Experiment hyperparameters
    parser = argparse.ArgumentParser(description="Runs NDLM tasks using model(NLM or NDLM) on 1D-ARC or Action Model learning datasets.")
    parser.add_argument("--model", default="NDLM", choices=["NDLM", "NLM"], help="Model to use currently: NDLM or NLM.")
    parser.add_argument("--task", type=str, choices=["1DARC", "ActionModel", "optGP", "GP", "hard_graphs"], help="'1DARC', 'ActionModel', 'optGP', 'GP', or 'hard_graphs'")
    parser.add_argument("--dump-dir", type=str, help="Directory to dump results.")
    parser.add_argument("--config-file", type=str, help="Path to a JSON config file to load configuration parameters from.")
    parser.add_argument(
        "--max-sampling-seconds-per-problem",
        type=float,
        default=None,
        help="Optional per-problem time limit (seconds) for BFS-based sampling.",
    )
    parser.add_argument("--data-path", type=str, default="/work/rleap1/jan.dornhege/NDLM/src/data/", help="Path to the data directory.")
    # 1D-ARC task hyperparameters
    parser.add_argument("--arc-task-name", type=str, help="Name of ARC task")
    parser.add_argument("--all", action="store_true", help="Load all datasets in the directory for 1D-ARC tasks.")

    # Planning tasks hyperparameters
    parser.add_argument("--domain", type=str, help="Name of the domain for Action Model learning tasks.")
    parser.add_argument("--num-train-states", type=int, default=100, help="Number of training states to sample for planning tasks.")
    parser.add_argument("--num-test-states", type=int, default=100, help="Number of testing states to sample for planning tasks.")
    

    # Action Model learning task hyperparameters
    parser.add_argument("--precondition", action="store_true", help="Load datasets for learning preconditions in Action Model learning tasks.")
    parser.add_argument("--full-applicability", action="store_true", help="Load datasets with full applicability in Action Model learning tasks.")
    parser.add_argument("--remove-predicates", action="store_true", help="Remove certain predicates from the Action Model learning datasets.")
    parser.add_argument("--remove-arguments", action="store_true", help="Remove certain arguments from the Action Model learning datasets.")
    parser.add_argument("--type-filtering", action="store_true", help="Remove certain objects from the Action Model learning datasets.")
    parser.add_argument("--max-actions", type=int, default=10, help="Number of non-applicable grounded actions to include in the Action Model learning datasets.")
    parser.add_argument("--ignore-type-filtering", action="store_true", default=False, help="Ignore type filtering in the Action Model learning datasets.")
    # optGP and GP task hyperparameters
    parser.add_argument("--prediction_type", default="state_full", choices=["state_action_nullary", "state_pair_nullary", "state_full"], help="Type of prediction for optGP and GP tasks.")
    parser.add_argument("--test_by_execution", action="store_true", help="Whether to test by execution for optGP and GP tasks.")
    parser.add_argument("--test_supervised", action="store_true", help="Whether to use supervised testset.")
    parser.add_argument(
        "--combined-test",
        action="store_true",
        help=(
            "Run training with the supervised train/test evaluation in one job. "
            "This does not run execution testing; use --test-only for the "
            "separate checkpoint execution workflow."
        ),
    )
    parser.add_argument("--sampling-method", default="bfs", choices=["bfs", "opt_then_bfs"], help="Sampling method for optGP and GP tasks.")

    # Hard graphs task hyperparameters
    parser.add_argument("--graph-problem-type", type=str, default="line", choices=["cycle", "line", "star"], help="Type of graph problem for hard graphs task.")
    parser.add_argument("--line-max-length", type=int, default=None, help="Maximum length of the graph problem for hard graphs task.")
    parser.add_argument("--line-min-length", type=int, default=None, help="Minimum length of the graph problem for hard graphs task.")

    # General model config hyperparameters
    parser.add_argument("--hidden-concepts", type=int, default=10, help="Hidden size of the model.")
    parser.add_argument("--hidden-roles", type=int, default=10, help="Hidden size of the model.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers in the model.")
    parser.add_argument("--activation-function", type=str, default="identity", choices=["identity", "sigmoid"], help="Activation function to use in the model.")
    
    # NDLM model config hyperparameters
    parser.add_argument("--mode", type=str, default="strict", choices=["strict", "relaxed"], help="Mode of the NDLM model.")
    parser.add_argument("--transitive-closure", action="store_true", help="Whether to use transitive closure in the model.")
    parser.add_argument("--residual", action="store_true", help="Whether to use residual connections in the model.")
    parser.add_argument("--input-residual", action="store_true", help="Whether to use input residual connections in the model.")
    
    # NLM specific model config hyperparameters
    parser.add_argument("--nlm-breadth", type=int, default=3, help="Breadth of the NLM model.")
    parser.add_argument("--nlm-exclude-self", action="store_true", help="Whether to exclude self in the NLM model.")
    parser.add_argument("--nlm-residual", action="store_true", help="Whether to use residual connections in the NLM model.")


    # Training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay for the optimizer.")
    parser.add_argument("--num-epochs", type=int, default=200, help="Number of epochs for training.")
    parser.add_argument("--test-interval", type=int, default=10, help="Interval (in epochs) to run tests during training.")     
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--weighted_loss", action="store_true", help="Whether to use weighted loss.")
    parser.add_argument("--loss-type", type=str, default="BCE", choices=["BCE", "Noisy_OR", "SoftmaxSet"], help="Type of loss to use: 'weighted' or 'unweighted'.")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Number of epochs with no improvement after which training will be stopped.")


    # Test options
    parser.add_argument("--test-only", action="store_true", help="Whether to only run tests without training.")
    parser.add_argument("--prune-model", action="store_true", help="Whether to apply model-pruning approach.")
    parser.add_argument("--checkpoint-path", type=str, help="Path to a checkpoint file for testing.")
    parser.add_argument("--benchmark-inference", action="store_true", help="Benchmark inference time and memory for hard_graphs inputs.")
    parser.add_argument(
        "--benchmark-max-length",
        type=int,
        default=150,
        help="Maximum line length to benchmark for hard_graphs.",
    )
    parser.add_argument(
        "--benchmark-interval-length",
        type=int,
        default=1,
        help="Interval in num objects to run benchmarks during training.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=20,
        help="Number of measured inference repetitions per size.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=5,
        help="Number of warmup inference passes per size.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=None,
        help="Optional CSV output path for benchmark results.",
    )
    parser.add_argument(
        "--benchmark-models",
        type=str,
        default="NDLM,NLM",
        help="Comma-separated model names to benchmark: NDLM, NLM, or both.",
    )
    parser.add_argument(
        "--benchmark-num-layers",
        type=str,
        default="2,4,6,8",
        help="Comma-separated num_layers values to benchmark.",
    )
    parser.add_argument(
        "--benchmark-hidden-sizes",
        type=str,
        default="5,8,10,15",
        help="Comma-separated hidden size values to benchmark for both concepts and roles.",
    )

    args = parser.parse_args()
    if args.precondition:
        args.target_mode = "applicability_vector"
    else:
        args.target_mode = "state"

    if args.combined_test and args.task not in {"GP", "optGP"}:
        parser.error("--combined-test is only supported with --task GP or --task optGP")
    if args.combined_test:
        args.test_supervised = True

    if args.benchmark_inference:
        if args.task != "hard_graphs":
            raise ValueError("--benchmark-inference currently supports only --task hard_graphs.")

        if args.dump_dir:
            args.experiment_name = f"benchmark_{int(time.time())}"
            args.experiment_path = Path(f"{args.dump_dir}/{args.experiment_name}")
            os.makedirs(args.experiment_path, exist_ok=True)
            init_logger(f"{args.experiment_path}/output.log")
            if args.benchmark_output is None:
                args.benchmark_output = str(args.experiment_path / "benchmark.csv")

        c = config.config_object()
        c.NUM_HIDDEN_CONCEPTS = args.hidden_concepts
        c.NUM_HIDDEN_ROLES = args.hidden_roles
        c.NUM_LAYERS = args.num_layers
        c.ACTIVATION_FUNCTION = torch.nn.Identity() if args.activation_function == "identity" else torch.nn.Sigmoid()
        c.RESIDUAL = args.residual
        c.MODE = args.mode
        c.TRANSITIVE_CLOSURE = args.transitive_closure
        c.LEARNING_RATE = args.learning_rate
        c.WEIGHT_DECAY = args.weight_decay
        c.INPUT_RESIDUAL = args.input_residual

        run_hard_graphs_benchmark(args, c)
        raise SystemExit(0)

    if args.test_only:
        if args.dump_dir:
            checkpoint_last_dir = str(args.checkpoint_path).split("/")[-2]
            args.experiment_name = str(time.time())
            args.experiment_path = Path(f"{args.dump_dir}/{args.experiment_name}")
            os.makedirs(args.experiment_path, exist_ok=True)
            init_logger(f"{args.dump_dir}/{args.experiment_name}/output.log")

                
        if not args.checkpoint_path:
            raise ValueError("Checkpoint path must be provided for test-only mode.")
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
        log(f"Loading checkpoint from: {args.checkpoint_path}")

        args_file = Path(args.checkpoint_path) / "args.txt"
        checkpoint_args = import_args_object_from_file(args_file)
        config_file = Path(args.checkpoint_path) / "config.json"
        c = config.config_from_json_file(config_file)
        ActionModel_data_encoding.test_policy_from_checkpoint_models(checkpoint_args, args, c)

    elif args.prune_model:
        if not args.checkpoint_path:
            raise ValueError("Checkpoint path must be provided for model pruning.")
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
        if args.dump_dir:
            checkpoint_last_dir = str(args.checkpoint_path).split("/")[-2]
            args.experiment_name = checkpoint_last_dir
            args.experiment_path = Path(f"{args.dump_dir}/{args.experiment_name}")
            os.makedirs(args.experiment_path, exist_ok=True)
            init_logger(f"{args.dump_dir}/{args.experiment_name}/output.log")
        log(f"Loading checkpoint from: {args.checkpoint_path}")    
        args_file = Path(args.checkpoint_path) / "args.txt"
        checkpoint_args = import_args_object_from_file(args_file)
        if hasattr(args, "num_test_states"):
            log(f"Number of test states: {args.num_test_states}")
            checkpoint_args.num_test_states = args.num_test_states
        
        config_file = Path(args.checkpoint_path) / "config.json"
        c = config.config_from_json_file(config_file)
        if checkpoint_args.task == "1DARC":
            datasets = get_1D_ARC_dataset(checkpoint_args)
        elif checkpoint_args.task == "ActionModel":
            datasets = get_Action_Model_learning_datasets(checkpoint_args)
        elif checkpoint_args.task == "optGP":
            datasets = get_OPTIMAL_GP_dataset(checkpoint_args)
        elif checkpoint_args.task == "GP":
            datasets = get_GP_dataset(checkpoint_args)
        elif checkpoint_args.task == "hard_graphs":
            datasets = get_hard_graphs_dataset(checkpoint_args)
        else:
            raise ValueError(f"Unknown task: {checkpoint_args.task}")
        log("Data loaded")
        Pruning.prune_model(checkpoint_args, args, c, datasets)

        


    if not args.test_only:
        args.experiment_name = f"{int(time.time())}"
        if args.dump_dir:
            args.experiment_path = Path(f"{args.dump_dir}/{args.experiment_name}")
            os.makedirs(args.experiment_path, exist_ok=True)
        else:
            args.experiment_path = Path(f"./tmp/{args.experiment_name}")
            os.makedirs(args.experiment_path, exist_ok=True)

        if args.dump_dir:
            init_logger(f"{args.dump_dir}/{args.experiment_name}/output.log")
        
           
        if args.config_file:
            c = config.config_from_json_file(args.config_file)
        else:
            c = config.config_object()
            c.NUM_HIDDEN_CONCEPTS = args.hidden_concepts
            c.NUM_HIDDEN_ROLES = args.hidden_roles
            c.NUM_LAYERS = args.num_layers
            c.ACTIVATION_FUNCTION = torch.nn.Identity() if args.activation_function == "identity" else torch.nn.Sigmoid()
            c.RESIDUAL = args.residual
            c.MODE = args.mode
            c.NUM_EPOCHS = args.num_epochs
            c.TEST_INTERVAL = args.test_interval
            c.TRANSITIVE_CLOSURE = args.transitive_closure
            c.LEARNING_RATE = args.learning_rate
            c.WEIGHT_DECAY = args.weight_decay
            c.INPUT_RESIDUAL = args.input_residual
        
        
        
        c.save_to_file(args.experiment_path / f"config.json")
        if args.task == "1DARC":
            datasets = get_1D_ARC_dataset(args)
        elif args.task == "ActionModel":
            datasets = get_Action_Model_learning_datasets(args)
        elif args.task == "optGP":
            datasets = get_OPTIMAL_GP_dataset(args)
        elif args.task == "GP":
            datasets = get_GP_dataset(args)
        elif args.task == "hard_graphs":
            datasets = get_hard_graphs_dataset(args)
        else:
            raise ValueError(f"Unknown task: {args.task}")
        log("Data loaded")

        log(f"Actions: {datasets.keys()}")
        args.io_dimensions = {}
        for action, (train, test) in datasets.items():
            log(f"Action: {action}, Train size: {len(train)}, Test size: {len(test)}")
            log(f"in_concepts: {train[0][0].shape[1]}, in_roles: {train[0][1].shape[1]}, out_concepts: {train[0][2].shape[1]}, out_roles: {train[0][3].shape[1]}")
            args.io_dimensions[action] = {
                "in_concepts": train[0][0].shape[1],
                "in_roles": train[0][1].shape[1],
                "out_concepts": train[0][2].shape[1],
                "out_roles": train[0][3].shape[1]
            }
        log(f"Starting task: {args.task}, Name: {args.experiment_name}, Model: {args.model}")
        export_args_to_file(args, args.experiment_path / f"args.txt")
        action_names=list(datasets.keys())
        action_names.sort()
        models_per_action = {}
        
        for action in action_names:
            train, test = datasets[action]
            if args.dump_dir:
                init_logger(f"{args.dump_dir}/{args.experiment_name}/output_{action}.log")
            log(f"Starting training for action: {action}, Train size: {len(train)}, Test size: {len(test)}")
            in_concepts = train[0][0].shape[1]
            in_roles = train[0][1].shape[1]
            out_concepts = train[0][2].shape[1]
            out_roles = train[0][3].shape[1]
            if args.model == "NDLM":
                model = modules.MultiLayerNDLM(in_concepts, in_roles, out_concepts, out_roles, c)
            else:
                model = layer.NLM_to_NDLM_Adapter(in_concepts, in_roles, out_concepts, out_roles, args)
            models_per_action[action] = model
            
            args.return_details = True
            action_start_time = time.perf_counter()
            result = NDLM_main.main(
                train,
                test,
                c,
                args,
                model,
                checkpoint_path=(
                    args.experiment_path / f"{action}/checkpoints"
                    if args.experiment_path
                    else None
                ),
                log=log,
            )
            train_results = result["train"]
            test_totals = result["test"]["totals"]
            tp_c = test_totals["tp_c"]
            tn_c = test_totals["tn_c"]
            fp_c = test_totals["fp_c"]
            fn_c = test_totals["fn_c"]
            tp_r = test_totals["tp_r"]
            tn_r = test_totals["tn_r"]
            fp_r = test_totals["fp_r"]
            fn_r = test_totals["fn_r"]
            
            if args.dump_dir:
                init_logger(f"{args.dump_dir}/{args.experiment_name}/output.log")
            log(f"Finished training for action: {action}, Train size: {len(train)}, Test size: {len(test)}")
            log(f"Final Train Results for action {action}: {train_results}")
            log(
                f"Final Test Results for action {action}: "
                f"{result['test']}"
            )
            train_totals = train_results["totals"]
            test_totals = result["test"]["totals"]
            log(
                f"Overview misclassifications for action {action}: "
                f"train #c_miss={train_totals['c_miss']}, "
                f"train #r_miss={train_totals['r_miss']}, "
                f"test #c_miss={test_totals['fp_c'] + test_totals['fn_c']}, "
                f"test #r_miss={test_totals['fp_r'] + test_totals['fn_r']}"
            )
            log(
                f"Elapsed training time for action {action}: "
                f"{time.perf_counter() - action_start_time:.4f} seconds"
            )


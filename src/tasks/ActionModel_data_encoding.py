from my_logging import log
from pyexpat import model

from jacinle import config
import random_trace_generator.main as random_trace_generator
import os
import json
import sys
import pymimir.advanced.formalism as formalism
import pymimir.advanced.search as search
import torch
import json
# import ffn_main_general_data as ffn_main
import ndlm.baselines as baselines
import ndlm.configs as Config
import ndlm.modules as modules
import time
from pathlib import Path
import difflogic.nn.neural_logic.layer as layer
import argparse

import json


def load_moose_dataset_paths(domain_name, max_states_train, max_states_test, parameter_domain=None):
    base_dir = Path("/work/rleap1/jan.dornhege/moose-dataset") / domain_name
    domain_path = base_dir / "domain.pddl"
    train_dir = base_dir / "training"
    test_dir = base_dir / "testing"

    train_problem_paths = sorted(str(p) for p in train_dir.glob("*.pddl"))
    test_problem_paths = sorted(str(p) for p in test_dir.glob("*.pddl"))

    if not domain_path.exists():
        raise FileNotFoundError(f"MOOSE domain file not found: {domain_path}")
    if not train_problem_paths:
        raise FileNotFoundError(f"No training PDDL files found in: {train_dir}")
    if not test_problem_paths:
        raise FileNotFoundError(f"No testing PDDL files found in: {test_dir}")

    if parameter_domain is None:
        return (
            domain_name,
            str(domain_path),
            train_problem_paths,
            test_problem_paths,
            max_states_train,
            max_states_test,
        )

    return (
        domain_name,
        str(domain_path),
        train_problem_paths,
        test_problem_paths,
        max_states_train,
        max_states_test,
        parameter_domain,
    )

def export_domain_information(domain_information, args):
    export_path = Path(args.experiment_path) / "domain_information.json"
    with open(export_path, 'w') as f:
        json.dump(domain_information, f, indent=4)
    log(f"Domain information exported to {export_path}")

def load_domain_information(args):
    base_path = Path(args.data_path)
    domain_name = args.domain
    domain_root = base_path / domain_name
    domain_information = json.load(open(domain_root / "domain_information.json", "r"))
    domain_path = domain_root / domain_information.get("domain_file", "domain.pddl")

    train_dir = domain_root / "train"
    test_dir = domain_root / "test"
    train_problem_paths = [
        str(
            train_dir / name
            if (train_dir / name).is_file()
            else domain_root / name
        )
        for name in domain_information["train_instance_names"]
    ]
    test_problem_paths = [
        str(
            test_dir / name
            if (test_dir / name).is_file()
            else domain_root / name
        )
        for name in domain_information["test_instance_names"]
    ]
    print(args.remove_arguments, args.remove_predicates, flush=True)
    if args.remove_arguments:

        argument_indices = domain_information["argument_indices"]
        if not argument_indices:
            raise ValueError(f"Argument indices to remove not specified in domain_information.json for domain {domain_name}.")
    else:
        argument_indices = None
    if args.remove_predicates:
        predicates = domain_information["restricted_predicates"]
        if not predicates:
            raise ValueError(f"Predicate indices to remove not specified in domain_information.json for domain {domain_name}.")
    else:
        predicates = set()
    # action_arity = domain_information.get("action_arity", {})
    # Only consider instances specified in the domain_information.json file if not specified take all instances 
    train_instance_names = domain_information["train_instance_names"]
    if train_instance_names:
        train_problem_paths = [p for p in train_problem_paths if any(name in p for name in train_instance_names)]
    test_instance_names = domain_information["test_instance_names"]
    if test_instance_names:
        test_problem_paths = [p for p in test_problem_paths if any(name in p for name in test_instance_names)]
    eval_instance_names = domain_information.get("eval_instance_names", [])
    if eval_instance_names:
        eval_problem_paths = [base_path / domain_name / "test" / p for p in eval_instance_names]
    else:
        eval_problem_paths = []
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain file not found: {domain_path}")
    if not train_problem_paths:
        raise FileNotFoundError(f"No training PDDL files found in: {Path(base_path) / domain_name / 'train'}")
    if not test_problem_paths:
        raise FileNotFoundError(f"No testing PDDL files found in: {Path(base_path) / domain_name / 'test'}")
     
    return {
        "domain": domain_name,
        "domain_path": str(domain_path),
        "train_problem_paths": train_problem_paths,
        "test_problem_paths": test_problem_paths,
        "eval_problem_paths": eval_problem_paths,
        "argument_indices": argument_indices,
        "predicates": predicates,
        # "action_arity": action_arity,
    }

# clear_domain_path = "/work/rleap1/jan.dornhege/MA/data/clear/domain.pddl" 
# clear_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/clear/test/_training_clear_5.pddl"
# clear_train_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/clear/test/test_clear_probBLOCKS-06-1.pddl"
# clear_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/clear/test/test_clear_probBLOCKS-10-0.pddl"
# clear_test_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/clear/test/test_clear_probBLOCKS-10-1.pddl"
# clear_test_problem_path_3 = "/work/rleap1/jan.dornhege/MA/data/clear/test/test_clear_probBLOCKS-16-2.pddl"

# npuzzle_domain_path = "/work/rleap1/jan.dornhege/graph_separator/pddl_files/npuzzle/cell-npuzzle.pddl"
# npuzzle_train_problem_path = "/work/rleap1/jan.dornhege/graph_separator/pddl_files/npuzzle/cell-npuzzle-3-2.pddl"
# npuzzle_test_problem_path = "/work/rleap1/jan.dornhege/graph_separator/pddl_files/npuzzle/cell-npuzzle-3-3.pddl"
# npuzzle_test_problem_path_2 = "/work/rleap1/jan.dornhege/graph_separator/pddl_files/npuzzle/cell-npuzzle-4-2.pddl"

# satellite_domain_path = "/work/rleap1/jan.dornhege/MA/data/satellite/satellite.pddl"
# satellite_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/satellite/satellite_9.pddl"
# satellite_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/satellite/satellite_1.pddl"

# logistics_domain_path = "data/logistics/logistics.pddl"
# logistics_train_problem_path = "data/logistics/logistics_2_2_2_2_2_2.pddl"
# logistics_train_problem_path_2 = "data/logistics/logistics_1_2_2_2_2_2.pddl"
# logistics_test_problem_path = "data/logistics/logistics_3_3_3_3_3_3.pddl"
# logistics_test_problem_path_2 = "data/logistics/logistics-1.pddl"


# spanner_domain_path = "/work/rleap1/jan.dornhege/MA/data/spanner/domain.pddl"
# spanner_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/spanner/train/prob-2-2-10.pddl"
# spanner_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/spanner/test/prob-4-2-5.pddl"

# delivery_domain_path = "/work/rleap1/jan.dornhege/MA/data/delivery/domain.pddl"
# delivery_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/delivery/train/instance_4_2_0.pddl"
# delivery_train_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/delivery/train/instance_3_3_0.pddl"
# delivery_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/delivery/test/instance_5_3_0.pddl"
# delivery_test_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/delivery/test/instance_7_2_0.pddl"


# miconic_domain_path = "/work/rleap1/jan.dornhege/MA/data/miconic/domain.pddl"
# miconic_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/miconic/train/training2.pddl"
# miconic_train_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/miconic/train/s4-0.pddl"
# miconic_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s01-0.pddl"
# miconic_test_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s04-1.pddl"
# miconic_test_problem_path_3 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s04-2.pddl"
# miconic_test_problem_path_4 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s07-0.pddl"
# miconic_test_problem_path_5 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s07-1.pddl"
# miconic_test_problem_path_6 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s10-0.pddl"
# miconic_test_problem_path_7 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s13-0.pddl"


# gripper_domain_path = "/work/rleap1/jan.dornhege/moose-dataset/gripper/domain.pddl"
# gripper_train_problem_path = "/work/rleap1/jan.dornhege/moose-dataset/gripper/training/prob01.pddl"

# gripper_test_problem_path_3 = "/work/rleap1/jan.dornhege/moose-dataset/gripper/testing/prob03.pddl"
# gripper_test_problem_path_4 = "/work/rleap1/jan.dornhege/moose-dataset/gripper/testing/prob04.pddl"
# gripper_test_problem_path_5 = "/work/rleap1/jan.dornhege/moose-dataset/gripper/testing/prob05.pddl"
# gripper_test_problem_path_6 = "/work/rleap1/jan.dornhege/moose-dataset/gripper/testing/prob06.pddl"


# gripper_single_goal_domain_path = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/domain.pddl"
# gripper_single_goal_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/training/p01.pddl"
# gripper_single_goal_train_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/training/p02.pddl"
# gripper_single_goal_train_problem_path_3 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/training/p03.pddl"

# gripper_single_goal_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_01.pddl"
# gripper_single_goal_test_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_02.pddl"
# gripper_single_goal_test_problem_path_3 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_03.pddl"
# gripper_single_goal_test_problem_path_4 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_04.pddl"
# gripper_single_goal_test_problem_path_5 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_05.pddl"
# gripper_single_goal_test_problem_path_6 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_06.pddl"


# blocks_3_clear_towers_domain_path = "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_clear_towers/domain.pddl"
# blocks_3_clear_towers_train_problem_paths = [
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_clear_towers/train/instance_h2_clear_bottom.pddl",
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_clear_towers/train/instance_h3_clear_bottom.pddl",
# ]
# blocks_3_clear_towers_test_problem_paths = [
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_clear_towers/train/instance_h4_clear_bottom.pddl",
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_clear_towers/train/instance_h5_clear_bottom.pddl",
#     # "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_clear_towers/test/instance_h6_clear_bottom.pddl",
#     # "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_clear_towers/test/instance_h7_clear_bottom.pddl",
#     # "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_clear_towers/test/instance_h8_clear_bottom.pddl",
#     # "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_clear_towers/test/instance_h9_clear_bottom.pddl",
# ]

# blocks_3_on_pairs_domain_path = "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/domain.pddl"
# blocks_3_on_pairs_train_problem_paths = [
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/train/diff_towers_0.pddl",
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/train/diff_towers_1.pddl",
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/train/diff_towers_2.pddl",
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/train/simple.pddl",
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/train/instance_same_xy_goal_true.pddl",
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/train/instance_same_yx_goal_false.pddl",
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/train/instance_diff_towers_goal_false.pddl",

# ]
# blocks_3_on_pairs_test_problem_paths = [
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/test/instance_same_xy_goal_true_large.pddl",
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/test/instance_same_yx_goal_false_large.pddl",
#     "/work/rleap1/jan.dornhege/NDLM/src/data/blocks_3_on_pairs/test/instance_diff_towers_goal_false_large.pddl",
# ]


# data_paths={
#     "clear": ("BLOCKS", clear_domain_path, [clear_train_problem_path, clear_train_problem_path_2], [clear_test_problem_path, clear_test_problem_path_2, clear_test_problem_path_3], 100, 10),
#     "npuzzle": ("cpuzzle", npuzzle_domain_path, [npuzzle_train_problem_path], [npuzzle_test_problem_path, npuzzle_test_problem_path_2], 30,30),
#     # "satellite": ("satellite", satellite_domain_path, [satellite_train_problem_path], [satellite_test_problem_path]),
#     "logistics": ("logistics", logistics_domain_path, [logistics_train_problem_path], [logistics_test_problem_path],200, 3000),
#     # "spanner": ("spanner", spanner_domain_path, [spanner_train_problem_path], [spanner_test_problem_path]),
#     "delivery": ("delivery", delivery_domain_path, [delivery_train_problem_path, delivery_train_problem_path_2], [delivery_test_problem_path, delivery_test_problem_path_2], 30,50),
#     "logistics_1": ("logistics", logistics_domain_path, [logistics_train_problem_path, logistics_test_problem_path], [logistics_test_problem_path_2], 50,100),
#     "miconic": ("miconic", miconic_domain_path, [miconic_train_problem_path, miconic_train_problem_path_2], [
#         miconic_test_problem_path,
#         miconic_test_problem_path_2,
#         miconic_test_problem_path_4,
#         ], 50,3),
#     "gripper": ("gripper", gripper_domain_path, [gripper_train_problem_path, gripper_single_goal_train_problem_path_2, gripper_single_goal_train_problem_path_3], [
#         gripper_single_goal_test_problem_path,
#         # gripper_train_problem_path,
#         gripper_single_goal_test_problem_path_2,
#         gripper_single_goal_test_problem_path_3,
#         gripper_single_goal_test_problem_path_4,
#         gripper_single_goal_test_problem_path_5,
#         gripper_single_goal_test_problem_path_6
#     ], 10,10),
#     "blocks_3_on": (
#         "blocksworld",
#         blocks_3_on_pairs_domain_path,
#         blocks_3_on_pairs_train_problem_paths,
#         blocks_3_on_pairs_test_problem_paths,
#         1000,
#         100,
#         "blocks_3_on_pairs",
#     ),
    
# }
#         # miconic_test_problem_path_3,
#         # miconic_test_problem_path_5,
#         # miconic_test_problem_path_6,
#         # miconic_test_problem_path_7,
 
# opt_GP_data_paths={
#     "clear": ("BLOCKS", clear_domain_path, [clear_train_problem_path, clear_train_problem_path_2,clear_test_problem_path], [clear_train_problem_path_2, clear_test_problem_path, clear_test_problem_path_2,clear_test_problem_path_3], 200,200),
#     "logistics": ("logistics", logistics_domain_path, [logistics_train_problem_path, logistics_test_problem_path], [logistics_test_problem_path_2], 100,100),
#     "miconic": ("miconic", miconic_domain_path, [miconic_train_problem_path, miconic_train_problem_path_2], [
#         miconic_test_problem_path,
#         miconic_test_problem_path_2,
#         miconic_test_problem_path_4,
#         ], 800,800),
#     "gripper_single_goal": ("gripper", gripper_domain_path, [gripper_train_problem_path, gripper_single_goal_train_problem_path_2, gripper_single_goal_train_problem_path_3], [
#         gripper_single_goal_test_problem_path,
#         # gripper_train_problem_path,
#         gripper_single_goal_test_problem_path_2,
#         gripper_single_goal_test_problem_path_3,
#         gripper_single_goal_test_problem_path_4,
#         gripper_single_goal_test_problem_path_5,
#         gripper_single_goal_test_problem_path_6
#     ], 10,10),
#     "gripper": ("gripper", gripper_domain_path, [gripper_train_problem_path], [
#         gripper_test_problem_path_3,
#         gripper_test_problem_path_4,
#         gripper_test_problem_path_5,
#         gripper_test_problem_path_6
#     ], 30,10),
#     "npuzzle": ("cpuzzle", npuzzle_domain_path, [npuzzle_train_problem_path, npuzzle_test_problem_path, npuzzle_test_problem_path_2], [npuzzle_test_problem_path, npuzzle_test_problem_path_2], 40,100),    
#     "blocks_3_clear_towers": (
#         "blocksworld",
#         blocks_3_clear_towers_domain_path,
#         blocks_3_clear_towers_train_problem_paths,
#         blocks_3_clear_towers_test_problem_paths,
#         4,
#         8,
#         "blocks_3_clear_towers",
#     ),
#     "blocks_3_on_pairs": (
#         "blocksworld",
#         blocks_3_on_pairs_domain_path,
#         blocks_3_on_pairs_train_problem_paths,
#         blocks_3_on_pairs_test_problem_paths,
#         100,
#         100,
#         "blocks_3_on_pairs",
#     ),
#     "moose_ferry": load_moose_dataset_paths("ferry", 200, 100, parameter_domain="ferry"),
#     "on": load_full_path("on", "/work/rleap1/jan.dornhege/NDLM/src/data/on", 100, 10, parameter_domain="on"),
# }


# bw_predicates = set(["clear","handempty","holding","number","object","on","ontable"]) 
# delivery_predicates = set(["=", "adjacent","at","carrying","cell","empty","locatable","number","object","package","truck"]) 
# c_puzzle_predicates = set(["above", "at", "blank", "cell", "number", "object", "right", "tile"])

# to_remove_predicates = {
#     "clear": set(["clear", "handempty"]),
#     "npuzzle": set(["blank"]),
#     "satellite": set(),
#     "logistics": set(),
#     "logistics_1": set(),
#     "spanner": set(),
#     "delivery": set(["empty"]),
#     "miconic": set(),
#     "gripper": set(),
#     "moose_ferry": set(),
#     "moose_gripper": set(),
#     "gripper_single_goal": set(),
#     "blocks_3_on_pairs": set(),
# }
# configurations = [
#         {"name":"NDLMs_id", "tc": False, "activation": "identity", "strict": True},
#         {"name": "NDLMs_sig","tc": False, "activation": "sigmoid", "strict": True},
#         {"name": "NDLMs+_id","tc": True, "activation": "identity", "strict": True},
#         {"name": "NDLMs+_sig","tc": True, "activation": "sigmoid", "strict": True},
#         {"name":"NDLMr_id", "tc": False, "activation": "identity", "strict": False},
#         {"name": "NDLMr_sig","tc": False, "activation": "sigmoid", "strict": False},
#         {"name": "NDLMr+_id","tc": True, "activation": "identity", "strict": False},
#         {"name": "NDLMr+_sig","tc": True, "activation": "sigmoid", "strict": False}
#         ]


# def _resolve_domain_entry(domain_map, domain_key):
#     """Resolve configured domain metadata, including optional parameter-index key."""
#     domain_data = domain_map[domain_key]
#     if len(domain_data) == 6:
#         domain_name, domain_path, train_problem_paths, test_problem_paths, max_states_train, max_states_test = domain_data
#         parameter_domain = domain_key
#     elif len(domain_data) == 7:
#         domain_name, domain_path, train_problem_paths, test_problem_paths, max_states_train, max_states_test, parameter_domain = domain_data
#     else:
#         raise ValueError(f"Unexpected domain entry format for '{domain_key}': {len(domain_data)} values")

#     return (
#         domain_name,
#         domain_path,
#         train_problem_paths,
#         test_problem_paths,
#         max_states_train,
#         max_states_test,
#         parameter_domain,
#     )

def get_dataset(args):

    domain_information = load_domain_information(args)
    
    train_dataset = random_trace_generator.generate_action_model_from_multiple_problems(
        domain_information["domain_path"],
        domain_information["train_problem_paths"],
        domain_information["argument_indices"],
        max_states=args.num_train_states,
        include_non_applicable=args.precondition,
        target_mode = "state" if args.target_mode == "state" else "applicability_vector",
        full_applicability=args.full_applicability,
        max_actions=args.max_actions,
        ignore_type_filtering=args.ignore_type_filtering,
        predicate_indices=domain_information["predicates"],
        max_sampling_seconds_per_problem=args.max_sampling_seconds_per_problem,
    )
    log("Train dataset generation done")
    
    test_dataset = random_trace_generator.generate_action_model_from_multiple_problems(
        domain_information["domain_path"],
        domain_information["test_problem_paths"],
        domain_information["argument_indices"],
        max_states=args.num_test_states,
        include_non_applicable=args.precondition,
        target_mode = "state" if args.target_mode == "state" else "applicability_vector",
        full_applicability=args.full_applicability,
        max_actions=args.max_actions,
        ignore_type_filtering=args.ignore_type_filtering,
        predicate_indices=domain_information["predicates"],
        max_sampling_seconds_per_problem=args.max_sampling_seconds_per_problem,
    )
    log("Test dataset generation done")
    
    return train_dataset, test_dataset


def get_OPTIMAL_GP_dataset(args):
    domain_information = load_domain_information(args)
    
    train_dataset, names_train = random_trace_generator.generate_optimal_general_policy_dataset_from_multiple_problems(
        args,
        domain_information["domain_path"],
        domain_information["train_problem_paths"],
        parameter_indices=domain_information["argument_indices"],
        predicate_indices=domain_information["predicates"],
        max_states=args.num_train_states,
        max_sampling_seconds_per_problem=args.max_sampling_seconds_per_problem,
       
    )
    log("Train dataset generation done")
    test_dataset, names_test = random_trace_generator.generate_optimal_general_policy_dataset_from_multiple_problems(
        args,
        domain_information["domain_path"],
        domain_information["test_problem_paths"],
        parameter_indices=domain_information["argument_indices"],
        predicate_indices=domain_information["predicates"],
        max_states=args.num_test_states,
        max_sampling_seconds_per_problem=args.max_sampling_seconds_per_problem,
    )
    log("Test dataset generation done")
    log(f"Train dataset names: {names_train}")
    log(f"Test dataset names: {names_test}")
        
    return train_dataset, test_dataset

def get_GP_dataset(args):
    domain_information = load_domain_information(args)
    export_domain_information(domain_information, args)
    train_dataset = random_trace_generator.create_general_policy_for_multiple_problems(
        domain_information["domain_path"],
        domain_information["train_problem_paths"],
        parameter_indices=domain_information["argument_indices"],
        domain_name=domain_information["domain"],
        max_states=args.num_train_states,
        max_sampling_seconds_per_problem=args.max_sampling_seconds_per_problem,
    )
    log("Train dataset generation done")
    
    test_dataset = random_trace_generator.create_general_policy_for_multiple_problems(
        domain_information["domain_path"],
        domain_information["test_problem_paths"],
        parameter_indices=domain_information["argument_indices"],
        domain_name=domain_information["domain"],
        max_states=args.num_test_states,
        max_sampling_seconds_per_problem=args.max_sampling_seconds_per_problem,
    )

    log("Test dataset generation done")
    
    return train_dataset, test_dataset
def _log_state_dict_shapes(action_name, state_dict, model=None, n=10):
    log(f"State dict preview for action '{action_name}' (first {n} entries):")
    for i, (key, value) in enumerate(state_dict.items()):
        if i >= n:
            break

        loaded_shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        message = f"[{i}] loaded: {key} -> {loaded_shape}"

        if model is not None:
            model_state = model.state_dict()
            if key in model_state:
                model_shape = tuple(model_state[key].shape) if hasattr(model_state[key], "shape") else type(model_state[key]).__name__
                message += f" | model: {model_shape}"

        log(message)
def load_model_from_checkpoint(checkpoint_file, checkpoint_args, config, action_name):
    in_concepts = checkpoint_args.io_dimensions[action_name]["in_concepts"]
    in_roles = checkpoint_args.io_dimensions[action_name]["in_roles"]
    out_concepts = checkpoint_args.io_dimensions[action_name]["out_concepts"]
    out_roles = checkpoint_args.io_dimensions[action_name]["out_roles"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = getattr(checkpoint_args, "model", "NDLM")
    if model_name == "NDLM":
        model = modules.MultiLayerNDLM(in_concepts, in_roles, out_concepts, out_roles, config).to(device)
    elif model_name == "NLM":
        model = layer.NLM_to_NDLM_Adapter(in_concepts, in_roles, out_concepts, out_roles, checkpoint_args).to(device)
    elif model_name == "MLP":
        model = baselines.PairwiseMLP(
            in_concepts,
            in_roles,
            out_concepts,
            out_roles,
            hidden_size=getattr(checkpoint_args, "baseline_hidden_size", 64),
        ).to(device)
    elif model_name == "GNN":
        model = baselines.MessagePassingPairClassifier(
            in_concepts,
            in_roles,
            out_concepts,
            out_roles,
            hidden_size=getattr(checkpoint_args, "baseline_hidden_size", 64),
            num_layers=getattr(checkpoint_args, "baseline_layers", 3),
        ).to(device)
    elif model_name == "2GNN":
        model = baselines.TwoGNN(
            in_concepts,
            in_roles,
            out_concepts,
            out_roles,
            hidden_size=getattr(checkpoint_args, "baseline_hidden_size", 64),
            num_layers=getattr(checkpoint_args, "baseline_layers", 3),
        ).to(device)
    elif model_name == "3GNN":
        model = baselines.ThreeGNN(
            in_concepts,
            in_roles,
            out_concepts,
            out_roles,
            hidden_size=getattr(checkpoint_args, "baseline_hidden_size", 64),
            num_layers=getattr(checkpoint_args, "baseline_layers", 3),
            max_objects=getattr(checkpoint_args, "three_gnn_max_objects", 32),
        ).to(device)
    elif model_name == "EdgeTransformer":
        model = baselines.EdgeTransformer(
            in_concepts,
            in_roles,
            out_concepts,
            out_roles,
            hidden_size=getattr(checkpoint_args, "baseline_hidden_size", 64),
            num_layers=getattr(checkpoint_args, "baseline_layers", 3),
            heads=getattr(checkpoint_args, "baseline_heads", 4),
        ).to(device)
    elif model_name == "PPGN":
        model = baselines.PPGN(
            in_concepts,
            in_roles,
            out_concepts,
            out_roles,
            hidden_size=getattr(checkpoint_args, "baseline_hidden_size", 64),
            num_layers=getattr(checkpoint_args, "baseline_layers", 3),
            mlp_depth=getattr(checkpoint_args, "baseline_ppgn_depth", 2),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type for checkpoint reload: {model_name!r}")

    log(action_name)
    log(checkpoint_args.io_dimensions[action_name])

    state_dict = torch.load(checkpoint_file, map_location=device)
    preview_n = getattr(checkpoint_args, "state_dict_preview_n", 10)
    # _log_state_dict_shapes(action_name, state_dict, model=model, n=preview_n)

    model.load_state_dict(state_dict)
    model.eval()
    return model



def test_policy_from_checkpoint_models(checkpoint_args, exp_args, config):
    checkpoint_args.data_path = exp_args.data_path
    domain_information = load_domain_information(checkpoint_args)
    # _ , domain_path, train_problem_paths, test_problem_paths, max_states_train, max_states_test = opt_GP_data_paths[domain]
    
    models_per_action = {}
    base_path = checkpoint_args.experiment_path
    for action_name in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, action_name)):
            checkpoint_file = os.path.join(base_path, action_name, "checkpoints", "checkpoint_final.pt")
            if not os.path.exists(checkpoint_file):
                if os.path.exists(os.path.join(base_path, action_name, "checkpoints")) and len(os.listdir(os.path.join(base_path, action_name, "checkpoints"))) > 0:
                    files = os.listdir(os.path.join(base_path, action_name, "checkpoints"))
                    files.sort()
                    best_file = files[-1]
                    checkpoint_file = os.path.join(base_path, action_name, "checkpoints", best_file)
                else:
                    log(f"No checkpoint files found for action '{action_name}' in {os.path.join(base_path, action_name, 'checkpoints')}")
                    continue
            print(action_name, checkpoint_file)
            models_per_action[action_name] = load_model_from_checkpoint(checkpoint_file, checkpoint_args, config, action_name)
    # pick_model = models_per_action.get("pick")

    # c,r = pick_model(C_in, R_in)
    # print("C_out:", c)
    # print("R_out:", r)
    # quit()
    res=random_trace_generator.test_model_on_test_problems(
        domain_information["domain_path"],
        domain_information["train_problem_paths"]+domain_information["test_problem_paths"]+domain_information["eval_problem_paths"],
        models_per_action,
        parameter_indices=domain_information["argument_indices"],
        max_steps=exp_args.num_test_states,
        soft_policy=False,
        model_type="state_full",
        one_step_cycle_check=False,
    )
    for f, r in res.items():
        log(f"file: {f}, steps_taken: {r['steps_taken']}, success: {r['goal_reached']}, optimal_steps: {r['optimal_steps']}")
    log("Final Results:")
    log(f"Total Successes: {sum(r['goal_reached'] for r in res.values())} / {len(res)} ({sum(r['goal_reached'] for r in res.values()) / len(res) * 100:.2f}%)")
    log(f"optimal solutions found: {sum(r['goal_reached'] and r['steps_taken'] == r['optimal_steps'] for r in res.values())} / {len(res)} ({sum(r['goal_reached'] and r['steps_taken'] == r['optimal_steps'] for r in res.values()) / len(res) * 100:.2f}%)")



# if __name__ == "__main__":

#     putdown = Path("/work/rleap1/jan.dornhege/NDLM/outputs/NDLM_OPTGP/c4_1/clear-10_train/optGP/clear/putdown/checkpoints/checkpoint_final.pt")
#     pickup = Path("/work/rleap1/jan.dornhege/NDLM/outputs/NDLM_OPTGP/c4_1/clear-10_train/optGP/clear/pickup/checkpoints/checkpoint_final.pt")
#     stack = Path("/work/rleap1/jan.dornhege/NDLM/outputs/NDLM_OPTGP/c4_1/clear-10_train/optGP/clear/stack/checkpoints/checkpoint_final.pt")
#     unstack = Path("/work/rleap1/jan.dornhege/NDLM/outputs/NDLM_OPTGP/c4_1/clear-10_train/optGP/clear/unstack/checkpoints/checkpoint_final.pt")

#     paths = {
#         "putdown": putdown,
#         "pickup": pickup,
#         "stack": stack,
#         "unstack": unstack,
#     }


#     print("starting")

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--domain", type=int, choices=range(len(data_paths)), default=0,
#                         help="Domain index: 0=clear, 1=npuzzle, 2=satellite, 3=logistics, 4=spanner, 5=delivery")
#     parser.add_argument("--config", type=int, choices=range(len(configurations)), default=0,
#                         help="Config index: 0-")
#     parser.add_argument("--precondition", type=parse_bool, default=False,
#                         help="Enable precondition learning (true/false)")
#     parser.add_argument("--full_applicability", type=parse_bool, default=False,
#                         help="Enable full applicability mode (true/false)")
#     args = parser.parse_args()

#     domain_names_list = list(data_paths.keys())
#     selected_domain = domain_names_list[args.domain]
#     selected_config = args.config

#     # test_problem_path_list = [
#     #     "/work/rleap1/jan.dornhege/MA/data/delivery/test/"+name for name in os.listdir("/work/rleap1/jan.dornhege/MA/data/delivery/test/")] 
#     # train_problem_path = "/work/rleap1/jan.dornhege/MA/data/clear/test/test_clear_probBLOCKS-03-0.pddl"
 

#     max_actions = 10
#     # parameter_indices = None
#     _, _, _, _, _, _, parameter_domain = _resolve_domain_entry(data_paths, selected_domain)
#     parameter_indices = random_trace_generator.load_parameter_indices(parameter_domain)
#     predicate_indices = to_remove_predicates[selected_domain]
#     print("Generating transitions...")
#     ignore_type_filtering = True
#     full_applicability = args.full_applicability
#     precondition_learning = args.precondition
    
#     if precondition_learning:
#         mode = "applicability_vector"
#     else:
#         mode = "state"
#     network_mode = "precondition" if precondition_learning else "effect"

#     domain_name, domain_path, train_problem_paths, test_problem_paths, max_states_train, max_states_test, _ = _resolve_domain_entry(data_paths, selected_domain)

#     # train_dataset = random_trace_generator.generate_action_model_from_multiple_problems(domain_path, 
#     #         train_problem_paths, 
#     #         parameter_indices, 
#     #         max_states=max_states_train, 
#     #         include_non_applicable=precondition_learning, 
#     #         full_applicability=full_applicability,
#     #         target_mode= mode, max_actions = max_actions, 
#     #         ignore_type_filtering=ignore_type_filtering, 
#     #         predicate_indices=predicate_indices)

#     # test_dataset = random_trace_generator.generate_action_model_from_multiple_problems(domain_path, 
#     #         test_problem_paths, 
#     #         parameter_indices, 
#     #         max_states=max_states_test, 
#     #         include_non_applicable=precondition_learning, 
#     #         full_applicability=full_applicability,
#     #         target_mode= mode, 
#     #         max_actions = max_actions, 
#     #         ignore_type_filtering=ignore_type_filtering, 
#     #         predicate_indices=predicate_indices)

#     train_dataset = random_trace_generator.generate_optimal_general_policy_dataset_from_multiple_problems(
#         domain_path,
#         train_problem_paths,
#         parameter_indices=parameter_indices,
#         max_states=max_states_train,
#     )
#     print("Train dataset generation done")

#     test_dataset = random_trace_generator.generate_optimal_general_policy_dataset_from_multiple_problems(
#         domain_path,
#         test_problem_paths,
#         parameter_indices=parameter_indices,
#         max_states=max_states_test,
#     )
#     print("Test dataset generation done")
       
#     # train_dataset = random_trace_generator.create_general_policy_for_multiple_problems(domain_path, train_problem_paths, parameter_indices, domain_name=domain_name, max_states=max_states_train) 
#     # test_dataset = random_trace_generator.create_general_policy_for_multiple_problems(domain_path, test_problem_paths, parameter_indices, domain_name=domain_name, max_states=max_states_test) 
#     # train_dataset={"general_policy": []}
#     # for data in train_dataset_gp:
#     #     train_dataset["general_policy"].append(data)

#     # print("EXP_NAME:", config.EXP_NAME)
#     # for action_name in list(train_dataset.keys()):
#     #     print(f"Train: Action: {action_name}, Sets: {len(train_dataset[action_name])}")
#     #     print(f"Test: Action: {action_name}, Sets: {len(test_dataset[action_name])}")
        
#     #     for data_point in train_dataset[action_name]:
#     #         print(f"Train:Data point: {len(data_point)}")
#     #     for data_point in test_dataset[action_name]:
#     #         print(f"Test:Data point: {len(data_point)}")
#     # quit()
    
#     # test_dataset={"general_policy": []} 
#     # for data in test_dataset_gp:
#     #     test_dataset["general_policy"].append(data)

    
#     # for ds in train_dataset.keys(): 
#     #     if len(train_dataset[ds])>500:
#     #         train_dataset[ds] = train_dataset[ds][:500]
#     # for ds in test_dataset.keys():
#     #     if len(test_dataset[ds])>500:
#     #         test_dataset[ds] = test_dataset[ds][:500]
#     config_settings = configurations[selected_config]

#     print("Generation complete!")
#     config = Config.config_object()
#     config.NUM_EPOCHS = 200
#     config.NUM_LAYERS = 4
#     config.NUM_HIDDEN_CONCEPTS = 3
#     config.NUM_HIDDEN_ROLES = 3
#     config.TRANSITIVE_CLOSURE = config_settings["tc"]
#     config.SKIP_CONNECTIONS = False
#     config.RAGG_NORM = False
#     config.RRA_NORM = False
#     config.CRA_NORM = False

#     config.ACTIVATION_FUNCTION = (
#             torch.nn.Sigmoid() if config_settings["activation"] == "sigmoid" else torch.nn.Identity()
#     )
#     config.LAYER_TYPE = "single_step_minmax" if config_settings["strict"] else "single_step"
#     config.EXP_NAME = f"{config_settings['name']}_{network_mode}_{config_settings['activation']}_tc_{config_settings['tc']}_{time.time()}"

#     config.EXP_PATH = f"ffn_experiments/NEW_GP/0/{selected_domain}/{config.EXP_NAME}/"
#     config.TEST_INTERVAL = 10
#     config.LEARNING_RATE = 0.001
#     config.write_config()
    
    
#     print("EXP_NAME:", config.EXP_NAME)
#     for action_name in list(train_dataset.keys()):
#         with open(config.EXP_PATH+"exp_first.txt", "a") as log_file:
#             if action_name not in test_dataset:
#                 log_file.write(f"Action {action_name} missing in train or test dataset, skipping...\n")
#                 continue
#             log_file.write(f"Train: Action: {action_name}, Sets: {len(train_dataset[action_name])}\n")
#             log_file.write(f"Test: Action: {action_name}, Sets: {len(test_dataset[action_name])}\n")
#             log_file.write(f"Train: First data point length: {len(train_dataset[action_name][0])}\n")
#             log_file.write(f"Test: First data point length: {len(test_dataset[action_name][0])}\n")
#             if len(train_dataset[action_name][0])==0 or len(test_dataset[action_name][0])==0:
#                 log_file.write("No data for this action, skipping...\n")
#                 continue
#             for data_point in train_dataset[action_name]:
#                 log_file.write(f"Train:Data point: {len(data_point)}\n")
#             for data_point in test_dataset[action_name]:
#                 log_file.write(f"Test:Data point: {len(data_point)}\n")
                   
#     for action_name in list(train_dataset.keys()):
#         if action_name not in test_dataset:
#             print(f"Action {action_name} missing in train or test dataset, skipping...")
#             continue
#         if len(train_dataset[action_name][0])==0 or len(test_dataset[action_name][0])==0:
#             print(f"No data for action {action_name}, skipping...")
#             continue
#         acc_history, test_miss, test_acc, time_stats = ffn_main.main(train_dataset[action_name], test_dataset[action_name], config)
        

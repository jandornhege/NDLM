import random_trace_generator.main as random_trace_generator
import os
import json
import sys
import pymimir.advanced.formalism as formalism
import pymimir.advanced.search as search
import torch
import json
# import ffn_main_general_data as ffn_main
import ndlm.configs as Config
import time
from pathlib import Path
import argparse


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false")


def load_moose_dataset_paths(domain_name, max_states_train, max_states_test, parameter_domain=None):
    base_dir = Path("/work/rleap1/jan.dornhege/moose-dataset") / domain_name
    domain_path = base_dir / "domain.pddl"
    train_dir = base_dir / "training"
    test_dir = base_dir / "testing"

    train_problem_paths = sorted(str(p) for p in train_dir.glob("*.pddl"))
    test_problem_paths = sorted(str(p) for p in test_dir.glob("*.pddl"))[:5]

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

clear_domain_path = "/work/rleap1/jan.dornhege/MA/data/clear/domain.pddl" 
clear_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/clear/test/_training_clear_5.pddl"
clear_train_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/clear/test/test_clear_probBLOCKS-06-1.pddl"
clear_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/clear/test/test_clear_probBLOCKS-10-0.pddl"
clear_test_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/clear/test/test_clear_probBLOCKS-10-1.pddl"

npuzzle_domain_path = "/work/rleap1/jan.dornhege/graph_separator/pddl_files/npuzzle/cell-npuzzle.pddl"
npuzzle_train_problem_path = "/work/rleap1/jan.dornhege/graph_separator/pddl_files/npuzzle/cell-npuzzle-3-2.pddl"
npuzzle_test_problem_path = "/work/rleap1/jan.dornhege/graph_separator/pddl_files/npuzzle/cell-npuzzle-3-3.pddl"
npuzzle_test_problem_path_2 = "/work/rleap1/jan.dornhege/graph_separator/pddl_files/npuzzle/cell-npuzzle-4-2.pddl"

satellite_domain_path = "/work/rleap1/jan.dornhege/MA/data/satellite/satellite.pddl"
satellite_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/satellite/satellite_9.pddl"
satellite_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/satellite/satellite_1.pddl"

logistics_domain_path = "data/logistics/logistics.pddl"
logistics_train_problem_path = "data/logistics/logistics_2_2_2_2_2_2.pddl"
logistics_train_problem_path_2 = "data/logistics/logistics_1_2_2_2_2_2.pddl"
logistics_test_problem_path = "data/logistics/logistics_3_3_3_3_3_3.pddl"
logistics_test_problem_path_2 = "data/logistics/logistics-1.pddl"


spanner_domain_path = "/work/rleap1/jan.dornhege/MA/data/spanner/domain.pddl"
spanner_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/spanner/train/prob-2-2-10.pddl"
spanner_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/spanner/test/prob-4-2-5.pddl"

delivery_domain_path = "/work/rleap1/jan.dornhege/MA/data/delivery/domain.pddl"
delivery_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/delivery/train/instance_4_2_0.pddl"
delivery_train_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/delivery/train/instance_3_3_0.pddl"
delivery_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/delivery/test/instance_5_3_0.pddl"
delivery_test_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/delivery/test/instance_7_2_0.pddl"


miconic_domain_path = "/work/rleap1/jan.dornhege/MA/data/miconic/domain.pddl"
miconic_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/miconic/train/training2.pddl"
miconic_train_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/miconic/train/s4-0.pddl"
miconic_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s01-0.pddl"
miconic_test_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s04-1.pddl"
miconic_test_problem_path_3 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s04-2.pddl"
miconic_test_problem_path_4 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s07-0.pddl"
miconic_test_problem_path_5 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s07-1.pddl"
miconic_test_problem_path_6 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s10-0.pddl"
miconic_test_problem_path_7 = "/work/rleap1/jan.dornhege/MA/data/miconic/test/s13-0.pddl"


gripper_domain_path = "/work/rleap1/jan.dornhege/moose-dataset/gripper/domain.pddl"
gripper_train_problem_path = "/work/rleap1/jan.dornhege/moose-dataset/gripper/training/prob01.pddl"

gripper_test_problem_path_3 = "/work/rleap1/jan.dornhege/moose-dataset/gripper/testing/prob03.pddl"
gripper_test_problem_path_4 = "/work/rleap1/jan.dornhege/moose-dataset/gripper/testing/prob04.pddl"
gripper_test_problem_path_5 = "/work/rleap1/jan.dornhege/moose-dataset/gripper/testing/prob05.pddl"
gripper_test_problem_path_6 = "/work/rleap1/jan.dornhege/moose-dataset/gripper/testing/prob06.pddl"


gripper_single_goal_domain_path = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/domain.pddl"
gripper_single_goal_train_problem_path = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/training/p01.pddl"
gripper_single_goal_train_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/training/p02.pddl"
gripper_single_goal_train_problem_path_3 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/training/p03.pddl"

gripper_single_goal_test_problem_path = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_01.pddl"
gripper_single_goal_test_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_02.pddl"
gripper_single_goal_test_problem_path_3 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_03.pddl"
gripper_single_goal_test_problem_path_4 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_04.pddl"
gripper_single_goal_test_problem_path_5 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_05.pddl"
gripper_single_goal_test_problem_path_6 = "/work/rleap1/jan.dornhege/MA/data/gripper_single_goal/testing/p0_06.pddl"


data_paths={
    "clear": ("BLOCKS", clear_domain_path, [clear_train_problem_path, clear_train_problem_path_2], [clear_test_problem_path, clear_test_problem_path_2], 100,1000),
    "npuzzle": ("cpuzzle", npuzzle_domain_path, [npuzzle_train_problem_path], [npuzzle_test_problem_path, npuzzle_test_problem_path_2], 30,30),
    # "satellite": ("satellite", satellite_domain_path, [satellite_train_problem_path], [satellite_test_problem_path]),
    "logistics": ("logistics", logistics_domain_path, [logistics_train_problem_path], [logistics_test_problem_path],200, 3000),
    # "spanner": ("spanner", spanner_domain_path, [spanner_train_problem_path], [spanner_test_problem_path]),
    "delivery": ("delivery", delivery_domain_path, [delivery_train_problem_path, delivery_train_problem_path_2], [delivery_test_problem_path, delivery_test_problem_path_2], 30,50),
    "logistics_1": ("logistics", logistics_domain_path, [logistics_train_problem_path, logistics_test_problem_path], [logistics_test_problem_path_2], 50,100),
    "miconic": ("miconic", miconic_domain_path, [miconic_train_problem_path, miconic_train_problem_path_2], [
        miconic_test_problem_path,
        miconic_test_problem_path_2,
        miconic_test_problem_path_4,
        ], 50,3),
    "gripper": ("gripper", gripper_domain_path, [gripper_train_problem_path, gripper_single_goal_train_problem_path_2, gripper_single_goal_train_problem_path_3], [
        gripper_single_goal_test_problem_path,
        # gripper_train_problem_path,
        gripper_single_goal_test_problem_path_2,
        gripper_single_goal_test_problem_path_3,
        gripper_single_goal_test_problem_path_4,
        gripper_single_goal_test_problem_path_5,
        gripper_single_goal_test_problem_path_6
    ], 10,10),
    
}
        # miconic_test_problem_path_3,
        # miconic_test_problem_path_5,
        # miconic_test_problem_path_6,
        # miconic_test_problem_path_7,
opt_GP_data_paths={
    "clear": ("BLOCKS", clear_domain_path, [clear_train_problem_path, clear_train_problem_path_2], [clear_test_problem_path, clear_test_problem_path_2], 20,10),
    "logistics": ("logistics", logistics_domain_path, [logistics_train_problem_path, logistics_test_problem_path], [logistics_test_problem_path_2], 100,100),
    "miconic": ("miconic", miconic_domain_path, [miconic_train_problem_path, miconic_train_problem_path_2], [
        miconic_test_problem_path,
        miconic_test_problem_path_2,
        miconic_test_problem_path_4,
        ], 50,4),
    "gripper_single_goal": ("gripper", gripper_domain_path, [gripper_train_problem_path, gripper_single_goal_train_problem_path_2, gripper_single_goal_train_problem_path_3], [
        gripper_single_goal_test_problem_path,
        # gripper_train_problem_path,
        gripper_single_goal_test_problem_path_2,
        gripper_single_goal_test_problem_path_3,
        gripper_single_goal_test_problem_path_4,
        gripper_single_goal_test_problem_path_5,
        gripper_single_goal_test_problem_path_6
    ], 10,10),
    "gripper": ("gripper", gripper_domain_path, [gripper_train_problem_path], [
        gripper_test_problem_path_3,
        gripper_test_problem_path_4,
        gripper_test_problem_path_5,
        gripper_test_problem_path_6
    ], 30,10),
    "npuzzle": ("cpuzzle", npuzzle_domain_path, [npuzzle_train_problem_path, npuzzle_test_problem_path, npuzzle_test_problem_path_2], [npuzzle_test_problem_path, npuzzle_test_problem_path_2], 40,100),    
    "moose_ferry": load_moose_dataset_paths("ferry", 200, 50, parameter_domain="ferry"),
}


bw_predicates = set(["clear","handempty","holding","number","object","on","ontable"]) 
delivery_predicates = set(["=", "adjacent","at","carrying","cell","empty","locatable","number","object","package","truck"]) 
c_puzzle_predicates = set(["above", "at", "blank", "cell", "number", "object", "right", "tile"])

to_remove_predicates = {
    "clear": set(["clear", "handempty"]),
    "npuzzle": set(["blank"]),
    "satellite": set(),
    "logistics": set(),
    "logistics_1": set(),
    "spanner": set(),
    "delivery": set(["empty"]),
    "miconic": set(),
    "gripper": set(),
    "moose_ferry": set(),
    "moose_gripper": set(),
    "gripper_single_goal": set(),
}
configurations = [
        {"name":"NDLMs_id", "tc": False, "activation": "identity", "strict": True},
        {"name": "NDLMs_sig","tc": False, "activation": "sigmoid", "strict": True},
        {"name": "NDLMs+_id","tc": True, "activation": "identity", "strict": True},
        {"name": "NDLMs+_sig","tc": True, "activation": "sigmoid", "strict": True},
        {"name":"NDLMr_id", "tc": False, "activation": "identity", "strict": False},
        {"name": "NDLMr_sig","tc": False, "activation": "sigmoid", "strict": False},
        {"name": "NDLMr+_id","tc": True, "activation": "identity", "strict": False},
        {"name": "NDLMr+_sig","tc": True, "activation": "sigmoid", "strict": False}
        ]


def _resolve_domain_entry(domain_map, domain_key):
    """Resolve configured domain metadata, including optional parameter-index key."""
    domain_data = domain_map[domain_key]
    if len(domain_data) == 6:
        domain_name, domain_path, train_problem_paths, test_problem_paths, max_states_train, max_states_test = domain_data
        parameter_domain = domain_key
    elif len(domain_data) == 7:
        domain_name, domain_path, train_problem_paths, test_problem_paths, max_states_train, max_states_test, parameter_domain = domain_data
    else:
        raise ValueError(f"Unexpected domain entry format for '{domain_key}': {len(domain_data)} values")

    return (
        domain_name,
        domain_path,
        train_problem_paths,
        test_problem_paths,
        max_states_train,
        max_states_test,
        parameter_domain,
    )

def get_dataset(domain, config, precondition, full_applicability, remove_predicates=True, max_sampling_seconds_per_problem=None):
    (
        domain_name,
        domain_path,
        train_problem_paths,
        test_problem_paths,
        max_states_train,
        max_states_test,
        parameter_domain,
    ) = _resolve_domain_entry(data_paths, domain)

    parameter_indices = random_trace_generator.load_parameter_indices(parameter_domain)
    if remove_predicates:
        predicate_indices = to_remove_predicates[domain]
    else:
        predicate_indices = set()
    ignore_type_filtering = True
    mode = "applicability_vector" if precondition else "state"
    train_dataset = random_trace_generator.generate_action_model_from_multiple_problems(
        domain_path,
        train_problem_paths,
        parameter_indices,
        max_states=max_states_train,
        include_non_applicable=precondition,
        full_applicability=full_applicability,
        target_mode=mode,
        max_actions=10,
        ignore_type_filtering=ignore_type_filtering,
        predicate_indices=predicate_indices,
        max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
    )
    test_dataset = random_trace_generator.generate_action_model_from_multiple_problems(
        domain_path,
        test_problem_paths,
        parameter_indices,
        max_states=max_states_test,
        include_non_applicable=precondition,
        full_applicability=full_applicability,
        target_mode=mode,
        max_actions=10,
        ignore_type_filtering=ignore_type_filtering,
        predicate_indices=predicate_indices,
        max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
    )
    return train_dataset, test_dataset


def get_OPTIMAL_GP_dataset(domain, config, max_sampling_seconds_per_problem=None):
    (
        domain_name,
        domain_path,
        train_problem_paths,
        test_problem_paths,
        max_states_train,
        max_states_test,
        parameter_domain,
    ) = _resolve_domain_entry(opt_GP_data_paths, domain)
    parameter_indices = random_trace_generator.load_parameter_indices(parameter_domain)
    predicate_indices = set()
    
    train_dataset = random_trace_generator.generate_optimal_general_policy_dataset_from_multiple_problems(
        domain_path,
        train_problem_paths,
        parameter_indices = parameter_indices,
        predicate_indices = predicate_indices,
        max_states = max_states_train,
        max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
       
    )
    test_dataset = random_trace_generator.generate_optimal_general_policy_dataset_from_multiple_problems(
        domain_path,
        test_problem_paths,
        parameter_indices = parameter_indices,
        predicate_indices = predicate_indices,
        max_states = max_states_test,
        max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
    )
    return train_dataset, test_dataset

def get_GP_dataset(domain, config, check_optimality=False):
    (
        domain_name,
        domain_path,
        train_problem_paths,
        test_problem_paths,
        max_states_train,
        max_states_test,
        parameter_domain,
    ) = _resolve_domain_entry(opt_GP_data_paths, domain)
    parameter_indices = random_trace_generator.load_parameter_indices(parameter_domain)
    predicate_indices = set()
    
    train_dataset = random_trace_generator.create_general_policy_for_multiple_problems(
        domain_path,
        train_problem_paths,
        parameter_indices = parameter_indices,
        domain_name = domain_name,
        max_states = max_states_train,
        check_optimality = check_optimality,
    )
    test_dataset = random_trace_generator.create_general_policy_for_multiple_problems(
        domain_path,
        test_problem_paths,
        parameter_indices = parameter_indices,
        domain_name = domain_name,
        max_states = max_states_test,
        check_optimality = check_optimality,
    )
    return train_dataset, test_dataset

if __name__ == "__main__":
    print("starting")

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=int, choices=range(len(data_paths)), default=0,
                        help="Domain index: 0=clear, 1=npuzzle, 2=satellite, 3=logistics, 4=spanner, 5=delivery")
    parser.add_argument("--config", type=int, choices=range(len(configurations)), default=0,
                        help="Config index: 0-")
    parser.add_argument("--precondition", type=parse_bool, default=False,
                        help="Enable precondition learning (true/false)")
    parser.add_argument("--full_applicability", type=parse_bool, default=False,
                        help="Enable full applicability mode (true/false)")
    args = parser.parse_args()

    domain_names_list = list(data_paths.keys())
    selected_domain = domain_names_list[args.domain]
    selected_config = args.config

    # test_problem_path_list = [
    #     "/work/rleap1/jan.dornhege/MA/data/delivery/test/"+name for name in os.listdir("/work/rleap1/jan.dornhege/MA/data/delivery/test/")] 
    # train_problem_path = "/work/rleap1/jan.dornhege/MA/data/clear/test/test_clear_probBLOCKS-03-0.pddl"
 

    max_actions = 10
    # parameter_indices = None
    _, _, _, _, _, _, parameter_domain = _resolve_domain_entry(data_paths, selected_domain)
    parameter_indices = random_trace_generator.load_parameter_indices(parameter_domain)
    predicate_indices = to_remove_predicates[selected_domain]
    print("Generating transitions...")
    ignore_type_filtering = True
    full_applicability = args.full_applicability
    precondition_learning = args.precondition
    
    if precondition_learning:
        mode = "applicability_vector"
    else:
        mode = "state"
    network_mode = "precondition" if precondition_learning else "effect"

    domain_name, domain_path, train_problem_paths, test_problem_paths, max_states_train, max_states_test, _ = _resolve_domain_entry(data_paths, selected_domain)

    # train_dataset = random_trace_generator.generate_action_model_from_multiple_problems(domain_path, 
    #         train_problem_paths, 
    #         parameter_indices, 
    #         max_states=max_states_train, 
    #         include_non_applicable=precondition_learning, 
    #         full_applicability=full_applicability,
    #         target_mode= mode, max_actions = max_actions, 
    #         ignore_type_filtering=ignore_type_filtering, 
    #         predicate_indices=predicate_indices)

    # test_dataset = random_trace_generator.generate_action_model_from_multiple_problems(domain_path, 
    #         test_problem_paths, 
    #         parameter_indices, 
    #         max_states=max_states_test, 
    #         include_non_applicable=precondition_learning, 
    #         full_applicability=full_applicability,
    #         target_mode= mode, 
    #         max_actions = max_actions, 
    #         ignore_type_filtering=ignore_type_filtering, 
    #         predicate_indices=predicate_indices)

    train_dataset = random_trace_generator.generate_optimal_general_policy_dataset_from_multiple_problems(
        domain_path,
        train_problem_paths,
        parameter_indices=parameter_indices,
        max_states=max_states_train,
    )
    print("Train dataset generation done")

    test_dataset = random_trace_generator.generate_optimal_general_policy_dataset_from_multiple_problems(
        domain_path,
        test_problem_paths,
        parameter_indices=parameter_indices,
        max_states=max_states_test,
    )
    print("Test dataset generation done")
       
    # train_dataset = random_trace_generator.create_general_policy_for_multiple_problems(domain_path, train_problem_paths, parameter_indices, domain_name=domain_name, max_states=max_states_train) 
    # test_dataset = random_trace_generator.create_general_policy_for_multiple_problems(domain_path, test_problem_paths, parameter_indices, domain_name=domain_name, max_states=max_states_test) 
    # train_dataset={"general_policy": []}
    # for data in train_dataset_gp:
    #     train_dataset["general_policy"].append(data)

    # print("EXP_NAME:", config.EXP_NAME)
    # for action_name in list(train_dataset.keys()):
    #     print(f"Train: Action: {action_name}, Sets: {len(train_dataset[action_name])}")
    #     print(f"Test: Action: {action_name}, Sets: {len(test_dataset[action_name])}")
        
    #     for data_point in train_dataset[action_name]:
    #         print(f"Train:Data point: {len(data_point)}")
    #     for data_point in test_dataset[action_name]:
    #         print(f"Test:Data point: {len(data_point)}")
    # quit()
    
    # test_dataset={"general_policy": []} 
    # for data in test_dataset_gp:
    #     test_dataset["general_policy"].append(data)

    
    # for ds in train_dataset.keys(): 
    #     if len(train_dataset[ds])>500:
    #         train_dataset[ds] = train_dataset[ds][:500]
    # for ds in test_dataset.keys():
    #     if len(test_dataset[ds])>500:
    #         test_dataset[ds] = test_dataset[ds][:500]
    config_settings = configurations[selected_config]

    print("Generation complete!")
    config = Config.config_object()
    config.NUM_EPOCHS = 200
    config.NUM_LAYERS = 4
    config.NUM_HIDDEN_CONCEPTS = 3
    config.NUM_HIDDEN_ROLES = 3
    config.TRANSITIVE_CLOSURE = config_settings["tc"]
    config.SKIP_CONNECTIONS = False
    config.RAGG_NORM = False
    config.RRA_NORM = False
    config.CRA_NORM = False

    config.ACTIVATION_FUNCTION = (
            torch.nn.Sigmoid() if config_settings["activation"] == "sigmoid" else torch.nn.Identity()
    )
    config.LAYER_TYPE = "single_step_minmax" if config_settings["strict"] else "single_step"
    config.EXP_NAME = f"{config_settings['name']}_{network_mode}_{config_settings['activation']}_tc_{config_settings['tc']}_{time.time()}"

    config.EXP_PATH = f"ffn_experiments/NEW_GP/0/{selected_domain}/{config.EXP_NAME}/"
    config.TEST_INTERVAL = 10
    config.LEARNING_RATE = 0.001
    config.write_config()
    
    
    print("EXP_NAME:", config.EXP_NAME)
    for action_name in list(train_dataset.keys()):
        with open(config.EXP_PATH+"exp_first.txt", "a") as log_file:
            if action_name not in test_dataset:
                log_file.write(f"Action {action_name} missing in train or test dataset, skipping...\n")
                continue
            log_file.write(f"Train: Action: {action_name}, Sets: {len(train_dataset[action_name])}\n")
            log_file.write(f"Test: Action: {action_name}, Sets: {len(test_dataset[action_name])}\n")
            log_file.write(f"Train: First data point length: {len(train_dataset[action_name][0])}\n")
            log_file.write(f"Test: First data point length: {len(test_dataset[action_name][0])}\n")
            if len(train_dataset[action_name][0])==0 or len(test_dataset[action_name][0])==0:
                log_file.write("No data for this action, skipping...\n")
                continue
            for data_point in train_dataset[action_name]:
                log_file.write(f"Train:Data point: {len(data_point)}\n")
            for data_point in test_dataset[action_name]:
                log_file.write(f"Test:Data point: {len(data_point)}\n")
                   
    for action_name in list(train_dataset.keys()):
        if action_name not in test_dataset:
            print(f"Action {action_name} missing in train or test dataset, skipping...")
            continue
        if len(train_dataset[action_name][0])==0 or len(test_dataset[action_name][0])==0:
            print(f"No data for action {action_name}, skipping...")
            continue
        acc_history, test_miss, test_acc, time_stats = ffn_main.main(train_dataset[action_name], test_dataset[action_name], config)
        

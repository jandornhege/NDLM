import data_encoding
import ActionModel_data_encoding
import argparse
import ndlm.main as NDLM_main
import ndlm.configs as config
import ndlm.modules as modules
import sys
import os
import torch
import time
import difflogic.nn.neural_logic.layer as layer


def merge_train_test_actions(train, test):
    """Keep only actions available in both train and test after pruning."""
    train_actions = set(train.keys())
    test_actions = set(test.keys())
    common_actions = sorted(train_actions.intersection(test_actions))

    dropped_train_only = sorted(train_actions - test_actions)
    dropped_test_only = sorted(test_actions - train_actions)
    if dropped_train_only:
        print(f"Skipping train-only actions after pruning: {dropped_train_only}")
    if dropped_test_only:
        print(f"Skipping test-only actions after pruning: {dropped_test_only}")

    return {action: (train[action], test[action]) for action in common_actions}

def datapoints_to_dataset(test, train):
    train_dataset = []
    for dataset in train:
        c = torch.stack([item[0] for item in dataset])  # (num_samples, num_concepts, num_objects)
        r = torch.stack([item[1] for item in dataset])      # (num_samples, num_roles, num_objects, num_objects)
        c_targets = torch.stack([item[2] for item in dataset])    # (num_samples, num_out_concepts, num_objects)
        r_targets = torch.stack([item[3] for item in dataset])    # (num_samples, num_out_roles, num_objects, num_objects)
        train_dataset.append((c, r, c_targets, r_targets))
        print("train_shapes:", c.shape, r.shape, c_targets.shape, r_targets.shape)
    
    test_dataset = []
    for dataset in test:
        c = torch.stack([item[0] for item in dataset])  # (num_samples, num_concepts, num_objects)
        r = torch.stack([item[1] for item in dataset])      # (num_samples, num_roles, num_objects, num_objects)
        c_targets = torch.stack([item[2] for item in dataset])    # (num_samples, num_out_concepts, num_objects)
        r_targets = torch.stack([item[3] for item in dataset])    # (num_samples, num_out_roles, num_objects, num_objects)
        test_dataset.append((c, r, c_targets, r_targets))
        print("test_shapes:", c.shape, r.shape, c_targets.shape, r_targets.shape)
        
    return train_dataset, test_dataset

def get_1D_ARC_dataset(task_name, all, config):
    task = f"/work/rleap1/jan.dornhege/NDLM/src/data/1D-ARC-main/dataset/{task_name}/"
    if all:
        train, test = data_encoding.create_dataset_from_dir(task+task_name, config)
    else:
        train, test = data_encoding.create_dataset_from_file(task, task_name+"_0.json", config)

    train_dataset, test_dataset = datapoints_to_dataset([test], [train])
    return train_dataset, test_dataset

def get_Action_Model_learning_datasets(domain, config, precondition, full_applicability=False, remove_predicates=False, max_sampling_seconds_per_problem=None):
    train, test = ActionModel_data_encoding.get_dataset(
        domain,
        config,
        precondition,
        full_applicability,
        remove_predicates=remove_predicates,
        max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
    )
    data_dir = merge_train_test_actions(train, test)
    for action, (train_data, test_data) in data_dir.items():
        train_dataset, test_dataset = datapoints_to_dataset(test_data, train_data)
        data_dir[action] = (train_dataset, test_dataset)
    return data_dir

def get_OPTIMAL_GP_dataset(domain, config, max_sampling_seconds_per_problem=None):
    train, test = ActionModel_data_encoding.get_OPTIMAL_GP_dataset(
        domain,
        config,
        max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
    )
    data_dir = merge_train_test_actions(train, test)
    for action, (train_data, test_data) in data_dir.items():
        train_dataset, test_dataset = datapoints_to_dataset(test_data, train_data)
        data_dir[action] = (train_dataset, test_dataset)
    return data_dir

def get_GP_dataset(domain, config, check_optimality=False):
    train, test = ActionModel_data_encoding.get_GP_dataset(domain, config, check_optimality=check_optimality)
    data_dir = merge_train_test_actions(train, test)
    for action, (train_data, test_data) in data_dir.items():
        train_dataset, test_dataset = datapoints_to_dataset(test_data, train_data)
        data_dir[action] = (train_dataset, test_dataset)
    return data_dir

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Runs NDLM tasks using model(NLM or NDLM) on 1D-ARC or Action Model learning datasets.")
    parser.add_argument("--model", default="NDLM", choices=["NDLM", "NLM"], help="Model to use currently: NDLM or NLM.")
    parser.add_argument("--task", type=str, required=True, choices=["1DARC", "ActionModel", "optGP", "GP"], help="'1DARC', 'ActionModel', 'optGP', or 'GP'")
    parser.add_argument("--name", type=str, required=True, help="Name of ARC task or Action Model domain.")
    parser.add_argument("--all", action="store_true", help="Load all datasets in the directory for 1D-ARC tasks.")
    parser.add_argument("--precondition", action="store_true", help="Load datasets for learning preconditions in Action Model learning tasks.")
    parser.add_argument("--full-applicability", action="store_true", help="Load datasets with full applicability in Action Model learning tasks.")
    parser.add_argument("--remove-predicates", action="store_true", help="Remove certain predicates from the Action Model learning datasets.")
    parser.add_argument("--check-optimality", action="store_true", help="Label GP transitions by optimality (A* distance) instead of loaded policy.")
    parser.add_argument(
        "--max-sampling-seconds-per-problem",
        type=float,
        default=None,
        help="Optional per-problem time limit (seconds) for BFS-based sampling.",
    )
    parser.add_argument("--dump-dir", type=str, help="Directory to dump results.")
    parser.add_argument("--config-file", type=str, help="Path to a JSON config file to load configuration parameters from.")
    args = parser.parse_args()

    c = config.config_object()
    if args.config_file:
        c = config.config_from_json_file(args.config_file)
    print("Config:", c.to_dict())
    print("Starting task:", args.task, "Name:", args.name, "Model:", args.model)
    if args.dump_dir:
        os.makedirs(f"{args.dump_dir}/{args.task}/{args.name}", exist_ok=True)
        sys.stdout = open(f"{args.dump_dir}/{args.task}/{args.name}/output_{time.time()}.log", "w",) 
        c.save_to_file(f"{args.dump_dir}/{args.task}/{args.name}/config_{time.time()}.json")
    
    
    if args.task == "1DARC":
        task_name = args.name
        train, test = get_1D_ARC_dataset(task_name, args.all, c)
        print(len(train), len(test))
        print(len(train[0]), len(test[0]))
        print(train[0][0].shape, train[0][1].shape, train[0][2].shape, train[0][3].shape)
    
        in_concepts = train[0][0].shape[1]
        in_roles = train[0][1].shape[1]
        out_concepts = train[0][2].shape[1]
        out_roles = train[0][3].shape[1]
        print("Data loaded")
        if args.model == "NDLM":
            model = modules.MultiLayerNDLM(in_concepts, in_roles, out_concepts, out_roles, c)
        else:
            model = layer.NLM_to_NDLM_Adapter(in_concepts, in_roles, out_concepts, out_roles, c)
        NDLM_main.main(train, test, c, model)
    
    elif args.task in ["ActionModel", "optGP", "GP"]:
        domain = args.name
        if args.task == "ActionModel":
            datasets = get_Action_Model_learning_datasets(
                domain,
                c,
                args.precondition,
                args.full_applicability,
                remove_predicates=args.remove_predicates,
                max_sampling_seconds_per_problem=args.max_sampling_seconds_per_problem,
            )
        elif args.task == "optGP":
            datasets = get_OPTIMAL_GP_dataset(
                domain,
                c,
                max_sampling_seconds_per_problem=args.max_sampling_seconds_per_problem,
            )
        else:
            datasets = get_GP_dataset(domain, c, check_optimality=args.check_optimality)
        print("Data loaded")
    
        for action, (train, test) in datasets.items():
            in_concepts = train[0][0].shape[1]
            in_roles = train[0][1].shape[1]
            out_concepts = train[0][2].shape[1]
            out_roles = train[0][3].shape[1]
            if args.model == "NDLM":
                model = modules.MultiLayerNDLM(in_concepts, in_roles, out_concepts, out_roles, c)
            else:
                model = layer.NLM_to_NDLM_Adapter(in_concepts, in_roles, out_concepts, out_roles, c)

            NDLM_main.main(train, test, c, model)

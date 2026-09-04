import os
from copy import deepcopy
import torch
import ndlm.configs as Config
import ndlm.modules as modules
from my_logging import log

import matplotlib.pyplot as plt
import numpy as np
from ndlm.configs import config_from_json_file
from pathlib import Path

def prune_weights_batchwise(
    model,
    evaluator,
    args,
    action_name,
    batch_size=1000,
    inplace=True,
    verbose=True
):
    original_state = deepcopy(model.state_dict())

    results = []

    # Collect all prunable tensors
    params = [
        (name, param)
        for name, param in model.state_dict().items()
        if torch.is_floating_point(param)
    ]
    seen = set()

    total_weights = sum(p.numel() for _, p in params)

    LOOKAHEAD = 50

    remaining = total_weights
    step = 0
    while True:
        step += 1

        # ------------------------------------------------------------------
        # Find the LOOKAHEAD smallest remaining weights
        # ------------------------------------------------------------------
        candidates = []

        for name, tensor in params:
            flat = tensor.view(-1)

            for idx in range(flat.numel()):
                if flat[idx] == 0:
                    continue
                if (name, idx) in seen:
                    continue

                candidates.append(
                    (flat[idx].abs().item(), name, tensor, idx)
                )

        if not candidates:
            if verbose:
                log("No non-zero weights remaining.")
                remaining = 0
                for name, tensor in params:
                    flat = tensor.view(-1)
                    remaining += (flat != 0).sum().item()
                log(f"Remaining non-zero weights: {remaining} of {total_weights}")


            break

        candidates.sort(key=lambda x: x[0])
        batch = candidates[:LOOKAHEAD]

        # ------------------------------------------------------------------
        # Prune entire batch
        # ------------------------------------------------------------------
        originals = []

        with torch.no_grad():
            for _, _, tensor, idx in batch:
                flat = tensor.view(-1)
                originals.append(flat[idx].item())
                flat[idx] = 0

        score, res = evaluator(model)
        results.append((score, res))

        if verbose:
            log(
                f"Step {step + 1}: tried pruning batch of "
                f"{len(batch)} weights, score={score}"
            )

        # ------------------------------------------------------------------
        # Batch succeeded
        # ------------------------------------------------------------------
        if score == 0:
            plot_weight_distribution(
                args,
                model,
                action_name + f"_step_{step + 1}"
            )
            continue

        # ------------------------------------------------------------------
        # Batch failed -> restore everything
        # ------------------------------------------------------------------
        with torch.no_grad():
            for (_, _, tensor, idx), value in zip(batch, originals):
                tensor.view(-1)[idx] = value

        if verbose:
            log("Batch failed, checking weights individually...")

        # ------------------------------------------------------------------
        # Try every weight individually
        # ------------------------------------------------------------------
        for (value, name, tensor, idx), old_value in zip(batch, originals):

            flat = tensor.view(-1)

            with torch.no_grad():
                flat[idx] = 0

            score, res = evaluator(model)
            results.append((score, res))

            if score > 0:
                with torch.no_grad():
                    flat[idx] = old_value

                seen.add((name, idx))

                if verbose:
                    log(
                        f"Keeping {name}[{idx}] "
                        f"({value:.2e})"
                    )
            else:
                if verbose:
                    log(
                        f"Pruned {name}[{idx}] "
                        f"({value:.2e})"
                    )
        plot_weight_distribution(
            args,
            model,
            action_name + f"_step_{step + 1}"
        )
    return model
        

def prune_smallest_weights_iteratively(
    model,
    evaluator,
    args,
    action_name,
    num_steps=None,
    inplace=True,
    verbose=True
):
    """
    Iteratively sets the smallest non-zero weights of a PyTorch model to zero
    and evaluates the model after each pruning step.

    Args:
        model (torch.nn.Module):
            The PyTorch model to prune.

        evaluator (callable):
            Function that takes the model as input and returns an evaluation
            metric/result.

        num_steps (int or None):
            Number of weights to prune. If None, prune until all weights are zero.

        inplace (bool):
            If False, restore original model weights after evaluation.
            If True, keep the pruned model.

        verbose (bool):
            Print pruning progress.

    Returns:
        list:
            List of evaluation results after each pruning step.
    """

    original_state = deepcopy(model.state_dict())

    results = []

    # Collect all prunable tensors
    params = [
        (name, param)
        for name, param in model.state_dict().items()
        if torch.is_floating_point(param)
    ]
    seen = set()

    total_weights = sum(p.numel() for _, p in params)

    if num_steps is None:
        num_steps = total_weights

    for step in range(num_steps):

        best = None

        for name, tensor in params:
            flat = tensor.view(-1)

            for idx in range(flat.numel()):
                if flat[idx] == 0:
                    continue
                if (name, idx) in seen:
                    continue

                absval = flat[idx].abs().item()

                if best is None or absval < best[0]:
                    best = (absval, name, tensor, idx)
        # No non-zero weights left
        if best is None:
            if verbose:
                log("No non-zero weights remaining.")
            break
        
        smallest_value, name, tensor, flat_idx = best
        flat = tensor.view(-1)
        old_value = flat[flat_idx].item()
        

        # Zero the selected weight
        with torch.no_grad():
            flat[flat_idx] = 0

        # Evaluate
        score, res = evaluator(model)
        results.append((score, res))
        if verbose:
            log(
                f"Step {step+1}/{num_steps}: "
                f"zeroed weight {smallest_value:.6e} "
                f"in {name}, score={score}"
            )
        if score > 0:
            # revert the last pruning step and exclude this weight from future pruning
            with torch.no_grad():
                flat[flat_idx] = old_value
                seen.add((name, flat_idx))

            if verbose:
                log(f"Reverted pruning step {step+1} due to positive score.")
        # plot every 10 steps
        if (step + 1) % 10 == 0:
            plot_weight_distribution(args, model, action_name+f"_step_{step+1}")

    # Restore if not inplace
    if not inplace:
        model.load_state_dict(original_state)

    return results


def load_model_from_checkpoint(checkpoint_path, checkpoint_args, config, action_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_args.model == "NDLM":
        log(f"Action: {action_name}, dimensions: {checkpoint_args.io_dimensions[action_name]}")
        model = modules.MultiLayerNDLM(
            checkpoint_args.io_dimensions[action_name]["in_concepts"],
            checkpoint_args.io_dimensions[action_name]["in_roles"],
            checkpoint_args.io_dimensions[action_name]["out_concepts"],
            checkpoint_args.io_dimensions[action_name]["out_roles"],
            config
        ).to(device)
    elif checkpoint_args.model == "NLM":
        model = layer.NLM_to_NDLM_Adapter(
            checkpoint_args.io_dimensions[action_name]["in_concepts"],
            checkpoint_args.io_dimensions[action_name]["in_roles"],
            checkpoint_args.io_dimensions[action_name]["out_concepts"],
            checkpoint_args.io_dimensions[action_name]["out_roles"],
            config
        ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def load_model_from_file(in_concepts, in_roles, out_concepts, out_roles, config, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = modules.MultiLayerNDLM(
        in_concepts,
        in_roles,
        out_concepts,
        out_roles,
        config
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def establish_predictions(model, train, test):
    out=[]
    for dataset in [train, test]:
        predictions = [[torch.zeros_like(c_target), torch.zeros_like(r_target)] for _, _, c_target, r_target in dataset]
        for i, instance in enumerate(dataset):
            concepts, roles, target_concepts, target_roles = instance
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            concepts = concepts.to(device)
            roles = roles.to(device)
            target_concepts = target_concepts.to(device)
            target_roles = target_roles.to(device)
            with torch.no_grad():
                for j in range(concepts.size(0)):
                    pred_concepts, pred_roles = model(concepts[j].unsqueeze(0), roles[j].unsqueeze(0))
                    predictions[i][0][j] = pred_concepts.squeeze(0)
                    predictions[i][1][j] = pred_roles.squeeze(0)
        for i in range(len(predictions)):
            predictions[i][0] = (predictions[i][0] > 0).float()
            predictions[i][1] = (predictions[i][1] > 0).float()
        # for i in range(len(predictions)):
            # log(f"Instance {i}: Predicted Concepts: {predictions[i][0].numel()}, Predicted Roles: {predictions[i][1].numel()}")
        out.append(predictions)
    return out

def evaluator(model, train, test, train_target, test_target):
    num_changes_c = 0
    num_changes_r = 0
    res= {}
    predictions = establish_predictions(model, train, test)
    for k, instance in enumerate(train):
        concepts, roles, target_concepts, target_roles = instance        
        pred_concepts = predictions[0][k][0]
        pred_roles = predictions[0][k][1]
        num_changes_c += ((pred_concepts > 0) != (train_target[0][k] > 0)).sum().item()
        num_changes_r += ((pred_roles > 0) != (train_target[1][k] > 0)).sum().item()
    res["train_c_changes"] = num_changes_c
    res["train_r_changes"] = num_changes_r
    
    num_changes_c = 0
    num_changes_r = 0
    
    for k, instance in enumerate(test):   
        pred_concepts = predictions[1][k][0]
        pred_roles = predictions[1][k][1]
        num_changes_c += ((pred_concepts > 0) != (test_target[0][k] > 0)).sum().item()
        num_changes_r += ((pred_roles > 0) != (test_target[1][k] > 0)).sum().item()
    res["test_c_changes"] = num_changes_c
    res["test_r_changes"] = num_changes_r

    cummulatice_changes = res["train_c_changes"] + res["train_r_changes"] + res["test_c_changes"] + res["test_r_changes"]
    return cummulatice_changes, res

def plot_weight_distribution(args, model, action_name):

    weights = []
    for name, param in model.named_parameters():
        if torch.is_floating_point(param):
            weights.append(param.detach().cpu().numpy().flatten())
    weights = np.concatenate(weights)
    weights = [abs(w) for w in weights]  # Filter out zero weights
    weights.sort()
    plt.figure(figsize=(10, 6))
    plt.plot(weights, color='blue')
    plt.title(f'Weight Distribution for Action: {action_name}')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.grid(True)
    
    output_dir = os.path.join(args.experiment_path,"weight_distributions")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'weight_distribution_{action_name}.png'))
    plt.close()

def save_model(model, args, action_name):
    output_dir = os.path.join(args.experiment_path,"pruned_models")
    os.makedirs(output_dir, exist_ok=True)
    config= model.config
    config.save_to_file(os.path.join(args.experiment_path,"pruned_models", f'pruned_model_config_{action_name}.json'))
    save_path = os.path.join(output_dir, f'model_{action_name}.pt')
    torch.save(model.state_dict(), save_path)
    log(f"Saved pruned model for action '{action_name}' to {save_path}")

def prune_model(checkpoint_args, exp_args, config, datasets):
    #load checkpoint models for each action
    models_per_action = {}
    base_path = exp_args.checkpoint_path
    for action_name in datasets.keys():
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
        else:
            log(f"No checkpoint files found for action '{action_name}' in {os.path.join(base_path, action_name, 'checkpoints')}")
            continue
    for action_name, model in models_per_action.items():    
        plot_weight_distribution(exp_args, model, action_name)
        save_model(model, exp_args, action_name+"_original")
    prune_results = {}
    action_names = list(models_per_action.keys())
    action_names.sort(reverse=True) 
    for action_name in action_names:
        if action_name != "pickup":
            continue
    
        model = models_per_action[action_name]
        log(f"Loaded model for action '{action_name}' from checkpoint.")
        
        predictions = establish_predictions(model, datasets[action_name][0], datasets[action_name][1])
        # Evaluate if the model correctly identifies the targets:
        for i in range(len(predictions[0])):
            log(f"Instance {i}: Predicted Concepts: {predictions[0][i][0].numel()}, Predicted Roles: {predictions[0][i][1].numel()}")
        
        targets_train_c_pred = [predictions[0][i][0] for i in range(len(predictions[0]))]
        targets_train_r_pred = [predictions[0][i][1] for i in range(len(predictions[0]))]
        targets_test_c_pred = [predictions[1][i][0] for i in range(len(predictions[1]))]
        targets_test_r_pred = [predictions[1][i][1] for i in range(len(predictions[1]))]
        targets_train_c = [datasets[action_name][0][i][2] for i in range(len(datasets[action_name][0]))]        
        targets_train_r = [datasets[action_name][0][i][3] for i in range(len(datasets[action_name][0]))]
        targets_test_c = [datasets[action_name][1][i][2] for i in range(len(datasets[action_name][1]))]
        targets_test_r = [datasets[action_name][1][i][3] for i in range(len(datasets[action_name][1]))]
        for i in range(len(targets_train_c)):
            print(targets_train_c[i].sum().item())
            print(targets_train_c_pred[i].sum().item())
        for i in range(len(targets_test_c)):
            print(targets_test_c[i].sum().item())
            print(targets_test_c_pred[i].sum().item())

        s, res = evaluator(model, datasets[action_name][0], datasets[action_name][1], [targets_train_c, targets_train_r], [targets_test_c, targets_test_r])
        for key, value in res.items():
            log(f"{key}: {value}")

        log(f"Pruning model for action '{action_name}'...")
        # prune_results[action_name] = prune_smallest_weights_iteratively(model, lambda m: evaluator(m, datasets[action_name][0], datasets[action_name][1], [targets_train_c, targets_train_r], [targets_test_c, targets_test_r]), exp_args, action_name, verbose=True)
        model = prune_results[action_name] = prune_weights_batchwise(model, lambda m: evaluator(m, datasets[action_name][0], datasets[action_name][1], [targets_train_c, targets_train_r], [targets_test_c, targets_test_r]), exp_args, action_name, verbose=True)
        save_model(model, exp_args, action_name+"_pruned")
        model2 = prune_results[action_name] = prune_weights_batchwise(model, lambda m: evaluator(model, datasets[action_name][0], datasets[action_name][1], [targets_train_c, targets_train_r], [targets_test_c, targets_test_r]), exp_args, action_name+"_rerun", verbose=True)
        save_model(model2, exp_args, action_name+"_pruned_rerun")
        model3 = prune_results[action_name] = prune_weights_batchwise(model, lambda m: evaluator(model2, datasets[action_name][0], datasets[action_name][1], [targets_train_c, targets_train_r], [targets_test_c, targets_test_r]), exp_args, action_name+"_rerun1", verbose=True)
        save_model(model3, exp_args, action_name+"_pruned_rerun1")
        model4 = prune_results[action_name] = prune_weights_batchwise(model, lambda m: evaluator(model3, datasets[action_name][0], datasets[action_name][1], [targets_train_c, targets_train_r], [targets_test_c, targets_test_r]), exp_args, action_name+"_rerun2", verbose=True)
        save_model(model4, exp_args, action_name+"_pruned_rerun2")
        model5 = prune_results[action_name] = prune_weights_batchwise(model, lambda m: evaluator(model4, datasets[action_name][0], datasets[action_name][1], [targets_train_c, targets_train_r], [targets_test_c, targets_test_r]), exp_args, action_name+"_rerun3", verbose=True)
        save_model(model5, exp_args, action_name+"_pruned_rerun3")
        model6 = prune_results[action_name] = prune_weights_batchwise(model, lambda m: evaluator(model5, datasets[action_name][0], datasets[action_name][1], [targets_train_c, targets_train_r], [targets_test_c, targets_test_r]), exp_args, action_name+"_rerun4", verbose=True)
        save_model(model6, exp_args, action_name+"_pruned_rerun4")
        model7 = prune_results[action_name] = prune_weights_batchwise(model, lambda m: evaluator(model6, datasets[action_name][0], datasets[action_name][1], [targets_train_c, targets_train_r], [targets_test_c, targets_test_r]), exp_args, action_name+"_rerun5", verbose=True)
        save_model(model7, exp_args, action_name+"_pruned_rerun5")
        description, concept, roles = model_to_verbal_description(model7, action_name)
        description_path = os.path.join(exp_args.experiment_path, "model_descriptions")
        os.makedirs(description_path, exist_ok=True)
        with open(os.path.join(description_path, f"pruned_model_description_{action_name}.txt"), 'w') as f:
            for key, value in description.items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(description_path, f"pruned_model_concept_names_{action_name}.txt"), 'w') as f:
            for i in concept:
                f.write(f"{i}\n")
        with open(os.path.join(description_path, f"pruned_model_role_names_{action_name}.txt"), 'w') as f:
            for i in roles:
                f.write(f"{i}\n")
        log(description)
        log(concept)
        log(roles)
        log(f"Pruned model for action '{action_name}' saved and described.")
    log(f"Pruning completed for all actions.")
    return prune_results


def model_to_verbal_description(model, action_name, names=None):
    """`names` overrides the built-in blocks-world naming with a dict holding
    the keys 'c', 'r', 'c_out' and 'r_out'."""

    per_action_name ={
        "unstack": {
            'c': ['clear', 'handempty', 'holding', 'number', 'object', 'ontable', 'arg_0'], 
            'r': ['on'], 
            "c_out": ['Prediction'], 
            "r_out": []
        },
        "putdown": {
            'c': ['clear', 'handempty', 'holding', 'number', 'object', 'ontable'], 
            'r': ['on'],
            "c_out": ['Prediction'], 
            "r_out": []
        },
        "stack": {
            'c': ['clear', 'handempty', 'holding', 'number', 'object', 'ontable', 'arg_0'], 
            'r': ['on'],
            "c_out": ['Prediction'], 
            "r_out": []
        },
        "pickup": {
            'c': ['clear', 'handempty', 'holding', 'number', 'object', 'ontable', "arg_0"], 
            'r': ['on'],
            "c_out": ['Prediction'], 
            "r_out": []
        }
    }
    
    per_action_name ={
        "unstack": {
            'c': ['clear', 'handempty', 'holding', 'number', 'object', 'ontable', 'arg_0'], 
            'r': ['on'], 
            "c_out": ["clear_add", "handempty_add", "holding_add", "ontable_add", "clear_del", "handempty_del", "holding_del", "ontable_del"],
            "r_out": ["on_add", "on_del"],

        },
        "putdown": {
            'c': ['clear', 'handempty', 'holding', 'number', 'object', 'ontable'], 
            'r': ['on'],
            "c_out": ["clear_add", "handempty_add", "holding_add", "ontable_add", "clear_del", "handempty_del", "holding_del", "ontable_del"],
            "r_out": ["on_add", "on_del"],
        },
        "stack": {
            'c': ['clear', 'handempty', 'holding', 'number', 'object', 'ontable', 'arg_0'], 
            'r': ['on'],
            "c_out": ["clear_add", "handempty_add", "holding_add", "ontable_add", "clear_del", "handempty_del", "holding_del", "ontable_del"],
            "r_out": ["on_add", "on_del"],
        },
        "pickup": {
            'c': ['clear', 'handempty', 'holding', 'number', 'object', 'ontable', 'arg_0'], 
            'r': ['on'],
            "c_out": ["clear_add", "handempty_add", "holding_add", "ontable_add", "clear_del", "handempty_del", "holding_del", "ontable_del"],
            "r_out": ["on_add", "on_del"],
        }
    }
    if names is None:
        names = per_action_name[action_name]

    input_concept_names = names['c']
    input_role_names = names['r']


    def stage_1_concept_map(in_concept_names, in_role_names, out_index, tc):

        n_c = len(in_concept_names)
        n_r = len(in_role_names)
        if out_index < n_c:
            return in_concept_names[out_index], zero_status[in_concept_names[out_index]]
        out_index -= n_c
        if out_index < 2*n_c*n_r:
            concept_idx = out_index // (2*n_r)
            role_idx = out_index % (2*n_r)
            if role_idx>=n_r:
                role_idx-=n_r
                return f"cra_min({in_concept_names[concept_idx]},{in_role_names[role_idx]})", zero_status[in_concept_names[concept_idx]] or zero_status[in_role_names[role_idx]]
            else:
                return f"cra_max({in_concept_names[concept_idx]},{in_role_names[role_idx]})", zero_status[in_concept_names[concept_idx]] or zero_status[in_role_names[role_idx]]
        out_index -= 2*n_c*n_r
        if out_index < n_r:
            role_idx = out_index
            return f"agg_max({in_role_names[role_idx]})", zero_status[in_role_names[role_idx]]
        out_index -= n_r
        if out_index < n_r:
            role_idx = out_index
            return f"agg_min({in_role_names[role_idx]})", zero_status[in_role_names[role_idx]]
        else:
            return f"unknown_{out_index}", True

    def stage_1_role_map(in_concept_names, in_role_names, out_index, tc):
        n_c = len(in_concept_names)
        n_r = len(in_role_names)
        if out_index < n_r:
            return in_role_names[out_index], zero_status[in_role_names[out_index]]
        out_index -= n_r
        if tc:
            if out_index < n_r:
                return f"tc({in_role_names[out_index]})", zero_status[in_role_names[out_index]]
            out_index -= n_r
        if out_index < 2*n_r*n_r:
            role_idx_1 = out_index // (2*n_r)
            role_idx_2 = out_index % (2*n_r)
            if role_idx_2>=n_r:
                role_idx_2-=n_r
                return f"rra_min({in_role_names[role_idx_1]},{in_role_names[role_idx_2]})", zero_status[in_role_names[role_idx_1]] or zero_status[in_role_names[role_idx_2]]
            else: 
                return f"rra_max({in_role_names[role_idx_1]},{in_role_names[role_idx_2]})", zero_status[in_role_names[role_idx_1]] or zero_status[in_role_names[role_idx_2]]
        out_index -= 2*n_r*n_r
        if out_index < n_r:
            return f"tr({in_role_names[out_index]})", zero_status[in_role_names[out_index]]
        out_index -= n_r
        if out_index < n_c:
            return f"ext_0({in_concept_names[out_index]})", zero_status[in_concept_names[out_index]]
        out_index -= n_c
        if out_index < n_c:
            return f"ext_1({in_concept_names[out_index]})", zero_status[in_concept_names[out_index]]
        else:
            return f"unknown_{out_index}", True


    def stringify_weight(weight_matrix, bias, in_names, c_or_r="c", zero_status_in=[]):
        if c_or_r == "r":
            weight_matrix = weight_matrix.transpose(0, 1)
            bias = bias.squeeze(0)
        descriptions = []
        for i_out in range(weight_matrix.size(0)):
            is_zero = False
            descriptions_for_i = ""
            for i_in in range(weight_matrix.size(1)):
                if weight_matrix[i_out][i_in].item() != 0:
                    if not zero_status_in[i_in]:
                        is_zero = False
                    descriptions_for_i += f"{weight_matrix[i_out][i_in]:.4f}*{in_names[i_in]} + "
            if bias[i_out] != 0:
                descriptions_for_i += f"{bias[i_out]:.4f}"
                is_zero = False
            descriptions_for_i = descriptions_for_i.rstrip(" + ")
            descriptions.append((descriptions_for_i, is_zero))
        return descriptions

    model.eval()
    config = model.config 
    tc = config.TRANSITIVE_CLOSURE
    concept_names_per_layer = [input_concept_names]
    role_names_per_layer = [input_role_names]
    descriptions_per_layer = {}
    zero_status = {i : False for i in input_concept_names + input_role_names}
    for layer_index in range(config.NUM_LAYERS):
        layer = model.layers[layer_index]
        in_concepts = layer.in_concepts
        in_roles = layer.in_roles
        out_concepts = layer.out_concepts
        out_roles = layer.out_roles
        
        in_concept_names = concept_names_per_layer[-1]
        in_role_names = role_names_per_layer[-1]
 
        intermediate_concept_names = []
        intermediate_role_names = []
        local_zero_status_c = []
        for intermediate_c_index in range(layer.total_intermediate_c):
            concept, is_zero = stage_1_concept_map(in_concept_names, in_role_names, intermediate_c_index, tc)
            intermediate_concept_names.append(concept)
            local_zero_status_c.append(is_zero)

        local_zero_status_r = []
        for intermediate_r_index in range(layer.total_intermediate_r):
            role, is_zero    = stage_1_role_map(in_concept_names, in_role_names, intermediate_r_index, tc)
            intermediate_role_names.append(role)
            local_zero_status_r.append(is_zero)
        # print("zero_status", local_zero_status_c)
        # print("zero_status r", local_zero_status_r)
        layer_concept_weight = layer.C_ffn.weight.data
        layer_role_weight = layer.R_ffn.weight.data
        layer_concept_bias = layer.C_ffn.bias.data
        layer_role_bias = layer.R_ffn.bias.data
        print(layer_index)
        print("concepts weights", layer_concept_weight)
        print("concept bias", layer_concept_bias)
        print("concept names", intermediate_concept_names)
        concept_descriptions = stringify_weight(layer_concept_weight, layer_concept_bias, intermediate_concept_names, c_or_r="c", zero_status_in=local_zero_status_c)
        role_descriptions = stringify_weight(layer_role_weight, layer_role_bias, intermediate_role_names, c_or_r="r", zero_status_in=local_zero_status_r)
        for i, (desc, is_zero) in enumerate(concept_descriptions):
            name = f"C_{layer_index}_{i}" if layer_index < config.NUM_LAYERS-1 else names["c_out"][i]
            descriptions_per_layer[name] = desc
            zero_status[name] = is_zero
        for i, (desc, is_zero) in enumerate(role_descriptions):
            name = f"R_{layer_index}_{i}" if layer_index < config.NUM_LAYERS-1 else names["r_out"][i] 
            descriptions_per_layer[name] = desc
            zero_status[name] = is_zero
        
        
        concept_names_per_layer.append([f"C_{layer_index}_{i}" if layer_index < config.NUM_LAYERS-1 else names["c_out"][i] for i in range(out_concepts)])
        role_names_per_layer.append([f"R_{layer_index}_{i}" if layer_index < config.NUM_LAYERS-1 else names["r_out"][i] for i in range(out_roles)])
    return descriptions_per_layer, concept_names_per_layer, role_names_per_layer



if __name__ == "__main__":
    
    # model_paths = [Path("/work/rleap1/jan.dornhege/NDLM/outputs/prune_results/ACTION_MODEL_precL2h4_correct_sil/1785401246"),
    #         Path("/work/rleap1/jan.dornhege/NDLM/outputs/prune_results/ACTION_MODEL_precL2h4_correct/1785401246"),
    #         Path("/work/rleap1/jan.dornhege/NDLM/outputs/prune_results/ACTION_MODEL_precL2h4_correct_l/1785401246"),
    #         Path("/work/rleap1/jan.dornhege/NDLM/outputs/prune_results/ACTION_MODEL_precL2h4_correct_ll/1785401246"),
    #         Path("/work/rleap1/jan.dornhege/NDLM/outputs/prune_results/ACTION_MODEL_precL2h4_correct_siil/1785401246"),
    #         Path("/work/rleap1/jan.dornhege/NDLM/outputs/prune_results/ACTION_MODEL_precL2h4_correct_sl/1785401246")]
    model_paths = [Path("/work/rleap1/jan.dornhege/NDLM/outputs/prune_results/ACTION_MODEL_effL1h4_rerun/1785240374")]
    from ndlm.configs import config_from_json_file
    from pathlib import Path

    actions = ["unstack", "stack", "pickup", "putdown"]

    for model_path in model_paths:
        print(model_path.parent.name)
        for action in actions:
            print(action)
            # io_dimmensions = {'in_concepts': 6, 'in_roles': 1, 'out_concepts': 1, 'out_roles': 0}

            io_dimmensions = {'in_concepts': 6, 'in_roles': 1, 'out_concepts': 8, 'out_roles': 2}
            if action != "putdown":
                io_dimmensions["in_concepts"]=7  
            in_concepts = io_dimmensions['in_concepts']
            in_roles = io_dimmensions['in_roles']
            out_concepts = io_dimmensions['out_concepts']
            out_roles = io_dimmensions['out_roles']


            if not os.path.isfile(model_path / "pruned_models" / f"pruned_model_config_{action}_pruned_rerun5.json"):
                continue
            config = config_from_json_file(model_path / "pruned_models" / f"pruned_model_config_{action}_pruned_rerun5.json")
            
            model_path_file = model_path / "pruned_models" / f"model_{action}_pruned_rerun5.pt"
            if not os.path.isfile(model_path_file):
                continue
            model = load_model_from_file(in_concepts, in_roles, out_concepts, out_roles, config, model_path_file)
            log_path = model_path / "model_descriptions"
            desc, c, r = model_to_verbal_description(model, action)

            os.makedirs(log_path, exist_ok=True)
            
            with open(os.path.join(log_path, f"pruned_model_description_{action}.txt"), 'w') as f:
                for key, value in desc.items():
                    f.write(f"{key}: {value}\n")
                    print(f"{key}: {value}")
            with open(os.path.join(log_path, f"pruned_model_concept_names_{action}.txt"), 'w') as f:
                for i in c:
                    f.write(f"{i}\n")
            with open(os.path.join(log_path, f"pruned_model_role_names_{action}.txt"), 'w') as f:
                for i in r:
                    f.write(f"{i}\n")
                
    
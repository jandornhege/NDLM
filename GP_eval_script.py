import os
import argparse

import torch
import pymimir.advanced.formalism as formalism
import pymimir.advanced.search as search


import mimir_to_tensors


gripper_instance_paths= [
    "gripper/train/gripper_b-16.pddl",
    "gripper/train/gripper_b-18.pddl",
    "gripper/train/gripper_b-20.pddl",
    "gripper/train/gripper_b-22.pddl",
    "gripper/train/gripper_b-24.pddl",
    "gripper/train/gripper_b-26.pddl",
    "gripper/train/gripper_b-28.pddl",
    "gripper/train/gripper_b-30.pddl",
    "gripper/train/gripper_b-32.pddl",
    "gripper/train/gripper_b-34.pddl",
    "gripper/train/gripper_b-36.pddl",
    "gripper/train/gripper_b-38.pddl",
    "gripper/train/gripper_b-40.pddl",
    "gripper/train/gripper_b-42.pddl",
    "gripper/train/gripper_b-44.pddl",
    "gripper/train/gripper_b-46.pddl",
    "gripper/train/gripper_b-48.pddl",
    "gripper/train/gripper_b-50.pddl",
    "gripper/test_original/gripper_b-16.pddl",
    "gripper/test_original/gripper_b-18.pddl",
    "gripper/test_original/gripper_b-20.pddl",
    "gripper/test_original/gripper_b-22.pddl",
    "gripper/test_original/gripper_b-24.pddl",
    "gripper/test_original/gripper_b-26.pddl",
    "gripper/test_original/gripper_b-28.pddl",
    "gripper/test_original/gripper_b-30.pddl",
    "gripper/test_original/gripper_b-32.pddl",
    "gripper/test_original/gripper_b-34.pddl",
    "gripper/test_original/gripper_b-36.pddl",
    "gripper/test_original/gripper_b-38.pddl",
    "gripper/test_original/gripper_b-40.pddl",
    "gripper/test_original/gripper_b-42.pddl",
    "gripper/test_original/gripper_b-44.pddl",
    "gripper/test_original/gripper_b-46.pddl",
    "gripper/test_original/gripper_b-48.pddl",
    "gripper/test_original/gripper_b-50.pddl"
]

logistics_instance_paths = [
    "logistics/train/problem_cities-2_packages-3_v-1.pddl",
    "logistics/train/problem_cities-2_packages-3_v-2.pddl",
    "logistics/train/problem_cities-2_packages-3_v-3.pddl",
    "logistics/train/problem_cities-2_packages-4_v-1.pddl",
    "logistics/train/problem_cities-2_packages-4_v-2.pddl",
    "logistics/train/problem_cities-2_packages-4_v-3.pddl",
    "logistics/train/problem_cities-2_packages-5_v-1.pddl",
    "logistics/train/problem_cities-2_packages-5_v-2.pddl",
    "logistics/train/problem_cities-2_packages-5_v-3.pddl",
    "logistics/train/problem_cities-3_packages-3_v-1.pddl",
    "logistics/train/problem_cities-3_packages-3_v-2.pddl",
    "logistics/train/problem_cities-3_packages-3_v-3.pddl",
    "logistics/train/problem_cities-3_packages-4_v-1.pddl",
    "logistics/train/problem_cities-3_packages-4_v-2.pddl",
    "logistics/train/problem_cities-3_packages-4_v-3.pddl",
    "logistics/train/problem_cities-4_packages-3_v-1.pddl",
    "logistics/train/problem_cities-4_packages-3_v-2.pddl",
    "logistics/train/problem_cities-4_packages-3_v-3.pddl",
    "logistics/train/problem_cities-4_packages-4_v-1.pddl",
    "logistics/train/problem_cities-4_packages-4_v-2.pddl",
    "logistics/train/problem_cities-4_packages-4_v-3.pddl",
    "logistics/train/problem_cities-5_packages-3_v-1.pddl",
    "logistics/train/problem_cities-5_packages-3_v-2.pddl",
    "logistics/train/problem_cities-5_packages-3_v-3.pddl",
    "logistics/test/problem_cities-15_packages-8.pddl",
    "logistics/test/problem_cities-15_packages-9.pddl",
    "logistics/test/problem_cities-15_packages-10.pddl",
    "logistics/test/problem_cities-15_packages-11.pddl",
    "logistics/test/problem_cities-16_packages-8.pddl",
    "logistics/test/problem_cities-16_packages-9.pddl",
    "logistics/test/problem_cities-16_packages-10.pddl",
    "logistics/test/problem_cities-16_packages-11.pddl",
    "logistics/test/problem_cities-17_packages-8.pddl",
    "logistics/test/problem_cities-17_packages-9.pddl",
    "logistics/test/problem_cities-17_packages-10.pddl",
    "logistics/test/problem_cities-17_packages-11.pddl",
    "logistics/test/problem_cities-18_packages-8.pddl",
    "logistics/test/problem_cities-18_packages-9.pddl",
    "logistics/test/problem_cities-18_packages-10.pddl",
    "logistics/test/problem_cities-18_packages-11.pddl",
    "logistics/test/problem_cities-19_packages-8.pddl",
    "logistics/test/problem_cities-19_packages-9.pddl",
    "logistics/test/problem_cities-19_packages-10.pddl",
    "logistics/test/problem_cities-19_packages-11.pddl"
]
logistics_domain_path = "logistics/domain.pddl"
gripper_domain_path = "gripper/domain.pddl"

logistics_parameter_indices = {
        "load-truck": [
            0,
            1
        ],
        "load-airplane": [
            0,
            1
        ],
        "unload-truck": [
            0,
            1
        ],
        "unload-airplane": [
            0,
            1
        ],
        "drive-truck": [
            0,
            2
        ],
        "fly-airplane": [
            0,
            2
        ]
} 
gripper_parameter_indices = {
        "move": [
            0,
            1
        ],
        "pick": [
            0,
            2
        ],
        "drop": [
            0,
            2
        ]
    }


logistics_models_per_action = {
    "load-truck": None,
    "load-airplane": None,
    "unload-truck": None,
    "unload-airplane": None,
    "drive-truck": None,
    "fly-airplane": None
}
gripper_models_per_action = {
    "move": None,
    "pick": None,
    "drop": None
}

DOMAIN_CONFIGS = {
    "gripper": {
        "domain_path": gripper_domain_path,
        "instance_paths": gripper_instance_paths,
        "parameter_indices": gripper_parameter_indices,
        "models_per_action": gripper_models_per_action,
    },
    "logistics": {
        "domain_path": logistics_domain_path,
        "instance_paths": logistics_instance_paths,
        "parameter_indices": logistics_parameter_indices,
        "models_per_action": logistics_models_per_action,
    },
}


def test_model_on_test_problems(domain_path, test_instance_paths, models_per_action, parameter_indices, max_steps, threshold=0.5):
    # The action-model evaluator is always run in the deterministic full state-action mode.
    # One-step cycle checking is optional and defaults to disabled.
    results = {}
    

    for test_instance_path in test_instance_paths:
        problem = formalism.Problem.create(domain_path, test_instance_path, formalism.ParserOptions())
        ctx_opts = search.SearchContextOptions(search.LiftedOptions(search.LiftedKPKCOptions()))
        search_context = search.SearchContext.create(problem, ctx_opts)
        state_repo = search_context.get_state_repository()

        aag = search.KPKCLiftedApplicableActionGenerator.create(problem, search.LiftedKPKCOptions())
        init, _ = state_repo.get_or_create_initial_state()
        opt_dist = 0

        current_state = init
        steps_taken = 0
        goal_reached = False
        print(f"Testing on problem: {str(test_instance_path).split('/')[-1]} with max_steps: {max_steps}")
        seen = set()
        seen.add(current_state)
        res = "max_steps"

        def _atom_signature(atom):
            pred = atom.get_predicate().get_name()
            objs = tuple(obj.get_index() for obj in atom.get_objects())
            return (pred, objs)

        def _state_satisfies_goal_condition(s, problem_obj):
            repos = problem_obj.get_repositories()
            if hasattr(s, "get_atoms"):
                state_atoms = s.get_atoms(ignore_static=False, ignore_fluent=False, ignore_derived=True)
            elif hasattr(s, "get_fluent_atoms"):
                fluent_indices = list(s.get_fluent_atoms())
                state_atoms = list(repos.get_fluent_ground_atoms_from_indices(fluent_indices)) + list(problem_obj.get_static_initial_atoms())
            else:
                raise TypeError("Unsupported state object: expected get_atoms or get_fluent_atoms")

            state_atom_signatures = {_atom_signature(atom) for atom in state_atoms}
            goal_condition = problem_obj.get_goal_condition()

            try:
                pos_static = list(goal_condition.get_static_positive_condition())
                pos_fluent = list(goal_condition.get_fluent_positive_condition())
                neg_static = list(goal_condition.get_static_negative_condition())
                neg_fluent = list(goal_condition.get_fluent_negative_condition())

                for atom in repos.get_static_ground_atoms_from_indices(pos_static):
                    if _atom_signature(atom) not in state_atom_signatures:
                        return False
                for atom in repos.get_fluent_ground_atoms_from_indices(pos_fluent):
                    if _atom_signature(atom) not in state_atom_signatures:
                        return False
                for atom in repos.get_static_ground_atoms_from_indices(neg_static):
                    if _atom_signature(atom) in state_atom_signatures:
                        return False
                for atom in repos.get_fluent_ground_atoms_from_indices(neg_fluent):
                    if _atom_signature(atom) in state_atom_signatures:
                        return False
                return True
            except AttributeError:
                for literal in goal_condition.get_literals(ignore_derived=True):
                    atom_sig = _atom_signature(literal.get_atom())
                    if literal.get_polarity() and atom_sig not in state_atom_signatures:
                        return False
                    if (not literal.get_polarity()) and atom_sig in state_atom_signatures:
                        return False
                return True

        def _is_goal_state(s):
            return _state_satisfies_goal_condition(s, problem)

        if _is_goal_state(current_state):
            print("Initial state is already a goal state.")
            results[test_instance_path] = {
                "final_state": current_state,
                "steps_taken": steps_taken,
                "goal_reached": True,
                "reason": "initial_goal",
                "optimal_steps": opt_dist,
            }
            continue
        if _is_goal_state(init):
            print("Initial state is already a goal state.")
            results[test_instance_path] = {
                "final_state": init,
                "steps_taken": steps_taken,
                "goal_reached": True,
                "reason": "initial_goal",
                "optimal_steps": opt_dist,
            }
            continue

        while steps_taken < max_steps:
            actions = aag.generate_applicable_actions(current_state)
            if not actions:
                print("No applicable actions available, stopping execution.")
                break

            action_scores = {}
            preds_per_action_name = {}
        
            for name in models_per_action:
                concepts, roles, _, _ = mimir_to_tensors.state_goal_to_concept_role_data(
                    current_state,
                    problem,
                    name,
                    parameter_indices=parameter_indices,
                )
                num_objects = concepts.shape[-1]
                model = models_per_action[name]
                out_concepts, out_roles = model(concepts, roles)
                preds_per_action_name[name] = (
                    out_concepts,
                    out_roles,
                    num_objects,
                )

            for a in actions:
                action_name = a.get_action().get_name()
                effective_action_arity = len(a.get_objects()) if parameter_indices is None or parameter_indices[action_name] is None else len(parameter_indices[action_name])
                parameter_indices_action = parameter_indices[action_name] if parameter_indices is not None and parameter_indices[action_name] is not None else list(range(effective_action_arity))
                action_object_indices = tuple(obj.get_index() for obj in a.get_objects())
                out_concepts, out_roles, num_objects = preds_per_action_name[action_name]

                if effective_action_arity == 0:
                    action_scores[a] = out_concepts.sum().item() / num_objects
                elif effective_action_arity == 1:
                    action_scores[a] = out_concepts[0, 0, action_object_indices[parameter_indices_action[0]]].item()
                elif effective_action_arity == 2:
                    action_scores[a] = out_roles[0, 0, action_object_indices[parameter_indices_action[0]], action_object_indices[parameter_indices_action[1]]].item()
                else:
                    raise ValueError(f"Unsupported action arity for {action_name}: {effective_action_arity}")

            all_scores = action_scores.copy()
            if threshold is not None:
                action_scores = {a: score for a, score in action_scores.items() if score >= threshold}
            if not action_scores:
                max_score = max(all_scores.values(), default=float('-inf'))
                print(f"No valid actions available with score above threshold, stopping execution. Max score is {max_score}.")
                res = "no-action"
                goal_reached = False
                break

            ranked_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
            chosen_action = None
            chosen_score = None
            chosen_successor = None

        
            chosen_action, chosen_score = ranked_actions[0]
            chosen_successor, _ = state_repo.get_or_create_successor_state(current_state, chosen_action, 0.0)

            print(
                f"Chosen action: {chosen_action.get_action().get_name()}, "
                f"Num valid Actions:{len(action_scores):.2f}, Score: {chosen_score}"
            )
            current_state = chosen_successor
            seen.add(current_state)
            steps_taken += 1

            goal_reached = _is_goal_state(current_state)
            if goal_reached:
                res = "success"
                break

        results[test_instance_path] = {
            "final_state": current_state,
            "steps_taken": steps_taken,
            "goal_reached": goal_reached,
            "reason": res,
            "optimal_steps": opt_dist,
        }
        print(f"Finished testing on problem: {str(test_instance_path).split('/')[-1]}, Steps taken: {steps_taken}, Goal reached: {goal_reached}, Reason: {res}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate action models on Gripper or Logistics problems.")
    parser.add_argument("--domain", choices=["gripper", "logistics"], default="gripper")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = DOMAIN_CONFIGS[args.domain]

    # Fill these entries with the trained model object for each action before running.
    models_per_action = {name: None for name in config["models_per_action"]}

    results = test_model_on_test_problems(
        domain_path=config["domain_path"],
        test_instance_paths=config["instance_paths"],
        models_per_action=models_per_action,
        parameter_indices=config["parameter_indices"],
        max_steps=args.max_steps,
        threshold=args.threshold,
    )

    print(f"Domain: {args.domain}")
    print(f"Problems evaluated: {len(results)}")
    print(f"Solved: {sum(r['goal_reached'] for r in results.values())}")
    print(f"Success rate: {sum(r['goal_reached'] for r in results.values()) / max(len(results), 1) * 100:.2f}%")
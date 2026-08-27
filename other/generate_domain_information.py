import os
import argparse
import json
import pymimir.advanced.formalism as formalism


def get_domain_information(domain_dir):
    domain_path = os.path.join(domain_dir, "domain.pddl")
    # file_names = os.listdir(os.path.join(domain_dir, "train"))
    #only ending in .pddl and not domain.pddl
    if os.path.exists(os.path.join(domain_dir, "train")):
        file_names = os.listdir(os.path.join(domain_dir, "train"))
        train_path = os.path.join(domain_dir, "train")
    else:
        file_names = os.listdir(domain_dir)
        train_path = domain_dir
    train_instances = [f for f in file_names if f.endswith(".pddl") and f != "domain.pddl"]
    train_instances.sort()
    print(domain_dir, train_instances)
    instance_0_path = os.path.join(train_path, train_instances[0])

    problem = formalism.Problem.create(domain_path, instance_0_path, formalism.ParserOptions())

    domain = problem.get_domain()
    action_names = [action.get_name() for action in domain.get_actions()]
    action_arities = {action.get_name(): len(action.get_parameters()) for action in domain.get_actions()}
    predicate_names = [predicate.get_name() for predicate in domain.get_static_predicates()]+[predicate.get_name() for predicate in domain.get_fluent_predicates()]
    test_instance_names = list(os.listdir(os.path.join(domain_dir, "test")))
    test_instance_names.sort()
    argument_indices = {action.get_name(): [i for i in range(len(action.get_parameters()))] for action in domain.get_actions()}
    return {
        "action_names": action_names,
        "argument_indices": argument_indices,
        "action_arities": action_arities,
        "predicate_names": predicate_names,
        "train_instance_names": train_instances,
        "test_instance_names": test_instance_names,
        "eval_instance_names": [],
    }
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--domain-dir", type=str)

    args = args.parse_args()
    # res = get_domain_information(args.domain_dir)
    for exp in os.listdir(args.domain_dir):
        print(exp)
        domain_path = os.path.join(args.domain_dir, exp)
        res = get_domain_information(domain_path)
        with open(os.path.join(domain_path, "domain_information.json"), "w") as f:
            json.dump(res, f, indent=4)
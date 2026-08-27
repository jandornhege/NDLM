#!/usr/bin/env python3

from cProfile import label
from cProfile import label
from collections import deque
from collections import defaultdict
import pymimir.advanced.formalism as formalism
import pymimir.advanced.search as search
import torch
import os
import json
import random
import time
import numpy as np
import ndlm.modules as modules
from my_logging import log

try: 
    import random_trace_generator.general_policy_sampler as gps
    import random_trace_generator.opt_gp_util as opt_gp_util
    import random_trace_generator.mimir_to_tensors as mimir_to_tensors
except:
    import general_policy_sampler as gps
    import opt_gp_util
    import mimir_to_tensors as mimir_to_tensor

def load_parameter_indices(name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir,f"necessary_parameters/{name}.json"), "r") as f:
        parameter_indices = json.load(f)
    return parameter_indices["result"]

def load_predicate_indices(name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir,f"necessary_parameters/{name}_predicates.json"), "r") as f:
        predicate_indices = json.load(f)
    return predicate_indices["result"]

def mimir_state_to_concept_role_data(state, problem, only_fluid=False, padding=0, predicate_indices=set()):
    out_roles = []
    out_concepts = []
    concept_p = []
    role_p = []
    # Prefer the state-attached problem/repository to avoid index-space mismatches
    # when states are sampled from a different context than a separately parsed problem.
    active_problem = problem
    if hasattr(state, "get_state_repository"):
        state_repository = state.get_state_repository()
        if hasattr(state_repository, "get_problem"):
            active_problem = state_repository.get_problem()

    domain = active_problem.get_domain()

    # collect all predicate names and check if all are unary or binary
    fp = list(domain.get_fluent_predicates() )
    sp = list(domain.get_static_predicates())
    if only_fluid:
        sp = []
    preds= fp+sp
    preds.sort(key=lambda p: p.get_name())
    for p in preds:
        if p.get_arity() == 1:
            if p.get_name() not in predicate_indices:
                concept_p.append(p.get_name())
        elif p.get_arity() == 0:
            if p.get_name() not in predicate_indices:
                concept_p.append(p.get_name())
        elif p.get_arity() == 2:
            if p.get_name() not in predicate_indices:
                role_p.append(p.get_name())
        elif p.get_arity() > 2:
            raise ValueError(f"Predicate {p.get_name()} has arity > 2, which is not supported")
        else:
            raise ValueError(f"Predicate {p.get_name()} has unsupported arity {p.get_arity()}")
    repositories = active_problem.get_repositories()
    if hasattr(state, "get_atoms"):
        atoms = state.get_atoms(ignore_static=only_fluid, ignore_fluent=False, ignore_derived=True)
    elif hasattr(state, "get_fluent_atoms"):
        fa = list(state.get_fluent_atoms())
        sa = list(active_problem.get_static_initial_atoms())
        if only_fluid:
            sa = []
        atoms = list(repositories.get_fluent_ground_atoms_from_indices(fa)) + sa
    else:
        raise TypeError("Unsupported state object: expected get_atoms or get_fluent_atoms")
    
    num_objects=len(active_problem.get_objects())

    #initialize empty tensors
    concepts_tensor = torch.zeros((len(concept_p), num_objects+padding), dtype=torch.float32)
    roles_tensor = torch.zeros((len(role_p), num_objects+padding, num_objects+padding), dtype=torch.float32)

    #fill with 1s where atoms are true in the state
    for atom in atoms:
        p = atom.get_predicate()
        if p.get_name() in concept_p:
            if p.get_arity() == 0:
                for i in range(num_objects+padding):
                    concepts_tensor[concept_p.index(p.get_name()), i] = 1.0
            else:
                concepts_tensor[concept_p.index(p.get_name()), atom.get_objects()[0].get_index()] = 1.0
        elif p.get_name() in role_p:
            roles_tensor[role_p.index(p.get_name()), atom.get_objects()[0].get_index(), atom.get_objects()[1].get_index()] = 1.0
    
    return concepts_tensor, roles_tensor, concept_p, role_p


def goal_condition_to_concept_role_data(problem, padding=0, predicate_indices=set(), include_negative=True):
    concept_p = []
    role_p = []
    domain = problem.get_domain()
    preds = list(domain.get_fluent_predicates()) + list(domain.get_static_predicates())
    preds.sort(key=lambda p: p.get_name())
    for p in preds:
        if p.get_arity() == 1:
            if p.get_name() not in predicate_indices:
                concept_p.append(p.get_name())
        elif p.get_arity() == 0:
            if p.get_name() not in predicate_indices:
                concept_p.append(p.get_name())
        elif p.get_arity() == 2:
            if p.get_name() not in predicate_indices:
                role_p.append(p.get_name())
        else:
            raise ValueError(f"Predicate {p.get_name()} has unsupported arity (0 or >2)")

    num_objects = len(problem.get_objects())
    concepts_tensor = torch.zeros((len(concept_p), num_objects + padding), dtype=torch.float32)
    roles_tensor = torch.zeros((len(role_p), num_objects + padding, num_objects + padding), dtype=torch.float32)

    def _write_atom(atom, polarity):
        if not polarity:
            log(f"Ignoring negative atom in goal condition (not supported in current setup): {atom}")
            return
        # print("writing atom:", atom, "polarity:", polarity)
        predicate = atom.get_predicate()
        name = predicate.get_name()
        if name in concept_p and predicate.get_arity() == 0:
            #fill whole concept row with polarity value
            for i in range(num_objects + padding):
                concepts_tensor[concept_p.index(name), i] = 1.0 if polarity else -1.0
        if name in concept_p and predicate.get_arity() == 1:
            obj_idx = atom.get_objects()[0].get_index()
            concepts_tensor[concept_p.index(name), obj_idx] = 1.0 if polarity else -1.0
        elif name in role_p and predicate.get_arity() == 2:
            obj1 = atom.get_objects()[0].get_index()
            obj2 = atom.get_objects()[1].get_index()
            roles_tensor[role_p.index(name), obj1, obj2] = 1.0 if polarity else -1.0
    goal_condition = problem.get_goal_condition()
    repositories = problem.get_repositories()

    try:
        pos_static = list(goal_condition.get_static_positive_condition())
        pos_fluent = list(goal_condition.get_fluent_positive_condition())
        neg_static = list(goal_condition.get_static_negative_condition()) if include_negative else []
        neg_fluent = list(goal_condition.get_fluent_negative_condition()) if include_negative else []

        for atom in repositories.get_static_ground_atoms_from_indices(pos_static):
            _write_atom(atom, True)
        for atom in repositories.get_fluent_ground_atoms_from_indices(pos_fluent):
            _write_atom(atom, True)
        for atom in repositories.get_static_ground_atoms_from_indices(neg_static):
            _write_atom(atom, False)
        for atom in repositories.get_fluent_ground_atoms_from_indices(neg_fluent):
            _write_atom(atom, False)
    except AttributeError:
        for literal in goal_condition.get_literals(ignore_derived=True):
            polarity = literal.get_polarity()
            if (not include_negative) and (not polarity):
                continue
            _write_atom(literal.get_atom(), polarity)

    return concepts_tensor, roles_tensor, concept_p, role_p


def action_parameters_to_concept_role_data(action, problem, parameter_indices, padding=0):
    action_p = action.get_action()
    domain = problem.get_domain()
    repositories = problem.get_repositories()
    param = action.get_objects()
    num_objects = len(problem.get_objects())
    if parameter_indices is None:
        parameter_indices = list(range(len(param)))
    
    out_concepts = torch.zeros((len(parameter_indices), num_objects+padding), dtype=torch.float32)
    for i, parameter_index in enumerate(parameter_indices):
        out_concepts[i, param[parameter_index].get_index()] = 1.0
    return out_concepts, [p.get_name() for p in param]

def transition_to_concept_role_data(s, a, s2, problem, parameter_indices, predicate_indices=set(), padding=0):
    c, r, c_n, r_n = mimir_state_to_concept_role_data(s, problem, padding=padding, predicate_indices=predicate_indices)
    c2, r2, c2_n, r2_n = mimir_state_to_concept_role_data(s2, problem, only_fluid=True, padding=padding, predicate_indices=predicate_indices)
    a_c, a_c_n = action_parameters_to_concept_role_data(a, problem, parameter_indices, padding=padding)
    
    #make targest to be only the difference(normalized to be 0 for removed and 1 for added concepts/roles)
    
    in_c = torch.cat([c, a_c], dim=0)
    out_c = c2
    for i, name in enumerate(c_n):
        if name in c2_n:
            out_c[c2_n.index(name),:]-=c[i,:]
    out_c1 = out_c.clone()
    out_c1 = torch.clamp(out_c1, min=0)
    out_c2 = torch.clamp(out_c, max=0)*(-1)
    out_c = torch.cat([out_c1, out_c2], dim=0)

    in_r = r
    out_r = r2
    for i, name in enumerate(r_n):
        if name in r2_n:
            out_r[r2_n.index(name),:,:]-=r[i,:,:]
    out_r1 = out_r.clone()
    out_r1 = torch.clamp(out_r1, min=0)
    out_r2 = torch.clamp(out_r, max=0)*(-1)
    out_r = torch.cat([out_r1, out_r2], dim=0)

    return (in_c, in_r,out_c, out_r), (c_n, r_n, a_c_n, c2_n, r2_n)


def state_action_to_applicability_data(s, a, problem, parameter_indices, is_applicable, padding=0, predicate_indices=set()):
    c, r, c_n, r_n = mimir_state_to_concept_role_data(s, problem, padding=padding, predicate_indices=predicate_indices)
    a_c, a_c_n = action_parameters_to_concept_role_data(a, problem, parameter_indices, padding=padding)
    in_c = torch.cat([c, a_c], dim=0)
    in_r = r
    num_objects = len(problem.get_objects())
    out_concepts = torch.ones((1,num_objects+padding), dtype=torch.float32) if is_applicable else torch.zeros((1,num_objects+padding), dtype=torch.float32)
    out_roles = torch.zeros((0, num_objects+padding, num_objects+padding), dtype=torch.float32)  # no role is predicted
    return (in_c, in_r, out_concepts, out_roles), (c_n, r_n, a_c_n)

def state_pair_to_concept_role_data(s, s2, problem, parameter_indices, label,  predicate_indices=set(), padding=0):
    c1, r1, c1_n, r1_n = mimir_state_to_concept_role_data(s, problem, padding=padding, predicate_indices=predicate_indices)
    c2, r2, c2_n, r2_n = mimir_state_to_concept_role_data(s2, problem, padding=padding, predicate_indices=predicate_indices)
    goal_c, goal_r, _, _ = goal_condition_to_concept_role_data(problem, padding=padding, predicate_indices=predicate_indices)
    in_c = torch.cat([c1, c2, goal_c], dim=0)
    in_r = torch.cat([r1, r2, goal_r], dim=0)
    num_objects = len(problem.get_objects())
    out_concepts = torch.ones((1,num_objects+padding), dtype=torch.float32) if label else torch.zeros((1,num_objects+padding), dtype=torch.float32)
    out_roles = torch.zeros((0, num_objects+padding, num_objects+padding), dtype=torch.float32)  # no role is predicted
    return (in_c, in_r, out_concepts, out_roles)

def get_full_applicability_data(s, problem, parameter_list, padding=0):
    agg = search.KPKCLiftedApplicableActionGenerator.create(problem, search.LiftedKPKCOptions())
    actions = agg.generate_applicable_actions(s)
    by_action_vector = {} 
    num_objects = len(problem.get_objects())

    for a in problem.get_domain().get_actions():
        action_name = a.get_name()
        if parameter_list[action_name] is None:
            parameter_indices = list(range(len(param)))
        else: 
            parameter_indices = parameter_list[action_name]
        if len(parameter_indices)==1 or len(parameter_indices)==0:            
            by_action_vector[action_name] = [torch.zeros((1, num_objects+padding), dtype=torch.float32), torch.zeros((0, num_objects+padding, num_objects+padding), dtype=torch.float32)]  
        elif len(parameter_indices)==2:
            by_action_vector[action_name] = [torch.zeros((0, num_objects+padding), dtype=torch.float32), torch.zeros((1, num_objects+padding, num_objects+padding), dtype=torch.float32)]

    for a in actions:
        action_name = a.get_action().get_name()
        if parameter_list[action_name] is None:
            parameter_indices = list(range(len(param)))
        else: 
            parameter_indices = parameter_list[action_name]
        # print(len(parameter_indices))
        param = a.get_objects()
        
        if len(parameter_indices)==1:
            by_action_vector[action_name][0][0, param[parameter_indices[0]].get_index()] = 1.0
        elif len(parameter_indices)==0:
            for i in range(num_objects+padding):
                by_action_vector[action_name][0][0, i] = 1.0
        elif len(parameter_indices)==2:
            by_action_vector[action_name][1][0, param[parameter_indices[0]].get_index(), param[parameter_indices[1]].get_index()] = 1.0
        else:
            raise ValueError(f"Unsupported number of parameters for action {action_name}: {len(parameter_indices)}")
    return by_action_vector
    
        
         

def _action_signature(ground_action):
    obj_indices = tuple(obj.get_index() for obj in ground_action.get_objects())
    return ground_action.get_action().get_name(), obj_indices


def _projected_action_signature(ground_action, parameter_indices):
    obj_indices = tuple(ground_action.get_objects()[i].get_index() for i in parameter_indices)
    return ground_action.get_action().get_name(), obj_indices


def _precompute_ground_actions_by_schema(
    problem,
    max_actions=5000,
    ignore_type_filtering=True,
    random_seed=None,
):
    all_objects = list(problem.get_problem_and_domain_objects())
    ground_actions_by_name = {}
    rng = random.Random(random_seed)

    for action_schema in problem.get_domain().get_actions():
        action_name = action_schema.get_name()
        parameter_domains = []

        for parameter in action_schema.get_parameters():
            if ignore_type_filtering:
                compatible_objects = all_objects
            else:
                parameter_type_indices = {t.get_index() for t in parameter.get_bases()}
                compatible_objects = []
                for obj in all_objects:
                    object_type_indices = {t.get_index() for t in obj.get_bases()}
                    if parameter_type_indices.intersection(object_type_indices):
                        compatible_objects.append(obj)
            parameter_domains.append(compatible_objects)

        grounded_actions = []
        if any(len(domain) == 0 for domain in parameter_domains):
            ground_actions_by_name[action_name] = grounded_actions
            continue

        if max_actions > 0:
            seen_bindings = set()
            max_attempts = max_actions * 20
            attempts = 0

            while len(grounded_actions) < max_actions and attempts < max_attempts:
                binding = tuple(rng.choice(domain) for domain in parameter_domains)
                signature = tuple(obj.get_index() for obj in binding)
                attempts += 1

                if signature in seen_bindings:
                    continue

                seen_bindings.add(signature)
                grounded_actions.append(problem.ground(action_schema, formalism.ObjectList(list(binding))))

        ground_actions_by_name[action_name] = grounded_actions

    return ground_actions_by_name


def generate_action_transitions_bfs(
    problem,
    max_states=100000,
    include_non_applicable=False,
    full_applicability=False,
    max_actions=5000,
    ignore_type_filtering=False,
    random_seed=None,
    parameter_indices=None,
    predicate_indices=set(),
    filter_function=None,
    max_sampling_seconds_per_problem=None,
):
    if filter_function is not None and include_non_applicable:
        raise ValueError("filter_function cannot be used when include_non_applicable is True, as it would be unclear how to apply the filter to non-applicable actions.")
    # generate transitions for target_action by exploring reachable states from initial state
      
    # Create applicable action generator and state repository with proper options
    aag = search.KPKCLiftedApplicableActionGenerator.create(problem, search.LiftedKPKCOptions())
    axiom_eval = search.KPKCLiftedAxiomEvaluator.create(problem)
    state_repo = search.StateRepository.create(axiom_eval)
    
    # Get initial state
    init, _ = state_repo.get_or_create_initial_state()
    q = deque([init])
    seen = {init}
    out = []  # (state, action, successor, is_applicable)
    sampling_start_time = time.time()

    ground_action_signatures = set()
    ground_actions_by_schema = _precompute_ground_actions_by_schema(problem, max_actions=max_actions, ignore_type_filtering=ignore_type_filtering, random_seed=random_seed)
    while q and len(seen) < max_states:
        if (
            max_sampling_seconds_per_problem is not None
            and max_sampling_seconds_per_problem > 0
            and (time.time() - sampling_start_time) >= max_sampling_seconds_per_problem
        ):
            log(f"Stopping BFS sampling early after {max_sampling_seconds_per_problem}s for problem (visited states: {len(seen)}, transitions: {len(out)}).")
            break
        #change to pop for DFS and popleft for BFS
        s = q.popleft()
        # Generate applicable actions
        actions = aag.generate_applicable_actions(s)
        observed_projected_signatures = {a.get_action().get_name(): set() for a in actions}
        for a in actions:
            # Apply action to get successor state
            s2, _ = state_repo.get_or_create_successor_state(s, a, 0.0)
            # if filter_function is not None:
            #     print("filter_application")
            #     out.append((s, a, s2, filter_function(s,s2)))
            #     print("filter_successful")
            # else:
            out.append((s, a, s2, True))
            if s2 not in seen:
                seen.add(s2)
                q.append(s2)
            action_name = a.get_action().get_name()
            if parameter_indices is None:
                observed_projected_signatures[action_name].add(_action_signature(a))
            else:
                observed_projected_signatures[action_name].add(_projected_action_signature(a, parameter_indices[action_name]))
        
        if include_non_applicable:
            for ground_actions in ground_actions_by_schema.values():
                #check applicability of projected signature by checking if it was observed as applicable for any action in the current state
                for a in ground_actions:
                    action_name = a.get_action().get_name()
                    projected_signature = _projected_action_signature(a, parameter_indices[action_name])
                    if action_name not in observed_projected_signatures or projected_signature not in observed_projected_signatures[action_name]:
                        out.append((s, a, s, False))
    
    
    return out


def _prune_empty_action_datasets(combined_dataset):
    """Remove empty per-problem buckets and actions without any remaining data."""
    if combined_dataset:
        log("Transition counts per instance before pruning:")
        for action_name, per_problem_data in combined_dataset.items():
            counts = [len(bucket) for bucket in per_problem_data]
            counts_str = ", ".join(f"instance_{idx}: {count}" for idx, count in enumerate(counts))
            log(f"Action {action_name}: {counts_str}")

    pruned = {}
    for action_name, per_problem_data in combined_dataset.items():
        non_empty = [bucket for bucket in per_problem_data if len(bucket) > 0]
        if non_empty:
            pruned[action_name] = non_empty
    return pruned

def optimal_general_policy_bfs(
    problem, 
    max_states,
    data_function,
    max_sampling_seconds_per_problem=None,
    ):
    ctx_opts = search.SearchContextOptions(search.LiftedOptions(search.LiftedKPKCOptions()))
    search_context = search.SearchContext.create(problem, ctx_opts)
    aag = search_context.get_applicable_action_generator()
    state_repo = search_context.get_state_repository()
    
    # Get initial state
    init, _ = state_repo.get_or_create_initial_state()
    q = deque([init])
    seen = {init}
    out = defaultdict(list)  # (state, action, successor, is_applicable)
    expanded = 0
    sampling_start_time = time.time()
    while q and expanded < max_states:
        if (
            max_sampling_seconds_per_problem is not None
            and max_sampling_seconds_per_problem > 0
            and (time.time() - sampling_start_time) >= max_sampling_seconds_per_problem
        ):
            log(f"Stopping optimal GP BFS early after {max_sampling_seconds_per_problem}s for problem (expanded states: {expanded}).")
            break
        t0=time.time()
        #change to pop for DFS and popleft for BFS
        s = q.pop()
        # Generate applicable actions
        actions = aag.generate_applicable_actions(s)
        data, state_is_goal = data_function(problem, s, actions, search_context)
        actions = [a for a in actions] 
        
        if state_is_goal:
            break
            # continue
            # log(f"Reached goal state, skipping further expansion. Expanded {expanded} states, queue size: {len(q)}, time for iteration: {time.time()-t0:.2f}s")
        actions.sort(key=lambda a: 1 if a in data[a.get_action().get_name()]["optimal_actions"] else 0)
        for a in actions:
            # Apply action to get successor state
            s2, _ = state_repo.get_or_create_successor_state(s, a, 0.0)
            
            if s2 not in seen:
                seen.add(s2)
                q.append(s2)
            action_name = a.get_action().get_name()
        
        
        expanded+=1
        for action_name in data:
            out[action_name].append(data[action_name])
        # log(f"Expanded state {expanded} in {time.time()-t0:.2f}s")
    return out



def optimal_general_policy_opt_then_bfs(
    problem, 
    max_states,
    data_function,
    max_sampling_seconds_per_problem=None,
    ):
    ctx_opts = search.SearchContextOptions(search.LiftedOptions(search.LiftedKPKCOptions()))
    search_context = search.SearchContext.create(problem, ctx_opts)
    aag = search_context.get_applicable_action_generator()
    state_repo = search_context.get_state_repository()
    
    # Get initial state
    init, _ = state_repo.get_or_create_initial_state()
    q = deque([init])
    seen = {init}
    out = defaultdict(list)  # (state, action, successor, is_applicable)
    goal_seen = False
    expanded = 0
    sampling_start_time = time.time()
    while q and expanded < max_states:
        if (
            max_sampling_seconds_per_problem is not None
            and max_sampling_seconds_per_problem > 0
            and (time.time() - sampling_start_time) >= max_sampling_seconds_per_problem
        ):
            log(f"Stopping optimal GP BFS early after {max_sampling_seconds_per_problem}s for problem (expanded states: {expanded}).")
            break
        t0=time.time()
        #change to pop for DFS and popleft for BFS
        if goal_seen:
            s = q.popleft()
        else:
            s = q.pop()
        # Generate applicable actions
        actions = aag.generate_applicable_actions(s)
        data, state_is_goal = data_function(problem, s, actions, search_context)
        if state_is_goal:
            goal_seen = True
            continue
        
        actions = [a for a in actions]
        actions.sort(key=lambda a: 1 if a in data[a.get_action().get_name()]["optimal_actions"] else 0)
        for a in actions:
            # Apply action to get successor state
            s2, _ = state_repo.get_or_create_successor_state(s, a, 0.0)
            
            if s2 not in seen:
                seen.add(s2)
                q.append(s2)
            action_name = a.get_action().get_name()
        
        expanded+=1
        for action_name in data:
            out[action_name].append(data[action_name])
        # log(f"Expanded state {expanded} in {time.time()-t0:.2f}s")
    return out


def generate_action_model_learning_dataset(
    domain_path,
    problem_path,
    parameter_list,
    full_applicability=False,
    predicate_indices=set(),
    max_states=100000,
    padding=0,
    include_non_applicable=False,
    target_mode="state",
    max_actions=5000,
    ignore_type_filtering=False,
    random_seed=None,
    max_transitions_per_action=5000,
    max_sampling_seconds_per_problem=None,
    ):
    if target_mode not in {"state", "applicability_vector"}:
        raise ValueError("target_mode must be one of: 'state', 'applicability_vector'")

    problem = formalism.Problem.create(domain_path, problem_path, formalism.ParserOptions())
    action_names = [a.get_name() for a in problem.get_domain().get_actions()]
    dataset = {action_name: [] for action_name in action_names}
    transitions = generate_action_transitions_bfs(
        problem,
        max_states=max_states,
        include_non_applicable=include_non_applicable,
        full_applicability=full_applicability,
        max_actions=max_actions,
        ignore_type_filtering=ignore_type_filtering,
        random_seed=random_seed,
        parameter_indices=parameter_list,
        max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
    )
    log(f"Generated {len(transitions)} transitions, processing into dataset...")

    if full_applicability:
        states = set([s for s, _, _, _ in transitions])
        for s in states:
            applicability_vector = get_full_applicability_data(s, problem, parameter_list, padding=padding)
            for action_name, vector in applicability_vector.items():
                s_concepts, s_roles, _, _ = mimir_state_to_concept_role_data(s, problem, padding=padding, predicate_indices=predicate_indices) 
                dataset[action_name].append((s_concepts, s_roles, vector[0], vector[1]))
        return dataset

    napp={}
    app={}
    concept_role_names = []
    for s, a, s2, is_applicable in transitions:
        action_name = a.get_action().get_name()
        if len(dataset[action_name])>max_transitions_per_action:
            continue
        if action_name not in napp:
            napp[action_name]=0
        if action_name not in app:
            app[action_name]=0
        if is_applicable:
            app[action_name]+=1
        else:
            napp[action_name]+=1
        if target_mode == "state":
            # target_state = s2 if is_applicable else s
            if parameter_list is None:
                parameter_indices = None
            else:
                parameter_indices = parameter_list[action_name]
            data_point, names = mimir_to_tensors.transition_to_concept_role_data(
                s,
                a,
                s2,
                problem,
                parameter_indices=parameter_indices,
                predicate_indices=predicate_indices,
                padding=padding,
            )
            concept_role_names=names
            
        else:
            data_point, names = state_action_to_applicability_data(
                s,
                a,
                problem,
                parameter_indices=parameter_list[action_name],
                predicate_indices=predicate_indices,
                is_applicable=is_applicable,
                padding=padding,
            )
            concept_role_names=names
        dataset[action_name].append(data_point)  
      
    for a in napp.keys():
        log(f"Action: {a}, Applicable: {app[a]}, Non-applicable: {napp[a]}, Names: {concept_role_names}")
    return dataset

def generate_action_model_from_multiple_problems(
    domain_path, 
    problem_paths, 
    parameter_list, 
    max_states=100000, 
    full_applicability=False,
    include_non_applicable=False, 
    target_mode="state", 
    max_actions=10000,
    ignore_type_filtering=True,
    predicate_indices=set(),
    max_sampling_seconds_per_problem=None,
):
    combined_dataset = {}
    p=0
    for i, problem_path in enumerate(problem_paths):
        dataset = generate_action_model_learning_dataset(
            domain_path,
            problem_path,
            parameter_list,
            max_states=max_states,
            include_non_applicable=include_non_applicable,
            full_applicability=full_applicability,
            target_mode=target_mode,
            max_actions=max_actions,
            ignore_type_filtering=False,
            predicate_indices=predicate_indices,
            max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
        )
        for action_name, data_points in dataset.items():
            if action_name not in combined_dataset:
                combined_dataset[action_name] = []
            combined_dataset[action_name].append(data_points)
    return _prune_empty_action_datasets(combined_dataset)

def generate_optimal_general_policy_dataset(
    args,
    domain_path,
    problem_path,
    parameter_indices=None,
    predicate_indices=set(),
    max_states=1000,
    max_sampling_seconds_per_problem=None,
):
    problem = formalism.Problem.create(domain_path, problem_path, formalism.ParserOptions())
    if not hasattr(args , "sampling_method"):
        args.sampling_method = "bfs"
    if args.sampling_method == "bfs":
        out = optimal_general_policy_bfs(
            problem,
            max_states,
            data_function= opt_gp_util.data_function,
            max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
        )
    elif args.sampling_method == "opt_then_bfs":
        out = optimal_general_policy_opt_then_bfs(
            problem,
            max_states,
            data_function= opt_gp_util.data_function,
            max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
        )
    log(f"Generation done")
    log(f"Collected {sum(len(v) for v in out.values())} data points across {len(out)} actions.")
    dataset = {}
    names = {}
    for action_name in out:
        concept_role_names = None
        dataset[action_name] = []
        for data_point in out[action_name]:
            data, names = opt_gp_util.to_tensor_data(data_point, problem, parameter_indices=parameter_indices, predicate_indices=predicate_indices)
            dataset[action_name].append(data)
            if concept_role_names is None:
                concept_role_names = names
            elif names!= concept_role_names:
                raise ValueError(f"Concept/role names mismatch across data points for action {action_name}: {names} vs {concept_role_names}")
        log(f"{len(dataset[action_name])} data points for action {action_name} with names {concept_role_names}")
        names[action_name] = concept_role_names
    return dataset, names

def generate_optimal_general_policy_dataset_from_multiple_problems(
    args,
    domain_path,
    problem_paths,
    parameter_indices=None,
    predicate_indices=set(),
    max_states=1000,
    max_sampling_seconds_per_problem=None):
    names_list = []
    combined_dataset = {}
    for problem_path in problem_paths:
        log(f"Generating optimal general policy dataset for problem: {problem_path}, max_states: {max_states}, max_sampling_seconds_per_problem: {max_sampling_seconds_per_problem}")
        dataset, names = generate_optimal_general_policy_dataset(
            args,
            domain_path,
            problem_path,
            parameter_indices=parameter_indices,
            predicate_indices=predicate_indices,
            max_states=max_states,
            max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
        )
        names_list.append(names)
        for action_name in dataset:
            if action_name not in combined_dataset:
                combined_dataset[action_name] = []
            combined_dataset[action_name].append(dataset[action_name])
    return _prune_empty_action_datasets(combined_dataset), names_list


def generate_general_policy_dataset(
    domain_path,
    problem_path,
    filter_function=None,
    parameter_indices=None,
    predicate_indices=set(),
    max_states=100000,
    max_sampling_seconds_per_problem=None,
    exclude_goal_states=True,
    precomputed_transitions=None,
    problem_override=None,
):
    problem = problem_override
    if precomputed_transitions is None:
        raise RuntimeError(
            "generate_general_policy_dataset requires precomputed transitions from SearchContext sampling. "
            "This guard prevents accidental full state-space generation paths."
        )
    if problem is None and len(precomputed_transitions) > 0:
        first_state = precomputed_transitions[0][0]
        if hasattr(first_state, "get_state_repository"):
            state_repository = first_state.get_state_repository()
            if hasattr(state_repository, "get_problem"):
                problem = state_repository.get_problem()
    if problem is None:
        problem = formalism.Problem.create(domain_path, problem_path, formalism.ParserOptions())
    action_names = [a.get_name() for a in problem.get_domain().get_actions()]
    dataset = {action_name: [] for action_name in action_names}
    log(f"Generating general policy dataset for problem: {problem_path}, max_states: {max_states}, max_sampling_seconds_per_problem: {max_sampling_seconds_per_problem}")
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
    transitions = precomputed_transitions
    log("Using precomputed transitions from SearchContext sampling")
    log(f"Generated {len(transitions)} transitions, processing into dataset...")

    following_transitions = 0
    skipped_goal_transitions = 0
    positive_labels_per_action = {action_name: 0 for action_name in action_names}
    evaluated_labels_per_action = {action_name: 0 for action_name in action_names}
    for s, a, s2, _ in transitions:
        if exclude_goal_states and _is_goal_state(s):
            skipped_goal_transitions += 1
            continue

        action_name = a.get_action().get_name()
        evaluated_labels_per_action[action_name] += 1
        label = True if filter_function is None else filter_function(s, s2)
        if label:
            following_transitions += 1
            positive_labels_per_action[action_name] += 1

        parameter_indices_for_action = parameter_indices[action_name] if parameter_indices is not None else None
        data_point = state_pair_to_concept_role_data(
            s,
            s2,
            problem,
            parameter_indices=parameter_indices_for_action,
            label=label,
            predicate_indices=predicate_indices,
            padding=0,
        )
        dataset[action_name].append(data_point)

    log(
        f"Total transitions: {len(transitions)}, "
        f"Transitions following policy/optimality: {following_transitions}, "
        f"Skipped from goal states: {skipped_goal_transitions}"
    )
    log("Label=1 transitions per action:")
    for action_name in action_names:
        evaluated = evaluated_labels_per_action[action_name]
        positive = positive_labels_per_action[action_name]
        ratio = (positive / evaluated) if evaluated > 0 else 0.0
        log(f"  {action_name}: {positive} / {evaluated} ({ratio:.2%})")
    return dataset

def create_general_policy_for_multiple_problems(
    domain_path, 
    problem_paths, 
    parameter_indices,
    domain_name,
    max_states=100000,
    max_sampling_seconds_per_problem=None,
):
    combined_dataset = {}
    for problem_path in problem_paths:
        runtime = gps.create_runtime(
            domain=domain_path,
            problems=[problem_path],
            domain_name=domain_name,
        )
        filter_function = lambda s, s2: gps.transition_follows_policy(runtime.policy, s, s2, runtime.denotation_repos)
        log("filter function created")
        search_context = runtime.ctx.get_search_contexts()[0]
        aag = search_context.get_applicable_action_generator()
        state_repo = search_context.get_state_repository()
        init, _ = state_repo.get_or_create_initial_state()
        q = deque([init])
        seen = {init}
        transitions = []
        expanded_states = 0
        sampling_start_time = time.time()
        while q and expanded_states < max_states:
            if (
                max_sampling_seconds_per_problem is not None
                and max_sampling_seconds_per_problem > 0
                and (time.time() - sampling_start_time) >= max_sampling_seconds_per_problem
            ):
                log(
                    f"Stopping SearchContext sampling early after {max_sampling_seconds_per_problem}s "
                    f"(expanded states: {expanded_states}, transitions: {len(transitions)})."
                )
                break

            s = q.popleft()
            actions = aag.generate_applicable_actions(s)
            for a in actions:
                s2, _ = state_repo.get_or_create_successor_state(s, a, 0.0)
                transitions.append((s, a, s2, True))
                if s2 not in seen:
                    seen.add(s2)
                    q.append(s2)
            expanded_states += 1
        log(f"Generating dataset for problem: {problem_path}")
        dataset = generate_general_policy_dataset(
            domain_path,
            problem_path,
            filter_function=filter_function,
            parameter_indices=parameter_indices,
            max_states=max_states,
            max_sampling_seconds_per_problem=max_sampling_seconds_per_problem,
            precomputed_transitions=transitions,
            problem_override=search_context.get_problem(),
        )
        log(f"Dataset generated for problem: {problem_path}, combining into overall dataset...")
        for action_name, data_points in dataset.items():
            if action_name not in combined_dataset:
                combined_dataset[action_name] = []
            combined_dataset[action_name].append(data_points)
    return _prune_empty_action_datasets(combined_dataset)
    

def test_model_on_test_problems(domain_path, test_instance_paths, models_per_action, parameter_indices, max_steps, soft_policy=False, model_type="state_pair", threshold=0.5, one_step_cycle_check=True):
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared_models = {}
    for name, model in models_per_action.items():
        model = model.to(device)
        model.eval()
        prepared_models[name] = model

    for test_instance_path in test_instance_paths:
        problem = formalism.Problem.create(domain_path, test_instance_path, formalism.ParserOptions())
        ctx_opts = search.SearchContextOptions(search.LiftedOptions(search.LiftedKPKCOptions()))
        search_context = search.SearchContext.create(problem, ctx_opts)
        # Get initial state
        state_repo = search_context.get_state_repository()

        aag = search.KPKCLiftedApplicableActionGenerator.create(problem, search.LiftedKPKCOptions())
        # state_repo = search.StateRepository.create(axiom_eval)
        init, _ = state_repo.get_or_create_initial_state()
        # opt_dist = opt_gp_util.goal_distance(init, problem, search_context)
        # log(f"Optimal distance to goal for problem {test_instance_path}: {opt_dist}")
        # continue
        opt_dist = 0
        
        # ctx_opts = search.SearchContextOptions(search.LiftedOptions(search.LiftedKPKCOptions()))
        # search_context = search.SearchContext.create(problem, ctx_opts)
        # aag = search_context.get_applicable_action_generator()
        # state_repo = search_context.get_state_repository()
        # init, _ = state_repo.get_or_create_initial_state()
        current_state = init
        steps_taken = 0
        goal_reached = False
        log(f"Testing on problem: {str(test_instance_path).split('/')[-1]} with max_steps: {max_steps}")
        seen = set()
        seen.add(current_state)
        res="max_steps"
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
            log("Initial state is already a goal state.")
            results[test_instance_path] = {
                "final_state": current_state,
                "steps_taken": steps_taken,
                "goal_reached": True,
                "reason": "initial_goal",
                "optimal_steps": opt_dist
            }
            continue
        if _is_goal_state(init):
            log("Initial state is already a goal state.")
            results[test_instance_path] = {
                "final_state": init,
                "steps_taken": steps_taken,
                "goal_reached": True,
                "reason": "initial_goal",
                "optimal_steps": opt_dist
            }
            continue
        while steps_taken < max_steps:
            actions = aag.generate_applicable_actions(current_state)
            if not actions:
                log("No applicable actions available, stopping execution.")
                break
            action_scores = {}
            # for a in actions:
            #     if a.get_action().get_name() =="pick":
            #         for i, obj in enumerate(a.get_objects()):
            #             print(f"Object: {obj.get_name()}, Index: {obj.get_index()}, Position in action: {i}")
            preds_per_action_name = {}
            with torch.inference_mode():
                for name in prepared_models:
                    if model_type == "state_pair_nullary" or model_type == "state_full":
                        concepts, roles, _, _ = mimir_to_tensors.state_goal_to_concept_role_data(
                            current_state,
                            problem,
                            name,
                            parameter_indices=parameter_indices,
                        )
                    if model_type == "state_action_nullary" or model_type == "state_action_full":
                        concepts, roles, _, _ = mimir_to_tensors.state_goal_to_concept_role_data(
                            current_state,
                            problem,
                            name,
                            parameter_indices=parameter_indices,
                        )
                    num_object_slots = concepts.shape[-1]
                    model = prepared_models[name]
                    concepts = concepts.to(device, non_blocking=True)
                    roles = roles.to(device, non_blocking=True)
                    out_concepts, out_roles = model(concepts, roles)
                    preds_per_action_name[name] = (
                        out_concepts.detach().cpu(),
                        out_roles.detach().cpu(),
                        num_object_slots,
                    )
            for a in actions:
                # s2 = state_repo.get_or_create_successor_state(current_state, a, 0.0)[0]
                action_name = a.get_action().get_name()
                practical_action_arity = len(a.get_objects()) if parameter_indices is None or parameter_indices[action_name] is None else len(parameter_indices[action_name])
                parameter_indices_action = parameter_indices[action_name] if parameter_indices is not None and parameter_indices[action_name] is not None else list(range(practical_action_arity))
                action_object_indices = tuple(obj.get_index() for obj in a.get_objects())
                out_concepts, out_roles, num_object_slots = preds_per_action_name[action_name]
                if model_type == "state_pair_nullary" or model_type == "state_action_nullary":
                    action_scores[a] = out_concepts.sum().item()  # Example scoring function
                elif model_type == "state_full" or model_type == "state_action_full":
                    if practical_action_arity == 0:
                        action_scores[a] = out_concepts.sum().item() / num_object_slots
                    if practical_action_arity == 1:
                        action_scores[a] = out_concepts[0, 0, action_object_indices[parameter_indices_action[0]]].item()
                    elif practical_action_arity == 2:
                        action_scores[a] = out_roles[0, 0,  action_object_indices[parameter_indices_action[0]], action_object_indices[parameter_indices_action[1]]].item()
            all_scores = action_scores.copy()
            if threshold is not None:
                action_scores = {a: score for a, score in action_scores.items() if score >= threshold}
            if not action_scores:
                max_score = max(all_scores.values(), default=float('-inf'))
                log(f"No valid actions available with score above threshold, stopping execution. Max score is {max_score}.")
                res = "no-action"
                goal_reached = False
                break
            if soft_policy:
                action_names = list(action_scores.keys())
                chosen_action = np.random.choice(action_names)
                chosen_score = action_scores[chosen_action]
                chosen_successor, _ = state_repo.get_or_create_successor_state(current_state, chosen_action, 0.0)
                if one_step_cycle_check and chosen_successor in seen:
                    log("Sampled action leads to a previously seen state, stopping execution to avoid loops.")
                    res = "cycle"
                    goal_reached = False
                    break
            else:
                ranked_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
                chosen_action = None
                chosen_score = None
                chosen_successor = None

                if one_step_cycle_check:
                    for candidate_action, candidate_score in ranked_actions:
                        successor_state, _ = state_repo.get_or_create_successor_state(current_state, candidate_action, 0.0)
                        if successor_state not in seen:
                            chosen_action = candidate_action
                            chosen_score = candidate_score
                            chosen_successor = successor_state
                            break
                        log(f"Action {candidate_action.get_action().get_name()} leads to a previously seen state, skipping.")

                    if chosen_action is None:
                        log("All valid actions lead to previously seen states, stopping execution to avoid a cycle.")
                        res = "cycle"
                        goal_reached = False
                        break
                else:
                    chosen_action, chosen_score = ranked_actions[0]
                    chosen_successor, _ = state_repo.get_or_create_successor_state(current_state, chosen_action, 0.0)
                    if chosen_successor in seen:
                        log("Encountered a previously seen state, stopping execution to avoid loops.")
                        res= "cycle"
                        goal_reached = False
                        break

            log(
                f"Chosen action: {chosen_action.get_action().get_name()}, "
                f"Num valid Actions:{len(action_scores):.2f}, Score: {chosen_score}"
            )
            current_state = chosen_successor
            seen.add(current_state)
            steps_taken += 1
            # Check if goal is reached
                
            goal_reached = _is_goal_state(current_state)
            if goal_reached:
                res="success"
                break
        results[test_instance_path] = {
            "final_state": current_state,
            "steps_taken": steps_taken,
            "goal_reached": goal_reached,
            "reason": res,
            "optimal_steps": opt_dist
        }
        log(f"Finished testing on problem: {str(test_instance_path).split('/')[-1]}, Steps taken: {steps_taken}, Goal reached: {goal_reached}, Reason: {res}")
    return results


if __name__ == "__main__":
    import sys

    def parse_bool(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
      
    # Default to logistics domain if no args provided
    # domain_path = sys.argv[1] if len(sys.argv) > 1 else "data/clear/domain.pddl"
    # problem_path = sys.argv[2] if len(sys.argv) > 2 else "data/clear/test/_training_clear_5.pddl"
    domain_path = "random_trace_generator/domain.pddl"
    problem_path = "random_trace_generator/test_instance.pddl"
    parameter_file_key = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(problem_path))[0]
    max_states = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    include_non_applicable = parse_bool(sys.argv[5]) if len(sys.argv) > 5 else True
    target_mode = sys.argv[6] if len(sys.argv) > 6 else "state"
    max_actions = int(sys.argv[7]) if len(sys.argv) > 7 else 5000
    ignore_type_filtering = parse_bool(sys.argv[8]) if len(sys.argv) > 8 else False
    random_seed = int(sys.argv[9]) if len(sys.argv) > 9 else None
    

    print(f"Domain:        {domain_path}")
    print(f"Problem:       {problem_path}")
    print(f"Parameter key: {parameter_file_key}")
    print(f"Max states:    {max_states}")
    print(f"Include non-applicable: {include_non_applicable}")
    print(f"Target mode:   {target_mode}")
    print(f"Max actions/schema: {max_actions}")
    print(f"Ignore type filtering: {ignore_type_filtering}")
    print(f"Random seed: {random_seed}")
    print("Generating transitions...")

    parameter_list = load_parameter_indices("clear")
    dataset = generate_optimal_general_policy_dataset(
        domain_path,
        problem_path,
        parameter_indices=parameter_list,
        max_states=5,
    )
    
    print("Generation complete!")

    for action_name, transitions in dataset.items():
        print(f"Action: {action_name}, Transitions: {len(transitions)}")
        print(transitions[0])
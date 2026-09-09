import torch
import pymimir.advanced.formalism as formalism
import pymimir.advanced.search as search
import pymimir.advanced.datasets as datasets


def mimir_state_to_concept_role_data(state, problem, only_fluid=False, padding=0, predicate_indices=set()):
    out_roles = []
    out_concepts = []
    concept_p = []
    role_p = []
    domain = problem.get_domain()

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

    repositories = problem.get_repositories()
    if hasattr(state, "get_atoms"):
        atoms = state.get_atoms(ignore_static=only_fluid, ignore_fluent=False, ignore_derived=True)
    elif hasattr(state, "get_fluent_atoms"):
        fa = list(state.get_fluent_atoms())
        sa = list(problem.get_static_initial_atoms())
        if only_fluid:
            sa = []
        atoms = list(repositories.get_fluent_ground_atoms_from_indices(fa)) + sa
    else:
        raise TypeError("Unsupported state object: expected get_atoms or get_fluent_atoms")
    
    num_objects=len(problem.get_objects())

    #initialize empty tensors
    concepts_tensor = torch.zeros((len(concept_p), num_objects+padding), dtype=torch.float32)
    roles_tensor = torch.zeros((len(role_p)+1, num_objects+padding, num_objects+padding), dtype=torch.float32)
    
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
    for i in range(num_objects):
        roles_tensor[-1, i, i] = 1.0  # add identity-role
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
            print("Ignoring negative atom in goal condition (not supported in current setup):", atom)
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

    # print(f"in_concept_names: {c_n}, in_role_names: {r_n}", flush=True)
    # print(f"out_concept_names: {c2_n}, out_role_names: {r2_n}", flush=True)
    # print(f"parameter_shape: {a_c.shape}", flush=True)
    out_c_names = [f"{name}_add" for name in c2_n] + [f"{name}_del" for name in c2_n] 
    out_r_names = [f"{name}_add" for name in r2_n] + [f"{name}_del" for name in r2_n] 
    return (in_c, in_r,out_c, out_r), (c_n, r_n, a_c_n, out_c_names, out_r_names)


def state_action_to_applicability_data(s, a, problem, parameter_indices, is_applicable, padding=0, predicate_indices=set()):
    c, r, c_n, r_n = mimir_state_to_concept_role_data(s, problem, padding=padding, predicate_indices=predicate_indices)
    a_c, a_c_n = action_parameters_to_concept_role_data(a, problem, parameter_indices, padding=padding)
    in_c = torch.cat([c, a_c], dim=0)
    in_r = r
    num_objects = len(problem.get_objects())
    out_concepts = torch.ones((1,num_objects+padding), dtype=torch.float32) if is_applicable else torch.zeros((1,num_objects+padding), dtype=torch.float32)
    out_roles = torch.zeros((0, num_objects+padding, num_objects+padding), dtype=torch.float32)  # no role is predicted
    return (in_c, in_r, out_concepts, out_roles)

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

def state_action_goal_to_concept_role_data(s, a, problem, parameter_indices, label,  predicate_indices=set(), padding=0):
    c, r, c_n, r_n = mimir_state_to_concept_role_data(s, problem, padding=padding, predicate_indices=predicate_indices)
    a_c, a_c_n = action_parameters_to_concept_role_data(a, problem, parameter_indices, padding=padding)
    goal_c, goal_r, _, _ = goal_condition_to_concept_role_data(problem, padding=padding, predicate_indices=predicate_indices)
    in_c = torch.cat([c, a_c, goal_c], dim=0)
    in_r = torch.cat([r, goal_r], dim=0)
    num_objects = len(problem.get_objects())
    out_concepts = torch.ones((1,num_objects+padding), dtype=torch.float32) if label else torch.zeros((1,num_objects+padding), dtype=torch.float32)
    out_roles = torch.zeros((0, num_objects+padding, num_objects+padding), dtype=torch.float32)  # no role is predicted
    return (in_c, in_r, out_concepts, out_roles)

def state_goal_to_concept_role_data(s, problem, action_name, predicate_indices=set(), parameter_indices=None, padding=0):
    c, r, c_n, r_n = mimir_state_to_concept_role_data(s, problem, padding=padding, predicate_indices=predicate_indices)
    goal_c, goal_r, g_c_n, g_r_c = goal_condition_to_concept_role_data(problem, padding=padding, predicate_indices=predicate_indices)
    action_applicability_all = get_full_applicability_data(s, problem, parameter_list=parameter_indices, padding=padding)
    action_applicability = action_applicability_all[action_name]
    in_c = torch.cat([c, action_applicability[0], goal_c], dim=0)
    in_r = torch.cat([r, action_applicability[1], goal_r], dim=0)
    in_c = in_c.unsqueeze(0)  # Add batch dimension
    in_r = in_r.unsqueeze(0)  # Add batch dimension
    num_objects = len(problem.get_objects())
   
    return (in_c, in_r, None, None)

def get_full_applicability_data(s, problem, parameter_list, padding=0):
    agg = search.KPKCLiftedApplicableActionGenerator.create(problem, search.LiftedKPKCOptions())
    actions = agg.generate_applicable_actions(s)
    by_action_vector = {} 
    num_objects = len(problem.get_objects())
    for a in problem.get_domain().get_actions():
        action_name = a.get_name()
        if parameter_list is None or parameter_list[action_name] is None:
            parameter_indices = list(range(a.get_arity()))
        else: 
            parameter_indices = parameter_list[action_name]
        if len(parameter_indices)==1 or len(parameter_indices)==0:            
            by_action_vector[action_name] = [torch.zeros((1, num_objects+padding), dtype=torch.float32), torch.zeros((0, num_objects+padding, num_objects+padding), dtype=torch.float32)]  
        elif len(parameter_indices)==2:
            by_action_vector[action_name] = [torch.zeros((0, num_objects+padding), dtype=torch.float32), torch.zeros((1, num_objects+padding, num_objects+padding), dtype=torch.float32)]

    for a in actions:
        action_name = a.get_action().get_name()
        if parameter_list is None or parameter_list[action_name] is None:
            parameter_indices = list(range(len(a.get_objects())))
        else: 
            parameter_indices = parameter_list[action_name]
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

def optimal_actions_to_concept_role_data(optimal_actions, problem, parameter_indices, padding=0, all_actions=None):
    domain = problem.get_domain()
    repositories = problem.get_repositories()
    num_objects = len(problem.get_objects())
    concept_p = torch.zeros((0, num_objects+padding), dtype=torch.float32)
    role_p = torch.zeros((0, num_objects+padding, num_objects+padding), dtype=torch.float32)
    
    a0= all_actions[0]
    if  parameter_indices is None or parameter_indices[a0.get_action().get_name()] is None:
        parameter_indices = list(range(len(a0.get_objects())))
    else: 
        parameter_indices = parameter_indices[a0.get_action().get_name()]

    arity = len(parameter_indices)

    if arity == 0:
        concept_p=torch.zeros((1, num_objects+padding), dtype=torch.float32)
    elif arity == 1:
        concept_p=torch.zeros((1, num_objects+padding), dtype=torch.float32)
    elif arity == 2:
        role_p=torch.zeros((1, num_objects+padding, num_objects+padding), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported action arity: {arity}")
    for a in optimal_actions:
        action_name = a.get_action().get_name()
        param = a.get_objects()
        if arity == 0:
            concept_p[0,:] = 1.0
        elif arity == 1:
            concept_p[0, param[parameter_indices[0]].get_index()] = 1.0
        elif arity == 2:
            role_p[0, param[parameter_indices[0]].get_index(), param[parameter_indices[1]].get_index()] = 1.0
    return concept_p, role_p
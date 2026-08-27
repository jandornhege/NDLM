import pymimir.advanced.search as adv_search
import pymimir.advanced.formalism as adv_formalism
import os
import re
import shutil
import subprocess
import tempfile
try: 
    from random_trace_generator.mimir_to_tensors import *
except:
    from mimir_to_tensors import *
from collections import defaultdict

# Cache for goal distances to avoid redundant A* searches.
_goal_distance_cache = {}

def goal_distance(state, problem, search_context):
    cache_key = (id(search_context), state)
    if cache_key in _goal_distance_cache:
        return _goal_distance_cache[cache_key]
    
    grounder = adv_search.LiftedGrounder(problem)
    heuristic = adv_search.MaxHeuristic.create(grounder)    
    # heuristic = adv_search.FFHeuristic.create(grounder)    
    options = adv_search.AStarLazyOptions()
    options.start_state = state
    result = adv_search.find_solution_astar_lazy(search_context, heuristic, options)
    distance = result.plan.get_cost() if result.plan is not None else float("inf")
    
    _goal_distance_cache[cache_key] = distance
    return distance


def identify_optimal_actions(state, problem, actions, search_context):
    optimal_actions = []
    # Use the Mimir-based distance for arbitrary in-memory states.
    state_cost = goal_distance(state, problem, search_context)
    state_is_goal = state_cost == 0
    if state_is_goal:
        return optimal_actions, state_is_goal
    state_repo = search_context.get_state_repository()
    for action in actions:
        next_state, _ = state_repo.get_or_create_successor_state(state, action, 0.0)
        next_cost = goal_distance(next_state, problem, search_context)
        if next_cost == state_cost - 1:
            optimal_actions.append(action)
    return optimal_actions, state_is_goal




# 
def data_function(problem, state, actions, search_context):
    optimal_actions, state_is_goal = identify_optimal_actions(state, problem, actions, search_context)
    if len(optimal_actions) == 0 and not state_is_goal:
        raise ValueError("No optimal actions found for a non-goal state. This should not happen.")
    if state_is_goal:
        return None, state_is_goal
    actions_grouped = defaultdict(list)
    optimal_actions_grouped = defaultdict(list)
    for action in actions:
        actions_grouped[action.get_action().get_name()].append(action)
        if action in optimal_actions:
            optimal_actions_grouped[action.get_action().get_name()].append(action)
    data = {}
    for action_name in actions_grouped:
        data[action_name] = {
            "state": state,
            "actions": actions_grouped[action_name],
            "optimal_actions": optimal_actions_grouped[action_name]
        }
    return data, state_is_goal

def to_tensor_data(data_point, problem, parameter_indices=None, predicate_indices=None):
    state = data_point["state"]
    actions = data_point["actions"]
    optimal_actions = data_point["optimal_actions"]
    
    action_name = actions[0].get_action().get_name()
    
    c, r, c_n, r_n = mimir_state_to_concept_role_data(state, problem, predicate_indices=predicate_indices)
    state_tensors = (c, r)
    
    applicability_all = get_full_applicability_data(state, problem, parameter_list=parameter_indices)
    
    action_applicability = applicability_all[action_name]
    if action_applicability[0].shape[0] != 0:
        app_c_n = "applicability"
        app_r_n = None
    if action_applicability[1].shape[0] != 0:
        app_r_n = "applicability"
        app_c_n = None
    g_c, g_r, g_c_n, g_r_n = goal_condition_to_concept_role_data(problem, predicate_indices=predicate_indices)
    goal_tensors = (g_c, g_r)
    in_c = torch.cat([state_tensors[0], action_applicability[0], goal_tensors[0]], dim=0) 
    in_r = torch.cat([state_tensors[1], action_applicability[1], goal_tensors[1]], dim=0)
    
    out_names = {
        "state_c": c_n,
        "state_r": r_n,
        "app_c": app_c_n,
        "app_r": app_r_n,
        "goal_c": g_c_n,
        "goal_r": g_r_n
    }

    out_c, out_r = optimal_actions_to_concept_role_data(optimal_actions, problem,  parameter_indices=parameter_indices, all_actions=actions)
    return (in_c, in_r, out_c, out_r), out_names

if __name__ == "__main__":
    problem = adv_formalism.Problem.create("domain.pddl", "test_instance.pddl", adv_formalism.ParserOptions())
    ctx_opts = adv_search.SearchContextOptions(adv_search.LiftedOptions(adv_search.LiftedKPKCOptions()))
    search_context = adv_search.SearchContext.create(problem, ctx_opts)

    # --- initial state ---
    state, _ = search_context.get_state_repository().get_or_create_initial_state()

    # --- test goal distance ---
    dist = goal_distance(state, problem, search_context)

    print("Goal distance from initial state:", dist)
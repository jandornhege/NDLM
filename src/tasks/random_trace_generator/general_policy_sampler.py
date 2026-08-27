from pathlib import Path
from dataclasses import dataclass

import pymimir.advanced.search as search
import pymimir.advanced.languages.description_logics as dl
import pymimir.advanced.languages.general_policies as gp

@dataclass
class GeneralPolicyRuntime:
    domain: Path
    problems: list[Path]
    ctx: object
    policy: object
    gp_repos: object
    dl_repos: object
    denotation_repos: object


def create_delivery_runtime(
    domain=None,
    problems=None,
    ctx=None,
    gp_repos=None,
    dl_repos=None,
    denotation_repos=None,
):
    if ctx is None:
        if domain is None or problems is None:
            raise ValueError("Either provide ctx or both domain and problems.")
        domain = Path(domain)
        problems = [Path(problem) for problem in problems]
        ctx = search.GeneralizedSearchContext.create(
            domain,
            problems,
            search.SearchContextOptions(search.LiftedOptions()),
        )
    else:
        domain = Path(domain) if domain is not None else Path(".")
        problems = [Path(problem) for problem in problems] if problems is not None else []

    gp_repos = gp_repos if gp_repos is not None else gp.Repositories()
    dl_repos = dl_repos if dl_repos is not None else dl.Repositories()
    denotation_repos = denotation_repos if denotation_repos is not None else dl.DenotationRepositories()

    policy = gp.GeneralPolicyFactory.get_or_create_general_policy_delivery(
        ctx.get_generalized_problem().get_domain(),
        gp_repos,
        dl_repos,
    )

    return GeneralPolicyRuntime(
        domain=domain,
        problems=problems,
        ctx=ctx,
        policy=policy,
        gp_repos=gp_repos,
        dl_repos=dl_repos,
        denotation_repos=denotation_repos,
    )

def create_runtime(
    domain=None,
    problems=None,
    domain_name=None,
    ctx=None,
    gp_repos=None,
    dl_repos=None,
    denotation_repos=None,
):
    if ctx is None:
        if domain is None or problems is None:
            raise ValueError("Either provide ctx or both domain and problems.")
        domain = Path(domain)
        problems = [Path(problem) for problem in problems]
        ctx = search.GeneralizedSearchContext.create(
            domain,
            problems,
            search.SearchContextOptions(search.LiftedOptions()),
        )
    else:
        domain = Path(domain) if domain is not None else Path(".")
        problems = [Path(problem) for problem in problems] if problems is not None else []
    gp_repos = gp_repos if gp_repos is not None else gp.Repositories()
    dl_repos = dl_repos if dl_repos is not None else dl.Repositories()
    denotation_repos = denotation_repos if denotation_repos is not None else dl.DenotationRepositories()
    domain_obj = ctx.get_generalized_problem().get_domain()
    if domain_name == "delivery":
        policy = gp.GeneralPolicyFactory.get_or_create_general_policy_delivery(
            domain_obj,
            gp_repos,
            dl_repos,
        )
    elif domain_name == "logistics":
        policy = gp.GeneralPolicyFactory.get_or_create_general_policy_logistics(
            domain_obj,
            gp_repos,
            dl_repos,
        )
    elif domain_name in ["blocks3ops", "blocksworld"]:
        print("Creating general policy for blocks3ops domain")
        policy = gp.GeneralPolicyFactory.get_or_create_general_policy_blocks3ops(
            domain_obj,
            gp_repos,
            dl_repos,
        )
        print("General policy created for blocks3ops domain")
    elif domain_name == "spanner":
        policy = gp.GeneralPolicyFactory.get_or_create_general_policy_spanner(
            domain_obj,
            gp_repos,
            dl_repos,
        )
    elif domain_name == "miconic":
        try:
            from random_trace_generator.custom_policies import MICONIC_POLICY_DESCRIPTION
        except ImportError:
            from custom_policies import MICONIC_POLICY_DESCRIPTION   
        policy = gp_repos.get_or_create_general_policy(
            MICONIC_POLICY_DESCRIPTION,
            domain_obj,
            dl_repos,
        )
    elif domain_name == "ferry":
        try:
            from random_trace_generator.custom_policies import FERRY_POLICY_DESCRIPTION
        except ImportError:
            from custom_policies import FERRY_POLICY_DESCRIPTION
        policy = gp_repos.get_or_create_general_policy(
            FERRY_POLICY_DESCRIPTION,
            domain_obj,
            dl_repos,
        )
    else:
        raise ValueError(f"Unsupported domain name: {domain_name}")
    return GeneralPolicyRuntime(
        domain=domain,
        problems=problems,
        ctx=ctx,
        policy=policy,
        gp_repos=gp_repos,
        dl_repos=dl_repos,
        denotation_repos=denotation_repos,
    )


def create_delivery_setup(domain, problems):
    runtime = create_delivery_runtime(domain=domain, problems=problems)
    return runtime.policy, runtime.gp_repos, runtime.denotation_repos


def get_problem_state_space(runtime, problem_index):
    del runtime, problem_index
    raise RuntimeError("State-space access via runtime has been removed. Use BFS transition sampling paths instead.")


def transition_follows_policy(policy, source_state, target_state, denotation_repos):
    source_context = dl.EvaluationContext(source_state, denotation_repos)
    target_context = dl.EvaluationContext(target_state, denotation_repos)
    return policy.evaluate(source_context, target_context)


def main():
    domain = Path("/work/rleap1/jan.dornhege/MA/data/delivery/domain.pddl")

    problems = [
        Path("/work/rleap1/jan.dornhege/MA/data/delivery/test/instance_3_3_0.pddl"),
        Path("/work/rleap1/jan.dornhege/MA/data/delivery/train/instance_4_2_0.pddl"),
        Path("/work/rleap1/jan.dornhege/MA/data/delivery/test/instance_5_3_0.pddl"),
        Path("/work/rleap1/jan.dornhege/MA/data/delivery/test/instance_7_2_0.pddl")
    ]
    test_problem_path = "/work/rleap1/jan.dornhege/MA/data/delivery/test/instance_5_3_0.pddl"
    test_problem_path_2 = "/work/rleap1/jan.dornhege/MA/data/delivery/test/instance_7_2_0.pddl"
    # policy, gp_repos, denotation_repos, kb = create_delivery_setup(domain, problems)
    # print("terminating:", policy.is_terminating(gp_repos))
    

    # state_space = kb.get_state_spaces()[0]
    # graph = state_space.get_graph()
    # initial_vertex_index = state_space.get_initial_vertex()
    # initial_vertex = graph.get_vertex(initial_vertex_index)
    # initial_state = datasets.get_state(initial_vertex)

    # sampler = datasets.StateSpaceSampler(state_space)
    # transitions = list(sampler.get_forward_transitions(initial_state))
    # if not transitions:
    #     print("No forward transitions found from initial state.")
    #     return

    # action, successor_state = transitions[0]

    # follows_direct = transition_follows_policy(policy, initial_state, successor_state, denotation_repos)
    # print("direct transition check:", follows_direct)

    # follows_pair, _ = state_action_follows_policy(
    #     policy,
    #     state_space,
    #     initial_state,
    #     action,
    #     denotation_repos,
    # )
    # print("state-action pair check:", follows_pair)


if __name__ == "__main__":
    main()

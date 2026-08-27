import resource
import sys
import time
from pathlib import Path
import argparse
import torch

from ndlm.configs import config_object
from ndlm.modules import MultiLayerNDLM
import ndlm.main as NDLM_main
import owl_parser.benchmark_owl_ndlm as benchmark_owl_ndlm
import os

from my_logging import log, init_logger

data = {
    "NTN": "src/tasks/owl_parser/ntn_entities.pt",
    "Biopax": "src/data/Ontolearn/KGs/Biopax/biopax.entities.pt",
    "Family": "src/data/Ontolearn/KGs/Family/family-benchmark_rich_background.entities.pt",
    "Lymphography": "src/data/Ontolearn/KGs/Lymphography/lymphography.entities.pt"
}

lp_files = {
    "Biopax": "src/data/Ontolearn/LPs/Biopax/lps.json",
    "Family": "src/data/Ontolearn/LPs/Family/lps_difficult.json",
    "Lymphography": "src/data/Ontolearn/LPs/Lymphography/lps.json"
}


name =  "Biopax"  # "NTN"  # "Biopax"  # "Family"  # "Lymphography"
tensor_path = data[name]

lp_path = Path(lp_files[name])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
payload = torch.load(Path(tensor_path), map_location="cpu", weights_only=True)
concepts = payload["concepts"].unsqueeze(0).to(device)
roles = payload["roles"].unsqueeze(0).to(device)
# print(payload.keys())
# print(payload["concepts"].shape)
# print(payload["roles"].shape)
# print(payload["concept_names"])
# print(payload["role_names"])
# print(payload["concept_index"])
# print(payload["role_index"])


target = benchmark_owl_ndlm.load_lp_target(payload, lp_path)
for i, t in enumerate(target):
    # print(t["mask"].shape)
    # print(t["target"].shape)

    c_target = t["target"].unsqueeze(0).unsqueeze(0).to(device)
    r_target = torch.zeros((1,0, roles.shape[2], roles.shape[2]), dtype=torch.float32).to(device)
    c_mask = t["mask"].unsqueeze(0).unsqueeze(0).to(device)
    r_mask = torch.zeros((1,0, roles.shape[2], roles.shape[2]), dtype=torch.bool).to(device)
    # print(c_target.shape)
    config = config_object()
    config.NUM_LAYERS = 4
    config.NUM_HIDDEN_CONCEPTS = 5
    config.NUM_HIDDEN_ROLES = 5
    config.MODE = "strict"
    config.TRANSITIVE_CLOSURE = False
    config.INITIAL_FFN = True

    args=  argparse.Namespace(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=1,
            loss_type = "BCE",
            num_epochs = 100,
            test_interval = 100,
            weighted_loss = False,
            experiment_path = Path(f"outputs/CONCEPT_LEARNER/benchmark_owl_ndlm/{name}_{i}_{time.time()}/"),
        )
    os.makedirs(args.experiment_path, exist_ok=True)
    init_logger(f"{args.experiment_path}/output.log")
    log(f"positive examples: {t['num_positive']}")
    log(f"negative examples: {t['num_negative']}")
    log(f"concept_shape: {concepts.shape}")
    log(f"role_shape: {roles.shape}")
    model = MultiLayerNDLM(
        concepts.shape[1], roles.shape[1], 1, 0, config
    ).to(device).eval()
    train = [(concepts, roles, c_target, r_target, c_mask, r_mask) ]
    NDLM_main.main(train, train, config, args, model, checkpoint_path=args.experiment_path / f"checkpoints" if args.experiment_path else None, log=log)


# torch.cuda.reset_peak_memory_stats(device)
# torch.cuda.synchronize(device)
# rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
# start = time.perf_counter()
# with torch.inference_mode():
#     output_concepts, output_roles = model(concepts, roles)
# torch.cuda.synchronize(device)
# elapsed = time.perf_counter() - start
# rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
# peak_cuda = torch.cuda.max_memory_allocated(device) / 2**20
# peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

# print(f"=== {name} ===")
# print(f"objects: {concepts.shape[2]}")
# print(f"raw concepts/roles: {concepts.shape[1]}/{roles.shape[1]}")
# print("mode: strict")
# print("projection: 10/10")
# print(f"output concepts: {tuple(output_concepts.shape)}")
# print(f"output roles: {tuple(output_roles.shape)}")
# print(f"inference time: {elapsed:.6f} s")
# print(f"CPU RSS before/after/peak: {rss_before:.1f}/{rss_after:.1f}/{peak_rss:.1f} MiB")
# print(f"CUDA peak allocated: {peak_cuda:.1f} MiB")

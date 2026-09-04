import resource
import sys
import time
from pathlib import Path
import argparse
import torch
from sklearn.model_selection import StratifiedKFold

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


name =  "Lymphography"  # "NTN"  # "Biopax"  # "Family"  # "Lymphography"
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


targets = benchmark_owl_ndlm.load_lp_target(payload, lp_path)


# ============================================================
# Run 10-fold CV for every learning problem
# ============================================================

for lp_idx, t in enumerate(targets):

    print(f"\n{'=' * 60}")
    print(f"LP {lp_idx}: {t['name']}")
    print(f"Positive examples: {t['num_positive']}")
    print(f"Negative examples: {t['num_negative']}")
    print(f"{'=' * 60}")

    # --------------------------------------------------------
    # Get labeled examples
    # --------------------------------------------------------

    labeled_indices = torch.where(t["mask"])[0].cpu().numpy()

    labels = (
        t["target"][labeled_indices]
        .cpu()
        .numpy()
        .astype(int)
    )

    # --------------------------------------------------------
    # Stratified 10-fold split
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(labeled_indices, labels)
    ):

        print(f"\n--- Fold {fold + 1}/10 ---")

        train_objects = labeled_indices[train_idx]
        test_objects = labeled_indices[test_idx]

        # ====================================================
        # Training target / mask
        # ====================================================

        train_target = torch.zeros_like(t["target"])
        train_mask = torch.zeros_like(t["mask"])

        train_target[train_objects] = t["target"][train_objects]
        train_mask[train_objects] = True

        # ====================================================
        # Test target / mask
        # ====================================================

        test_target = torch.zeros_like(t["target"])
        test_mask = torch.zeros_like(t["mask"])

        test_target[test_objects] = t["target"][test_objects]
        test_mask[test_objects] = True

        # Add batch dimensions
        c_train_target = (
            train_target
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        c_train_mask = (
            train_mask
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        c_test_target = (
            test_target
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        c_test_mask = (
            test_mask
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        # ====================================================
        # No role learning
        # ====================================================

        r_train_target = torch.zeros(
            (1, 0, roles.shape[2], roles.shape[2]),
            dtype=torch.float32,
            device=device
        )

        r_train_mask = torch.zeros(
            (1, 0, roles.shape[2], roles.shape[2]),
            dtype=torch.bool,
            device=device
        )

        r_test_target = torch.zeros(
            (1, 0, roles.shape[2], roles.shape[2]),
            dtype=torch.float32,
            device=device
        )

        r_test_mask = torch.zeros(
            (1, 0, roles.shape[2], roles.shape[2]),
            dtype=torch.bool,
            device=device
        )

        # ====================================================
        # Configuration
        # ====================================================

        config = config_object()

        config.NUM_LAYERS = 4
        config.NUM_HIDDEN_CONCEPTS = 10
        config.NUM_HIDDEN_ROLES = 10
        config.MODE = "strict"
        config.TRANSITIVE_CLOSURE = False
        config.INITIAL_FFN = True

        args = argparse.Namespace(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=5,
            loss_type="BCE",
            num_epochs=500,
            test_interval=100,
            weighted_loss=False,
            experiment_path=Path(
                f"outputs/CONCEPT_LEARNER/"
                f"benchmark_owl_ndlm/"
                f"{name}_{lp_idx}_fold_{fold}_{time.time()}/"
            ),
        )

        os.makedirs(args.experiment_path, exist_ok=True)

        init_logger(
            f"{args.experiment_path}/output.log"
        )
        log(f"concepts shape: {concepts.shape}")
        log(f"roles shape: {roles.shape}")
        log(f"Train examples: {len(train_objects)}")
        log(f"Test examples: {len(test_objects)}")
        log(f"Train positives: {train_target.sum().item()}")
        log(f"Test positives: {test_target.sum().item()}")

        # ====================================================
        # Model
        # ====================================================

        model = MultiLayerNDLM(
            concepts.shape[1],
            roles.shape[1],
            1,
            0,
            config
        ).to(device).eval()

        # ====================================================
        # Train / test data
        # ====================================================

        train = [(
            concepts,
            roles,
            c_train_target,
            r_train_target,
            c_train_mask,
            r_train_mask
        )]

        test = [(
            concepts,
            roles,
            c_test_target,
            r_test_target,
            c_test_mask,
            r_test_mask
        )]

        # ====================================================
        # Train and evaluate
        # ====================================================

        tp, tn, fp, fn, _, _, _, _ = NDLM_main.main(
            train,
            test,
            config,
            args,
            model,
            checkpoint_path=(
                args.experiment_path / "checkpoints"
                if args.experiment_path
                else None
            ),
            log=log
        )

        print(
            f"TP={tp}, FP={fp}, TN={tn}, FN={fn}"
        )

        # Accumulate across folds
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    # ========================================================
    # Final 10-fold metrics
    # ========================================================

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    accuracy = (
        (total_tp + total_tn)
        / (total_tp + total_tn + total_fp + total_fn)
    )

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {t['name']}")
    print(f"{'=' * 60}")
    print(f"TP:        {total_tp}")
    print(f"FP:        {total_fp}")
    print(f"TN:        {total_tn}")
    print(f"FN:        {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
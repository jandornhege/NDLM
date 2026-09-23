import math
import os
from pathlib import Path
from typing_extensions import final

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
import time

class NoisyORLoss(nn.Module):
    def __init__(self, lambda_neg=1.0, eps=1e-8):
        super().__init__()
        self.lambda_neg = lambda_neg
        self.eps = eps

    def forward(self, logits, target):

        target = target.float()
        probs = torch.sigmoid(logits)

        # flatten argument dimensions
        probs = probs.flatten(start_dim=2)
        target = target.flatten(start_dim=2)

        # ----- positive loss -----

        success = 1 - torch.prod(
            1 - probs * target,
            dim=-1
        )

        loss_pos = -torch.log(success + self.eps)

        # ----- negative loss -----

        loss_neg = F.binary_cross_entropy_with_logits(
            logits.flatten(start_dim=2),
            torch.zeros_like(probs),
            reduction="none",
        )

        loss_neg = (loss_neg * (1 - target)).sum(dim=-1)

        return (loss_pos + self.lambda_neg * loss_neg).mean()

class SoftmaxSetLoss(nn.Module):
    """
    Maximizes probability assigned to the set of optimal groundings.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, target):

        target = target.bool()

        logits = logits.flatten(start_dim=2)
        target = target.flatten(start_dim=2)

        log_probs = F.log_softmax(logits, dim=-1)

        masked = log_probs.masked_fill(
            ~target,
            float("-inf")
        )

        loss = -torch.logsumexp(masked, dim=-1)

        return loss.mean()


def plateau_stop_criterion(
    current_loss,
    best_loss,
    grad_norm,
    stop_patience,
    min_delta=1e-4,
    grad_norm_threshold=1e-5,
    mode="loss_and_grad_plateau",
    plateau_count=0,
    grad_plateau_count=0,
):
    """Check whether the training loop is plateauing.

    The criterion considers three behaviors:
      1. loss is no longer improving beyond min_delta for stop_patience epochs;
      2. the average gradient norm has fallen below grad_norm_threshold;
      3. the plateau is sustained for enough consecutive epochs to be considered
         effectively stationary.
    """
    if stop_patience <= 0:
        return False, "Early stopping disabled."

    if mode not in {"converged", "loss_only", "gradient_only", "loss_and_grad_plateau"}:
        raise ValueError(f"Unknown early-stop mode: {mode!r}")

    if current_loss < best_loss - min_delta:
        return False, "Loss is still improving."

    if mode == "converged":
        should_stop = plateau_count >= stop_patience
        reason = (
            f"Convergence detected: loss improvement stayed below {min_delta:.3e} "
            f"for {plateau_count}/{stop_patience} epochs."
        )
        return should_stop, reason

    if mode == "loss_only":
        should_stop = plateau_count >= stop_patience
        reason = (
            f"Loss plateau detected for {plateau_count}/{stop_patience} epochs; "
            "the model has stopped improving."
        )
        return should_stop, reason

    if mode == "gradient_only":
        should_stop = grad_plateau_count >= stop_patience
        reason = (
            f"Gradient norm has remained below {grad_norm_threshold:.3e} for "
            f"{grad_plateau_count}/{stop_patience} epochs."
        )
        return should_stop, reason

    should_stop = (
        plateau_count >= stop_patience
        and grad_plateau_count >= min(stop_patience, max(1, stop_patience // 2))
        and grad_norm <= grad_norm_threshold
    )
    reason = (
        f"Loss and gradient plateau detected: loss plateau {plateau_count}/{stop_patience}, "
        f"grad plateau {grad_plateau_count}/{stop_patience}, grad_norm={grad_norm:.3e}."
    )
    return should_stop, reason


def compute_pos_weight(targets, log):
    # targets shape: (N, ...)
    if len(targets) == 0:
        return torch.ones(0)  
    if targets[0].shape[1] == 0:
        return torch.ones(0)  
    positive_counts = torch.zeros([targets[0].shape[1]], dtype=torch.float32, device=targets[0].device)
    num_elements = 0
   

    for ds in targets:
        if len(ds.shape) == 3:
            positive_counts += ds.sum(dim=(0,2))  # Sum over the last dimension if it's a 3D tensor
            num_elements += ds.shape[0]*ds.shape[2] 
        elif len(ds.shape) ==4:
            positive_counts += ds.sum(dim=(0,2,3))  # Sum over the last two dimensions if it's a 4D tensor 
            num_elements += ds.shape[0]*ds.shape[2]*ds.shape[3]
    pos_weight = (num_elements - positive_counts) / (positive_counts + 1e-8)
    pos_weight = torch.clamp(pos_weight, max=100)
    log(f"Computed pos_weight: {pos_weight}")
    return pos_weight

def main(train_dataset, test_dataset, config, args, model, checkpoint_path=None, log=print):
    #expected data format:
    # train and test are lists of datasets, where each dataset is a list of tuples (concepts, roles, c_targets, r_targets)
    # concepts: (num_concepts, num_objects)
    # roles: (num_roles, num_objects, num_objects)
    # c_targets: (num_out_concepts, num_objects)
    # r_targets: (num_out_roles, num_objects, num_objects)  
    tp_c, tn_c, fp_c, fn_c, tp_r, tn_r, fp_r, fn_r = [None] * 8
    log(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} testing samples.")

   
   
    in_concepts = train_dataset[0][0].shape[1]
    in_roles = train_dataset[0][1].shape[1]
    out_concepts = train_dataset[0][2].shape[1]
    out_roles = train_dataset[0][3].shape[1]
    
    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    pos_weight_c = compute_pos_weight([ds[2] for ds in train_dataset], log).to(device)
    pos_weight_r = compute_pos_weight([ds[3] for ds in train_dataset], log).to(device)
    pos_weight_c = pos_weight_c[:, None]
    pos_weight_r = pos_weight_r[:, None, None]
    
   
    # Cross Entropy loss for multi-label (applies Sigmoid and weights positive examples according to their ration in the whole dataset)
    if args.loss_type == "Noisy_OR":
        criterion_c = NoisyORLoss()
        criterion_r = NoisyORLoss()
    elif args.loss_type == "SoftmaxSet":
        criterion_c = SoftmaxSetLoss()
        criterion_r = SoftmaxSetLoss()
    elif args.loss_type == "BCE" and args.weighted_loss:
        criterion_c = nn.BCEWithLogitsLoss(pos_weight=pos_weight_c)     
        criterion_r = nn.BCEWithLogitsLoss(pos_weight=pos_weight_r)
    else:
        criterion_c = nn.BCEWithLogitsLoss()
        criterion_r = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    lr_scheduler_name = getattr(args, "lr_scheduler", None)
    if lr_scheduler_name in (None, "none"):
        lr_scheduler = None
    elif lr_scheduler_name == "plateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20,
        )
    elif lr_scheduler_name == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs,
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler_name!r}")

    final_train_res = {}
    final_test_res = {}

    def result_with_details(test_totals, test_per_dataset):
        train_totals = {
            "c_miss": sum(value["c_miss"] for value in final_train_res.values()),
            "r_miss": sum(value["r_miss"] for value in final_train_res.values()),
            "overc": sum(value["overc"] for value in final_train_res.values()),
            "underc": sum(value["underc"] for value in final_train_res.values()),
            "overr": sum(value["overr"] for value in final_train_res.values()),
            "underr": sum(value["underr"] for value in final_train_res.values()),
        }
        test_tp_c, test_tn_c, test_fp_c, test_fn_c, test_tp_r, test_tn_r, test_fp_r, test_fn_r = test_totals
        details = {
            "train": {
                "per_dataset": dict(final_train_res),
                "totals": train_totals,
            },
            "test": {
                "per_dataset": {
                    f"dataset_{index}": {
                        "tp_c": values[0],
                        "tn_c": values[1],
                        "fp_c": values[2],
                        "fn_c": values[3],
                        "tp_r": values[4],
                        "tn_r": values[5],
                        "fp_r": values[6],
                        "fn_r": values[7],
                    }
                    for index, values in enumerate(test_per_dataset)
                },
                "totals": {
                    "tp_c": test_tp_c,
                    "tn_c": test_tn_c,
                    "fp_c": test_fp_c,
                    "fn_c": test_fn_c,
                    "tp_r": test_tp_r,
                    "tn_r": test_tn_r,
                    "fp_r": test_fp_r,
                    "fn_r": test_fn_r,
                },
            },
        }
        if getattr(args, "return_details", False):
            return details
        return test_totals

    # ------------------------------
    # Training loop
    # ------------------------------


    def run_test_round(epoch_label):
        totals = [0] * 8
        per_dataset_totals = []
        model.eval()
        with torch.no_grad():
            for dataset_idx, data in enumerate(test_dataset):
                if len(data) == 4:
                    concepts, roles, c_targets, r_targets = data
                    c_mask = torch.ones_like(c_targets, dtype=torch.bool)
                    r_mask = torch.ones_like(r_targets, dtype=torch.bool)
                elif len(data) == 6:
                    concepts, roles, c_targets, r_targets, c_mask, r_mask = data
                else:
                    raise ValueError(f"Unexpected data length: {len(data)}")
                
                concepts = concepts.to(device)
                roles = roles.to(device)

                c_mask = c_mask.to(device)
                r_mask = r_mask.to(device)

                c_targets = c_targets.to(device)
                r_targets = r_targets.to(device)
                
                out_concepts, out_roles = model(concepts, roles)
                # if "final" in epoch_label:
                #     log(epoch_label)
                    # torch.set_printoptions(threshold=float('inf'))
                    # log(f"Concepts: {concepts}")
                    # log(f"Roles: {roles}")
                    # log(f"C Targets: {c_targets}")
                    # log(f"R Targets: {r_targets}")
                    # log(f"Out Concepts: {out_concepts}")
                    # log(f"Out Roles: {out_roles}")

                


                c_preds = ((out_concepts > 0) & c_mask).int()
                c_missclassifcations = ((c_preds != c_targets) & c_mask).sum().item()
                c_correct = ((c_preds == c_targets) & c_mask).sum().item()
                c_accuracy = (c_preds == c_targets)[c_mask].float().mean()
                over_estimates_c = ((c_preds > c_targets) & c_mask).sum().item()
                under_estimates_c = ((c_preds < c_targets) & c_mask).sum().item()

                r_preds = ((out_roles > 0) & r_mask).int()
                role_missclassifcations = ((r_preds != r_targets) & r_mask).sum().item()
                r_correct = ((r_preds == r_targets) & r_mask).sum().item()
                r_accuracy = (
                    (r_preds == r_targets)[r_mask].float().mean()
                    if r_mask.any()
                    else torch.tensor(float("nan"), device=device)
                )
                over_estimates_r = ((r_preds > r_targets) & r_mask).sum().item()
                under_estimates_r = ((r_preds < r_targets) & r_mask).sum().item()

                tp_c = ((c_preds == 1) & (c_targets == 1) & c_mask).sum().item()
                tn_c = ((c_preds == 0) & (c_targets == 0) & c_mask).sum().item()
                fp_c = over_estimates_c
                fn_c = under_estimates_c
                tp_r = ((r_preds == 1) & (r_targets == 1) & r_mask).sum().item()
                tn_r = ((r_preds == 0) & (r_targets == 0) & r_mask).sum().item()
                fp_r = over_estimates_r
                fn_r = under_estimates_r

                dataset_totals = (tp_c, tn_c, fp_c, fn_c, tp_r, tn_r, fp_r, fn_r)
                per_dataset_totals.append(dataset_totals)
                totals = [total + value for total, value in zip(totals, dataset_totals)]
                
                if verbose:
                    log(f"Test Misclassifications (Dataset {dataset_idx}) [{epoch_label}]: c:{c_missclassifcations}, r:{role_missclassifcations}; overc:{over_estimates_c}, underc:{under_estimates_c}; overr:{over_estimates_r}, underr:{under_estimates_r}")
                    log(f"Test Accuracy (Dataset {dataset_idx}) [{epoch_label}]: c:{c_accuracy}, r:{r_accuracy}")
                final_test_res[f"dataset_{dataset_idx}"] = {"c_miss": c_missclassifcations, "r_miss": role_missclassifcations, "overc": over_estimates_c, "underc": under_estimates_c, "overr": over_estimates_r, "underr": under_estimates_r}
            model.train()
            log(f"Test summary [{epoch_label}]: c_miss={totals[2]+totals[3]}, r_miss={totals[6]+totals[7]} over {len(test_dataset)} datasets")
            return tuple(totals), per_dataset_totals

    early_stop_patience = getattr(args, "early_stop_patience", 5)
    early_stop_mode = getattr(args, "early_stop_mode", "loss_and_grad_plateau")
    early_stop_min_delta = getattr(args, "early_stop_min_delta", 1e-4)
    early_stop_grad_norm_threshold = getattr(args, "early_stop_grad_norm_threshold", 1e-5)
    zero_miss_streak = 0
    plateau_count = 0
    grad_plateau_count = 0
    best_epoch_loss = float("inf")
    shuffle_datasets = True
    fold_time_limit_seconds = getattr(args, "fold_time_limit_seconds", None)
    fold_start_time = time.time()
    verbose = getattr(args, "verbose", False)

    total_accs=[]
    time_stats = []
    for epoch in range(1, args.num_epochs+1):
        
        epoch_start = time.time()
        epoch_total_miss = 0
        epoch_loss_sum = 0.0
        epoch_grad_norm_sum = 0.0
        epoch_grad_updates = 0
        epoch_c_miss_total = 0
        epoch_r_miss_total = 0
        dataset_order = (
            torch.randperm(len(train_dataset)).tolist()
            if shuffle_datasets
            else list(range(len(train_dataset)))
        )

        # Iterate over datasets (dataset is a tuple of (concepts, roles, c_targets, r_targets), such that instances within the datasets have the same number of objects)
        for ds_idx in dataset_order:
            data = train_dataset[ds_idx]
            dataset_start = time.time()
            if len(data) == 4:
                concepts, roles, c_targets, r_targets = data
                c_mask = torch.ones_like(c_targets, dtype=torch.bool)
                r_mask = torch.ones_like(r_targets, dtype=torch.bool)
            elif len(data) == 6:
                concepts, roles, c_targets, r_targets, c_mask, r_mask = data
            else:
                raise ValueError(f"Unexpected data length: {len(data)}")
            
                    
            accs=[]
            miss=[]
            over=[]
            under=[]
            avg_loss = 0.0
            batch_size = args.batch_size if args.batch_size > 0 else concepts.shape[0]  # use all samples if batch size is 0 or negative
            num_samples = concepts.shape[0]
            perm = torch.randperm(num_samples)

            # if c_targets.numel() != 0:
                # log(f"c_targets shape: {c_targets.shape}")
                # log(f"c_targets max: {c_targets.max()}")
                # if c_targets.max() == 0:
                    # log(f"All c_targets are zero for dataset {ds_idx}. This should not happen.")
            # if r_targets.numel() != 0:
                # log(f"r_targets shape: {r_targets.shape}")
                # log(f"r_targets max: {r_targets.max()}")
                # if r_targets.max() == 0:
                    # log(f"All r_targets are zero for dataset {ds_idx}. This should not happen.")
            
            # Go in batches through the dataset with random order
            for i in range(0, num_samples, batch_size):
                idx = perm[i:i+batch_size]
                concepts_batch = concepts[idx].to(device) # shape: (B, num_concepts, num_objects)
                roles_batch = roles[idx].to(device)   # shape: (B, num_roles, num_objects, num_objects)
                c_targets_batch = c_targets[idx].to(device)   # shape: (B, num_out_concepts, num_objects)
                r_targets_batch = r_targets[idx].to(device)   # shape: (B, num_out_roles, num_objects, num_objects)
                c_mask_batch = c_mask[idx].to(device)
                r_mask_batch = r_mask[idx].to(device)
                optimizer.zero_grad()
                out_concepts, out_roles = model(concepts_batch, roles_batch)  # (B, ...)
                # Combine loss of concepts and roles
                c_loss = F.binary_cross_entropy_with_logits(
                    out_concepts,
                    c_targets_batch.float(),
                    pos_weight=pos_weight_c if args.weighted_loss else None,
                    reduction="none"
                )

                r_loss = F.binary_cross_entropy_with_logits(
                    out_roles,
                    r_targets_batch.float(),
                    pos_weight=pos_weight_r if args.weighted_loss else None,
                    reduction="none"
                )

                c_loss = (
                    c_loss[c_mask_batch].mean()
                    if c_mask_batch.any()
                    else out_concepts.sum() * 0.0
                )
                r_loss = (
                    r_loss[r_mask_batch].mean()
                    if r_mask_batch.any()
                    else out_roles.sum() * 0.0
                )

                loss = c_loss + r_loss

                # loss = criterion_c(out_concepts, c_targets_batch) + criterion_r(out_roles, r_targets_batch)
                
                loss.backward()
                grad_norm_sq = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm_sq += param.grad.detach().pow(2).sum().item()
                epoch_grad_norm_sum += math.sqrt(grad_norm_sq)
                epoch_grad_updates += 1
                optimizer.step()

                # Add loss to average loss for whole dataset
                avg_loss += loss.item() * len(idx)

                
                
                # Missclassifications are computed by thresholding the outputs at 0.5 and comparing to targets
                c_preds = (out_concepts > 0).int()

                c_correct = (c_preds == c_targets_batch)
                c_incorrect = (c_preds != c_targets_batch)

                c_missclassifcations = c_incorrect[c_mask_batch].sum().item()
                c_accuracy = c_correct[c_mask_batch].float().mean()

                over_estimates_c = ((c_preds > c_targets_batch) & c_mask_batch).sum().item()
                under_estimates_c = ((c_preds < c_targets_batch) & c_mask_batch).sum().item()


                r_preds = (out_roles > 0).int()

                r_correct = (r_preds == r_targets_batch)
                r_incorrect = (r_preds != r_targets_batch)

                r_missclassifications = r_incorrect[r_mask_batch].sum().item()
                r_accuracy = (
                    r_correct[r_mask_batch].float().mean()
                    if r_mask_batch.any()
                    else torch.tensor(float("nan"), device=device)
                )

                over_estimates_r = ((r_preds > r_targets_batch) & r_mask_batch).sum().item()
                under_estimates_r = ((r_preds < r_targets_batch) & r_mask_batch).sum().item()
                                
                # if r_missclassifcations >= 1:
                #     log(f"Missclassifications in concepts: {r_missclassifcations} out of {r_targets_batch.numel()} samples.")
                #     log(f"Predictions: {r_preds}")
                #     log(f"Targets: {r_targets_batch}")
                #     log(f"Input concepts: {concepts_batch}")
                #     log(f"Input roles: {roles_batch}")

                accs.append((c_accuracy, r_accuracy))
                miss.append((c_missclassifcations , r_missclassifications))
                over.append((over_estimates_c, over_estimates_r))
                under.append((under_estimates_c, under_estimates_r))

            avg_loss /= num_samples
            epoch_loss_sum += avg_loss
            epoch_c_miss_total += sum(i[0] for i in miss)
            epoch_r_miss_total += sum(i[1] for i in miss)
            if verbose:
                log(f"Epoch {epoch:4d} (Dataset {ds_idx}) | Average Accuracy: {(sum([i[0] for i in accs])/len(accs)).item(), (sum([i[1] for i in accs])/len(accs)).item()}")
                log(f"           | Total Misclassifications: {(sum([i[0] for i in miss]), sum([i[1] for i in miss]))}, over: {(sum([i[0] for i in over]), sum([i[1] for i in over]))}, under: {(sum([i[0] for i in under]), sum([i[1] for i in under]))}")
                log(f"           | Loss: {avg_loss:.4f}")
                log(f"           | Time taken: {time.time()-dataset_start:.4f} seconds")
            final_train_res[f"dataset_{ds_idx}"] = {"c_miss": sum(i[0] for i in miss), "r_miss": sum(i[1] for i in miss), "overc": sum(i[0] for i in over), "underc": sum(i[0] for i in under), "overr": sum(i[1] for i in over), "underr": sum(i[1] for i in under)}
            epoch_total_miss += sum(i[0] for i in miss) + sum(i[1] for i in miss)

            
        epoch_time = time.time() - epoch_start
        time_stats.append(epoch_time)
        num_datasets = len(train_dataset)
        epoch_avg_loss = epoch_loss_sum / num_datasets
        epoch_avg_grad_norm = epoch_grad_norm_sum / max(epoch_grad_updates, 1)
        log(
            f"Epoch {epoch:4d} | Total time taken: {epoch_time:.4f} seconds | "
            f"avg_loss={epoch_avg_loss:.4f} | "
            f"avg_grad_norm={epoch_avg_grad_norm:.4e} | "
            f"miss(c,r)=({epoch_c_miss_total},{epoch_r_miss_total})"
        )
        if lr_scheduler is not None:
            if lr_scheduler_name == "plateau":
                lr_scheduler.step(epoch_avg_loss)
            else:
                lr_scheduler.step()
        total_accs.append((sum([i[0] for i in accs])/len(accs), sum([i[1] for i in accs])/len(accs)))

        if epoch_avg_loss < best_epoch_loss - early_stop_min_delta:
            best_epoch_loss = epoch_avg_loss
            plateau_count = 0
        else:
            plateau_count += 1

        if epoch_avg_grad_norm <= early_stop_grad_norm_threshold:
            grad_plateau_count += 1
        else:
            grad_plateau_count = 0

        if epoch_total_miss == 0:
            zero_miss_streak += 1
        else:
            zero_miss_streak = 0

        should_stop, stop_reason = plateau_stop_criterion(
            current_loss=epoch_avg_loss,
            best_loss=best_epoch_loss,
            grad_norm=epoch_avg_grad_norm,
            stop_patience=early_stop_patience,
            min_delta=early_stop_min_delta,
            grad_norm_threshold=early_stop_grad_norm_threshold,
            mode=early_stop_mode,
            plateau_count=plateau_count,
            grad_plateau_count=grad_plateau_count,
        )

        if should_stop:
            log(f"Early stopping at epoch {epoch}: {stop_reason}")
            log("Running one additional test round before stopping.")
            if checkpoint_path is not None:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), Path(checkpoint_path)/f"checkpoint_final.pt")
                log(f"Saved final model checkpoint to {Path(checkpoint_path)/f'checkpoint_final.pt'}")
            test_totals, per_dataset_totals = run_test_round(f"epoch {epoch} (final)")
            return result_with_details(test_totals, per_dataset_totals)

        if zero_miss_streak >= early_stop_patience:
            log( f"Early stopping at epoch {epoch}: training misclassifications were 0 for {early_stop_patience} consecutive epochs.")
            log("Running one additional test round before stopping.")
            if checkpoint_path is not None:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), Path(checkpoint_path)/f"checkpoint_final.pt")
                log(f"Saved final model checkpoint to {Path(checkpoint_path)/f'checkpoint_final.pt'}")
            test_totals, per_dataset_totals = run_test_round(f"epoch {epoch} (final)")
            return result_with_details(test_totals, per_dataset_totals)

        if fold_time_limit_seconds is not None and time.time() - fold_start_time >= fold_time_limit_seconds:
            log(f"Stopping at epoch {epoch}: fold time limit of {fold_time_limit_seconds} seconds reached.")
            log("Running one additional test round before stopping.")
            if checkpoint_path is not None:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), Path(checkpoint_path)/f"checkpoint_final.pt")
                log(f"Saved final model checkpoint to {Path(checkpoint_path)/f'checkpoint_final.pt'}")
            test_totals, per_dataset_totals = run_test_round(f"epoch {epoch} (final, time limit)")
            return result_with_details(test_totals, per_dataset_totals)

        if args.test_interval > 0 and epoch % args.test_interval == 0:
            (tp_c, tn_c, fp_c, fn_c, tp_r, tn_r, fp_r, fn_r), per_dataset_totals = run_test_round(f"epoch {epoch}")
            
            # Log Model Checkpoint
            if checkpoint_path is not None:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), Path(checkpoint_path)/f"checkpoint_intermediate.pt")
                log(f"Saved model checkpoint to {Path(checkpoint_path)/f'checkpoint_intermediate.pt'}")
    
    if checkpoint_path is not None:
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(model.state_dict(), Path(checkpoint_path)/"checkpoint_final.pt")
        log(f"Saved final model checkpoint to {Path(checkpoint_path)/'checkpoint_final.pt'}")

    test_totals, per_dataset_totals = run_test_round(f"epoch {args.num_epochs} (final)")
    return result_with_details(test_totals, per_dataset_totals)

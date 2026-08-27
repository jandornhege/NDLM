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

    
def compute_pos_weight(targets, log):
    # targets shape: (N, ...)
    if len(targets) == 0:
        return torch.ones(0)  
    if targets[0].shape[1] == 0:
        return torch.ones(0)  
    positive_counts = torch.zeros([targets[0].shape[1]], dtype=torch.float32)
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
    final_train_res = {}
    final_test_res = {}

    # ------------------------------
    # Training loop
    # ------------------------------


    def run_test_round(epoch_label):
            
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_dataset):
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
                
                out_c= []
                out_r = []
                for i in range(concepts.shape[0]):
                    c, r = model(concepts[i:i+1], roles[i:i+1])
                    out_c.append(c)
                    out_r.append(r)
                out_concepts = torch.cat(out_c, dim=0)
                out_roles = torch.cat(out_r, dim=0)
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
                c_accuracy = ((c_preds == c_targets) & c_mask).float().mean()
                over_estimates_c = ((c_preds > c_targets) & c_mask).sum().item()
                under_estimates_c = ((c_preds < c_targets) & c_mask).sum().item()

                r_preds = ((out_roles > 0) & r_mask).int()
                role_missclassifcations = ((r_preds != r_targets) & r_mask).sum().item()
                r_correct = ((r_preds == r_targets) & r_mask).sum().item()
                r_accuracy = ((r_preds == r_targets) & r_mask).float().mean()
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
                
                log(f"Test Misclassifications (Dataset {i}) [{epoch_label}]: c:{c_missclassifcations}, r:{role_missclassifcations}; overc:{over_estimates_c}, underc:{under_estimates_c}; overr:{over_estimates_r}, underr:{under_estimates_r}")
                log(f"Test Accuracy (Dataset {i}) [{epoch_label}]: c:{c_accuracy}, r:{r_accuracy}")
                final_test_res[f"dataset_{i}"] = {"c_miss": c_missclassifcations, "r_miss": role_missclassifcations, "overc": over_estimates_c, "underc": under_estimates_c, "overr": over_estimates_r, "underr": under_estimates_r}
                return tp_c, tn_c, fp_c, fn_c, tp_r, tn_r, fp_r, fn_r
        model.train()

    early_stop_patience = 3
    zero_miss_streak = 0
    
    total_accs=[]
    time_stats = []
    for epoch in range(1, args.num_epochs+1):
        
        t0=time.time()
        epoch_total_miss = 0

        # Iterate over datasets (dataset is a tuple of (concepts, roles, c_targets, r_targets), such that instances within the datasets have the same number of objects)
        for ds_idx, data in enumerate(train_dataset):
            if len(data) == 4:
                concepts, roles, c_targets, r_targets = data
                c_mask = torch.ones_like(c_targets, dtype=torch.bool)
                r_mask = torch.ones_like(r_targets, dtype=torch.bool)
            elif len(data) == 6:
                concepts, roles, c_targets, r_targets, c_mask, r_mask = data
            
                    
            accs=[]
            miss=[]
            over=[]
            under=[]
            avg_loss = 0.0
            batch_size = args.batch_size if args.batch_size > 0 else concepts.shape[0]  # use all samples if batch size is 0 or negative
            num_samples = concepts.shape[0]
            perm = torch.randperm(num_samples)

            if c_targets.numel() != 0:
                # log(f"c_targets shape: {c_targets.shape}")
                # log(f"c_targets max: {c_targets.max()}")
                if c_targets.max() == 0:
                    log(f"All c_targets are zero for dataset {ds_idx}. This should not happen.")
            if r_targets.numel() != 0:
                # log(f"r_targets shape: {r_targets.shape}")
                # log(f"r_targets max: {r_targets.max()}")
                if r_targets.max() == 0:
                    log(f"All r_targets are zero for dataset {ds_idx}. This should not happen.")
            
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
                    reduction="none"
                )

                r_loss = F.binary_cross_entropy_with_logits(
                    out_roles,
                    r_targets_batch.float(),
                    reduction="none"
                )

                c_loss = c_loss[c_mask_batch].mean()
                r_loss = r_loss[r_mask_batch].mean()

                loss = c_loss + r_loss

                # loss = criterion_c(out_concepts, c_targets_batch) + criterion_r(out_roles, r_targets_batch)
                
                loss.backward()
                optimizer.step()

                # Add loss to average loss for whole dataset
                avg_loss += loss/(num_samples)

                
                
                # Missclassifications are computed by thresholding the outputs at 0.5 and comparing to targets
                c_preds = (out_concepts > 0).int()

                c_correct = (c_preds == c_targets)
                c_incorrect = (c_preds != c_targets)

                c_missclassifcations = c_incorrect[c_mask].sum().item()
                c_accuracy = c_correct[c_mask].float().mean()

                over_estimates_c = ((c_preds > c_targets) & c_mask).sum().item()
                under_estimates_c = ((c_preds < c_targets) & c_mask).sum().item()


                r_preds = (out_roles > 0).int()

                r_correct = (r_preds == r_targets)
                r_incorrect = (r_preds != r_targets)

                r_missclassifications = r_incorrect[r_mask].sum().item()
                r_accuracy = r_correct[r_mask].float().mean()

                over_estimates_r = ((r_preds > r_targets) & r_mask).sum().item()
                under_estimates_r = ((r_preds < r_targets) & r_mask).sum().item()
                                
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

            log(f"Epoch {epoch:4d} (Dataset {ds_idx}) | Average Accuracy: {(sum([i[0] for i in accs])/len(accs)).item(), (sum([i[1] for i in accs])/len(accs)).item()}")
            log(f"           | Total Misclassifications: {(sum([i[0] for i in miss]), sum([i[1] for i in miss]))}, over: {(sum([i[0] for i in over]), sum([i[1] for i in over]))}, under: {(sum([i[0] for i in under]), sum([i[1] for i in under]))}")
            log(f"           | Loss: {avg_loss:.4f}")
            log(f"           | Time taken: {time.time()-t0:.4f} seconds")
            final_train_res[f"dataset_{ds_idx}"] = {"c_miss": c_missclassifcations, "r_miss": r_missclassifications, "overc": over_estimates_c, "underc": under_estimates_c, "overr": over_estimates_r, "underr": under_estimates_r}
            epoch_total_miss += sum(i[0] for i in miss) + sum(i[1] for i in miss)

            
        time_stats.append(time.time()-t0)
        total_accs.append((sum([i[0] for i in accs])/len(accs), sum([i[1] for i in accs])/len(accs)))

        if epoch_total_miss == 0:
            zero_miss_streak += 1
        else:
            zero_miss_streak = 0

        if zero_miss_streak >= early_stop_patience:
            log( f"Early stopping at epoch {epoch}: training misclassifications were 0 for {early_stop_patience} consecutive epochs.")
            log("Running one additional test round before stopping.")
            if checkpoint_path is not None:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), Path(checkpoint_path)/f"checkpoint_final.pt")
                log(f"Saved final model checkpoint to {Path(checkpoint_path)/f'checkpoint_final.pt'}")
            tp_c, tn_c, fp_c, fn_c, tp_r, tn_r, fp_r, fn_r = run_test_round(f"epoch {epoch} (final)")
            return tp_c, tn_c, fp_c, fn_c, tp_r, tn_r, fp_r, fn_r

        if epoch % args.test_interval == 0:
            # ------------------------------
            # Test outputs
            # ------------------------------
            tp_c, tn_c, fp_c, fn_c, tp_r, tn_r, fp_r, fn_r = run_test_round(f"epoch {epoch}")
            
            # Log Model Checkpoint
            if checkpoint_path is not None:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), Path(checkpoint_path)/f"checkpoint_intermediate.pt")
                log(f"Saved model checkpoint to {Path(checkpoint_path)/f'checkpoint_intermediate.pt'}")
    
    return tp_c, tn_c, fp_c, fn_c, tp_r, tn_r, fp_r, fn_r

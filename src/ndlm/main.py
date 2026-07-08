from xml.parsers.expat import model

import torch
import torch.nn as nn
import torch.optim as optim

import sys
import time

def main(train_dataset, test_dataset, config, model):
    #expected data format:
    # train and test are lists of datasets, where each dataset is a list of tuples (concepts, roles, c_targets, r_targets)
    # concepts: (num_concepts, num_objects)
    # roles: (num_roles, num_objects, num_objects)
    # c_targets: (num_out_concepts, num_objects)
    # r_targets: (num_out_roles, num_objects, num_objects)

    print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} testing samples.")

   
   
    in_concepts = train_dataset[0][0].shape[1]
    in_roles = train_dataset[0][1].shape[1]
    out_concepts = train_dataset[0][2].shape[1]
    out_roles = train_dataset[0][3].shape[1]
    
    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Cross Entropy loss for multi-label (applies Sigmoid)
    criterion = nn.BCEWithLogitsLoss()    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # ------------------------------
    # Training loop
    # ------------------------------
    def run_test_round(epoch_label):
        model.eval()
        with torch.no_grad():
            for i, (concepts, roles, c_targets, r_targets) in enumerate(test_dataset):
                concepts = concepts.to(device)
                roles = roles.to(device)
                c_targets = c_targets.to(device)
                r_targets = r_targets.to(device)

                out_concepts, out_roles = model(concepts, roles)

                c_preds = (out_concepts > 0.5).int()
                c_missclassifcations = (c_preds != c_targets).sum().item()
                c_accuracy = (c_preds == c_targets).float().mean()

                r_preds = (out_roles > 0.5).int()
                role_missclassifcations = (r_preds != r_targets).sum().item()
                r_accuracy = (r_preds == r_targets).float().mean()

                print(f"Test Misclassifications (Dataset {i}) [{epoch_label}]: c:{c_missclassifcations}, r:{role_missclassifcations}", flush=True)
                print(f"Test Accuracy (Dataset {i}) [{epoch_label}]: c:{c_accuracy}, r:{r_accuracy}", flush=True)

        model.train()

    early_stop_patience = 10
    zero_miss_streak = 0
    
    total_accs=[]
    time_stats = []
    for epoch in range(1, config.NUM_EPOCHS+1):
        
        t0=time.time()
        epoch_total_miss = 0

        # Iterate over datasets (dataset is a tuple of (concepts, roles, c_targets, r_targets), such that instances within the datasets have the same number of objects)
        for ds_idx, (concepts, roles, c_targets, r_targets) in enumerate(train_dataset):
            accs=[]
            miss=[]
            avg_loss = 0.0
            batch_size = config.BATCH_SIZE if config.BATCH_SIZE > 0 else concepts.shape[0]  # use all samples if batch size is 0 or negative
            num_samples = concepts.shape[0]
            perm = torch.randperm(num_samples)
            
            # Go in batches through the dataset with random order
            for i in range(0, num_samples, batch_size):
                idx = perm[i:i+batch_size]
                concepts_batch = concepts[idx].to(device)
                roles_batch = roles[idx].to(device)
                c_targets_batch = c_targets[idx].to(device)
                r_targets_batch = r_targets[idx].to(device)

                optimizer.zero_grad()
                out_concepts, out_roles = model(concepts_batch, roles_batch)  # (B, ...)
                
                # Combine loss of concepts and roles
                loss = criterion(out_concepts, c_targets_batch) + criterion(out_roles, r_targets_batch)

                loss.backward()
                optimizer.step()

                # Add loss to average loss for whole dataset
                avg_loss += loss/(num_samples)

                
                
                # Missclassifications are computed by thresholding the outputs at 0.5 and comparing to targets
                c_preds = (out_concepts > 0.5).int()
                c_missclassifcations = (c_preds != c_targets_batch).sum().item()
                c_accuracy = (c_preds == c_targets_batch).float().mean()
                
                r_preds = (out_roles > 0.5).int()   # Predicted classes for roles
                role_missclassifcations = (r_preds != r_targets_batch).sum().item()
                r_accuracy = (r_preds == r_targets_batch).float().mean()

                accs.append((c_accuracy, r_accuracy))
                miss.append((c_missclassifcations , role_missclassifcations))

            print(f"Epoch {epoch:4d} (Dataset {ds_idx}) | Average Accuracy: {(sum([i[0] for i in accs])/len(accs)).item(), (sum([i[1] for i in accs])/len(accs)).item()}")
            print(f"           | Total Misclassifications: {(sum([i[0] for i in miss]), sum([i[1] for i in miss]))}", flush=True)
            print(f"           | Loss: {avg_loss:.4f}", flush=True)
            print(f"           | Time taken: {time.time()-t0:.4f} seconds", flush=True)

            epoch_total_miss += sum(i[0] for i in miss) + sum(i[1] for i in miss)

            
        time_stats.append(time.time()-t0)
        total_accs.append((sum([i[0] for i in accs])/len(accs), sum([i[1] for i in accs])/len(accs)))

        if epoch_total_miss == 0:
            zero_miss_streak += 1
        else:
            zero_miss_streak = 0

        if zero_miss_streak >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: training misclassifications were 0 "
                f"for {early_stop_patience} consecutive epochs.",
                flush=True,
            )
            print("Running one additional test round before stopping.", flush=True)
            run_test_round(f"epoch {epoch} (final)")
            break

        if epoch % config.TEST_INTERVAL == 0:
            # ------------------------------
            # Test outputs
            # ------------------------------
            run_test_round(f"epoch {epoch}")

    return model

import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import wandb

from model import GeoguessrModel
from clusters import ClusterManager
from dataset import create_dataloader
from evaluator import Evaluator 

# --- CONFIG ---
LOCAL_DATA_DIR = "./osv5m_local_data"
VAL_CACHE_DIR = "./val_cache"
MICRO_BATCH_SIZE = 256
ACCUM_STEPS = 1
LEARNING_RATE = 1e-4
STEPS = 1000
EVAL_INTERVAL = 100
DEVICE = "cuda"
NUM_WORKERS = 12
NUM_CLUSTERS = 200
SIGMA_KM = 150
LR_BACKBONE = 1e-6
LR_HEAD = 1e-3
LAMBDA_CLS = 1.0
LAMBDA_REG = 30.0
PROJECT_NAME = "neuroguessr"

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    # 1. Initialize WandB
    wandb.init(
        project=PROJECT_NAME,
        config={
            "batch_size": MICRO_BATCH_SIZE * ACCUM_STEPS,
            "lr": LEARNING_RATE,
            "clusters": NUM_CLUSTERS,
            "steps": STEPS
        }
    )

    # 2. Setup Clusters & Model
    cluster_manager = ClusterManager(n_clusters=NUM_CLUSTERS)
    try:
        cluster_centers, loss_weights = cluster_manager.load(device=DEVICE)
    except FileNotFoundError:
        cluster_centers, loss_weights = cluster_manager.generate()
        loss_weights = loss_weights.to(DEVICE)

    soft_targets_matrix = cluster_manager.generate_soft_targets(sigma_km=SIGMA_KM)
    soft_targets_matrix = soft_targets_matrix.to(DEVICE)

    search_path = os.path.join(LOCAL_DATA_DIR, "train", "*.tar")
    tar_files = glob.glob(search_path)
    if not tar_files:
        tar_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.tar"))
        
    model = GeoguessrModel(num_classes=NUM_CLUSTERS).to(DEVICE)
    model = torch.compile(model)
    
    train_loader = create_dataloader(
        tar_files=tar_files,
        model_config=model.get_config(),
        cluster_centers=cluster_centers,
        batch_size=MICRO_BATCH_SIZE,
        workers=NUM_WORKERS,
        mode='train'
    )
    
    # 3. Initialize Evaluator
    evaluator = None
    if os.path.exists(VAL_CACHE_DIR):
        print("Initializing Validator...")
        evaluator = Evaluator(
            val_dir=VAL_CACHE_DIR, 
            model_config=model.get_config(),
            num_clusters=NUM_CLUSTERS,
            batch_size=256, 
            device=DEVICE
        )
    else:
        print("WARNING: No validation cache found. Run prepare_val.py first!")
    
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "backbone" in name:
            backbone_params.append(param)
        elif "classifier" in name or "regressor" in name:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params, 'lr': LR_HEAD}
    ])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[LR_BACKBONE, LR_HEAD], 
        total_steps=STEPS, 
        pct_start=0.1
    )
    
    crit_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
    crit_reg = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    print("--- TRAINING START ---")
    model.train()
    optimizer.zero_grad()

    for step, (imgs, labels, true_xyz) in enumerate(train_loader):
        if step >= (STEPS * ACCUM_STEPS):
            break

        imgs, labels, true_xyz = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True), true_xyz.to("cuda")
        target_probs = soft_targets_matrix[labels]
        
        with torch.amp.autocast('cuda'):
            logits, coords = model(imgs)

            l_cls = crit_cls(logits, labels)
            l_reg = crit_reg(coords, true_xyz)
            loss = ((LAMBDA_CLS * l_cls) + (LAMBDA_REG * l_reg)) / ACCUM_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # --- LOGGING ---
            curr_lr = scheduler.get_last_lr()[0]
            loss_val = loss.item() * ACCUM_STEPS
            
            # Log Train Metrics instantly
            wandb.log({
                "train/loss": loss_val,
                "train/loss_class": l_cls.item(),
                "train/loss_reg": l_reg.item(),
                "train/step": step + 1,
                "lr/head": scheduler.get_last_lr()[1]
            }, step=step+1)
            
            if (step + 1) % 10 == 0:
                 print(f"Step {step+1}/{STEPS} | Loss: {loss_val:.6f}")

            # --- VALIDATION ---
            if evaluator and (((step + 1) % EVAL_INTERVAL == 0) or step == 0):
                # Get metrics dict (mean_km, median_km, geo_score)
                val_metrics = evaluator.run(model)
                
                # Log Validation Metrics to WandB
                wandb.log(val_metrics, step=step+1)
                
                # Save Checkpoint
                torch.save(model.state_dict(), f"checkpoint_last.pth")

    torch.save(model.state_dict(), "geoguessr_cluster_1k_soft.pth")
    wandb.finish()
    print("Training Complete.")
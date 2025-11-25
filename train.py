import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import wandb

from model import GeoguessrModel
from clusters import ClusterManager
from dataset import create_dataloader, get_geo_transforms
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
SIGMA_KM = 300
LR_BACKBONE = 1e-6
LR_HEAD = 5e-4
LAMBDA_CLS = 1.0
LAMBDA_REG = 30.0
PROJECT_NAME = "neuroguessr"


# MULTITASK WEIGHTS
W_LOC = 1.0
W_CLIM = 0.2
W_LAND = 0.2
W_SOIL = 0.2
W_MONTH = 0.1

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    # 1. Initialize WandB
    wandb.init(
        project=PROJECT_NAME,
        config={
            "batch_size": MICRO_BATCH_SIZE * ACCUM_STEPS,
            "lr": LEARNING_RATE,
            "clusters": NUM_CLUSTERS,
            "steps": STEPS,
            "sigma": SIGMA_KM,
            "weights": [W_LOC, W_CLIM, W_LAND, W_SOIL, W_MONTH],
        }
    )

    # 2. Setup Clusters & Model
    cluster_manager = ClusterManager(n_clusters=NUM_CLUSTERS)
    try:
        cluster_centers, _loss_weights = cluster_manager.load(device=DEVICE)
    except FileNotFoundError:
        cluster_centers, _loss_weights = cluster_manager.generate()
        # _loss_weights = _loss_weights.to(DEVICE)

    soft_targets_matrix = cluster_manager.generate_soft_targets(sigma_km=SIGMA_KM)
    soft_targets_matrix = soft_targets_matrix.to(DEVICE)

    search_path = os.path.join(LOCAL_DATA_DIR, "train", "*.tar")
    tar_files = glob.glob(search_path)
    if not tar_files:
        tar_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.tar"))
        
    model = GeoguessrModel(num_classes=NUM_CLUSTERS).to(DEVICE)
    model = torch.compile(model)

    model_config = model.get_config()
    geo_transforms = get_geo_transforms(model_config, is_training=True)
    
    train_loader = create_dataloader(
        tar_files=tar_files,
        model_config=model_config,
        cluster_centers=cluster_centers,
        batch_size=MICRO_BATCH_SIZE,
        workers=NUM_WORKERS,
        mode='train',
        custom_transform=geo_transforms,
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
        else:
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


    crit_cluster = nn.CrossEntropyLoss()  # label_smoothing=0.1
    crit_climate = nn.CrossEntropyLoss(ignore_index=-1)
    crit_land = nn.CrossEntropyLoss(ignore_index=-1)
    crit_soil = nn.CrossEntropyLoss(ignore_index=-1)
    crit_month = nn.CrossEntropyLoss(ignore_index=-1)

    scaler = torch.amp.GradScaler('cuda')

    print("--- TRAINING START ---")
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        if step >= (STEPS * ACCUM_STEPS):
            break

        imgs = batch[0].to(DEVICE, non_blocking=True)
        lbl_cluster = batch[1].to(DEVICE, non_blocking=True)
        lbl_climate = batch[2].to(DEVICE, non_blocking=True)
        lbl_land = batch[3].to(DEVICE, non_blocking=True)
        lbl_soil = batch[4].to(DEVICE, non_blocking=True)
        lbl_month = batch[5].to(DEVICE, non_blocking=True)

        target_probs = soft_targets_matrix[lbl_cluster]
        
        with torch.amp.autocast('cuda'):
            out_cluster, out_climate, out_land, out_soil, out_month = model(imgs)

            loss_cluster = crit_cluster(out_cluster, target_probs)
            loss_climate = crit_climate(out_climate, lbl_climate)
            loss_land = crit_land(out_land, lbl_land)
            loss_soil = crit_soil(out_soil, lbl_soil)
            loss_month = crit_month(out_month, lbl_month)

            # weighted loss
            loss = (W_LOC * loss_cluster) + \
                   (W_CLIM * loss_climate) + \
                   (W_LAND * loss_land) + \
                   (W_SOIL * loss_soil) + \
                   (W_MONTH * loss_month)
            
            loss = loss / ACCUM_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # --- LOGGING ---
            curr_lr = scheduler.get_last_lr()[0]
            loss_val = loss.item() * ACCUM_STEPS
            
            wandb.log({
                "loss/total": loss.item() * ACCUM_STEPS,
                "loss/cluster": loss_cluster.item(),
                "loss/clim": loss_climate.item(),
                "loss/land": loss_land.item(),
                "loss/soil": loss_soil.item(),
                "loss/month": loss_month.item(),
                "lr/head": scheduler.get_last_lr()[1],
            }, step=step+1)
            
            if (step + 1) % 10 == 0:
                 print(f"Step {step+1}/{STEPS} | Loss: {loss_val:.6f}")

            # --- EVAL ---
            if evaluator and (((step + 1) % EVAL_INTERVAL == 0) or step == 0):
                metrics = evaluator.run(model)
                wandb.log(metrics, step=step+1)
                
                # Save Checkpoint
                torch.save(model.state_dict(), f"checkpoint_last.pth")

    torch.save(model.state_dict(), "geoguessr_multitask_1.pth")
    wandb.finish()
    print("Training Complete.")
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import wandb  # <--- NEW IMPORT

from model import GeoguessrModel
from clusters import ClusterManager
from dataset import create_dataloader
from evaluator import Evaluator 

# --- CONFIG ---
LOCAL_DATA_DIR = "./osv5m_local_data"
VAL_CACHE_DIR = "./val_cache"
MICRO_BATCH_SIZE = 256
ACCUM_STEPS = 1
LEARNING_RATE = 5e-4
STEPS = 1000
EVAL_INTERVAL = 100
DEVICE = "cuda"
NUM_WORKERS = 12
NUM_CLUSTERS = 1000
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

    search_path = os.path.join(LOCAL_DATA_DIR, "train", "*.tar")
    tar_files = glob.glob(search_path)
    if not tar_files:
        tar_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.tar"))
        
    model = GeoguessrModel(num_classes=NUM_CLUSTERS).to(DEVICE)
    model.head = torch.compile(model.head)
    
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
            n_clusters=NUM_CLUSTERS,
            batch_size=256, 
            device=DEVICE
        )
    else:
        print("WARNING: No validation cache found. Run prepare_val.py first!")

    optimizer = optim.AdamW(model.head.parameters(), lr=LEARNING_RATE)
    # Using OneCycleLR (Recommended for fast convergence)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=STEPS, pct_start=0.1
    )
    
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    scaler = torch.amp.GradScaler('cuda')

    print("--- TRAINING START ---")
    model.train()
    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels) / ACCUM_STEPS
        
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
                "train/lr": curr_lr,
                "train/step": step + 1
            }, step=step+1)
            
            if (step + 1) % 10 == 0:
                 print(f"Step {step+1}/{STEPS} | Loss: {loss_val:.6f}")

            # --- VALIDATION ---
            if evaluator and (step + 1) % EVAL_INTERVAL == 0:
                # Get metrics dict (mean_km, median_km, geo_score)
                val_metrics = evaluator.run(model)
                
                # Log Validation Metrics to WandB
                wandb.log(val_metrics, step=step+1)
                
                # Save Checkpoint
                torch.save(model.state_dict(), f"checkpoint_last.pth")

        if step >= (STEPS * ACCUM_STEPS):
            break

    torch.save(model.state_dict(), "geoguessr_final.pth")
    wandb.finish()
    print("Training Complete.")
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
from model import GeoguessrModel
from clusters import ClusterManager
from dataset import create_dataloader

LOCAL_DATA_DIR = "./osv5m_local_data"
MICRO_BATCH_SIZE = 512
ACCUM_STEPS = 2
LEARNING_RATE = 5e-4
STEPS = 400
DEVICE = "cuda"
NUM_WORKERS = 12
NUM_CLUSTERS = 200

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
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

    optimizer = optim.AdamW(model.head.parameters(), lr=LEARNING_RATE)
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
            
            if (step + 1) % 1 == 0:
                print(f"Step {step+1}/{STEPS} | Loss: {loss.item()*ACCUM_STEPS:.8f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if step >= (STEPS * ACCUM_STEPS):
            break

    torch.save(model.state_dict(), "geoguessr_model_1.pth")
    print("Saved model.")

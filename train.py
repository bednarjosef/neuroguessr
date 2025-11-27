import numpy as np
import torch
import torch.optim as optim

from model_clip import CLIPModel
from clusters import get_clusters
from dataset import create_dataloader
from evaluator import Evaluator
from loss import PIGEONLoss

import wandb

tar_directory = './osv5m_local/train'
val_directory = './osv5m_val_small_hf'

countries = [
    'AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY', 'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN',
]

CONFIG = {
    'device': 'cuda',
    'cache_dir': 'cache',
    'eval_interval': 500,
    'countries': countries,
    'num_countries': len(countries),
    'steps': 7500,
    # 'max_lr_backbone': 5e-6,
    'max_lr_head': 1e-4,
    'batch_size': 512,
    'accum_steps': 1,
    'clusters': 256,
    'tau_km': 150,
    'model': 'ViT-L/14@336px'
}


def train():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # init clusters
    print('Initializing clusters...')
    cluster_centers = get_clusters(CONFIG)

    # init model
    print('Initializing model...')
    model = CLIPModel(CONFIG).to(CONFIG['device'])
    model = torch.compile(model)

    train_transform = model.train_transform
    eval_transform = model.eval_transform

    # data loader
    print('Initializing data loader...')
    train_loader = create_dataloader(CONFIG, tar_directory, cluster_centers, train_transform, workers=12)

    # evaluator
    print('Initializing evaluator...')
    evaluator = Evaluator(CONFIG, cluster_centers, eval_transform, val_directory)

    # optimizer
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            # {"params": backbone_params, "lr": CONFIG['max_lr_backbone']},
            {"params": head_params, "lr": CONFIG['max_lr_head']},
        ],
        # weight_decay=0.01,
    )

    print('Initializing scheduler...')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["steps"],
    )

    print('Initializing PIGEON loss...')
    pigeon = PIGEONLoss(CONFIG, cluster_centers)
    scaler = torch.amp.GradScaler('cuda')

    # init wandb
    wandb.init(
        project="neuroguessr",
        config=CONFIG
    )

    # train
    print('Beginning training...')
    model.train()
    optimizer.zero_grad()
    best_median_km = float('inf')
    seen_clusters = np.zeros(CONFIG['clusters'], dtype=bool)

    for step, batch in enumerate(train_loader):
        if step >= (CONFIG['steps'] * CONFIG['accum_steps']):
            break
        images = batch[0].to(CONFIG['device'], non_blocking=True)
        true_clusters = batch[1].to(CONFIG['device'], non_blocking=True)
        true_lats = batch[2].to(CONFIG['device'], non_blocking=True)
        true_lons = batch[3].to(CONFIG['device'], non_blocking=True)

        seen_clusters[true_clusters.cpu().numpy()] = True

        with torch.amp.autocast('cuda'):
            logits = model(images)
            smooth_loss = pigeon.loss(CONFIG, logits, true_clusters, true_lats, true_lons)
            smooth_loss = smooth_loss / CONFIG['accum_steps']

        scaler.scale(smooth_loss).backward()

        # update weights, log
        if (step + 1) % CONFIG['accum_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # curr_lr_backbone = scheduler.get_last_lr()[0]
            curr_lr_head = scheduler.get_last_lr()[0]
            loss_val = smooth_loss.item() * CONFIG['accum_steps']

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                train_acc = (preds == true_clusters).float().mean().item() * 100
                # num_seen = int(seen_clusters.sum())
                
            
            wandb.log({
                "loss/total": loss_val,
                # "lr/backbone": curr_lr_backbone,
                "lr/head": curr_lr_head,
                "train/acc_top1": train_acc,
                # "train/seen_clusters": num_seen,
            }, step=step+1)

            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{CONFIG['steps'] * CONFIG['accum_steps']} | Loss: {loss_val:.6f} | Train acc: {train_acc:.2f}%")
            
            # evaluate
            if evaluator and (((step + 1) % CONFIG['eval_interval'] == 0) or step == 0):
                metrics = evaluator.run(model)
                wandb.log(metrics, step=step+1)

                current_median = metrics['val/top1_median_km']
                
                # save latest
                torch.save(model.state_dict(), f"checkpoint_last.pth")
                
                # save best
                if current_median < best_median_km:
                    best_median_km = current_median
                    torch.save(model.state_dict(), "geoguessr_best.pth")
                    print(f"🔥 New Best Model! Median Error: {best_median_km:.0f} km")
    

    torch.save(model.state_dict(), "geoguessr_fresh.pth")
    wandb.finish()
    print("Training Complete.")


if __name__ == '__main__':
    train()